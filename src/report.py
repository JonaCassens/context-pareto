"""
Reporting module.
Builds DataFrame, calculates metrics, plots Pareto frontier.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pydantic import ValidationError

from src.config import HYBRID_RECENT_WINDOW
from src.config import DATASET_PATH, PLOTS_DIR, RESULTS_DIR, RESULTS_PATH
from src.dataset import load_dataset
from src.models import Conversation, EvalRecord

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
	datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class CliffPoint:
	strategy: str
	hyperparams_key: str
	compression_ratio_pct: float
	accuracy_pct: float
	token_gain_pct: float
	accuracy_drop_pct: float


def _load_eval_records(path: Path) -> list[EvalRecord]:
	if not path.exists():
		logger.warning("Results file does not exist: %s", path)
		return []

	records: list[EvalRecord] = []
	with path.open("r", encoding="utf-8") as fh:
		for lineno, line in enumerate(fh, start=1):
			line = line.strip()
			if not line:
				continue
			try:
				records.append(EvalRecord.model_validate_json(line))
			except ValidationError as exc:
				logger.warning("Skipping malformed result line %d: %s", lineno, exc.errors()[0]["msg"])
	return records


def _build_dataframe(records: list[EvalRecord]) -> pd.DataFrame:
	if not records:
		return pd.DataFrame()

	rows = []
	for rec in records:
		row = rec.model_dump()
		row["hyperparams_key"] = ", ".join(
			f"{k}={v}" for k, v in sorted(rec.hyperparams.items(), key=lambda item: item[0])
		)
		row["strategy_with_params"] = f"{rec.strategy} ({row['hyperparams_key']})"
		rows.append(row)
	return pd.DataFrame(rows)


def _attach_dataset_metadata(df: pd.DataFrame) -> pd.DataFrame:
	if df.empty:
		return df

	conversations = load_dataset(DATASET_PATH)
	meta = {
		conv.id: {"domain": conv.domain, "turn_count": len(conv.history)}
		for conv in conversations
	}

	out = df.copy()
	out["domain"] = out["conversation_id"].map(lambda cid: meta.get(cid, {}).get("domain"))
	out["turn_count"] = out["conversation_id"].map(lambda cid: meta.get(cid, {}).get("turn_count"))
	return out


def _save_summary_tables(df: pd.DataFrame) -> None:
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)

	by_strategy = (
		df.groupby("strategy", dropna=False)
		.agg(
			runs=("is_correct", "count"),
			accuracy=("is_correct", "mean"),
			avg_token_reduction=("token_reduction_ratio", "mean"),
			std_token_reduction=("token_reduction_ratio", "std"),
			min_token_reduction=("token_reduction_ratio", "min"),
			max_token_reduction=("token_reduction_ratio", "max"),
		)
		.reset_index()
		.sort_values("accuracy", ascending=False)
	)
	by_strategy["accuracy"] = by_strategy["accuracy"] * 100.0
	by_strategy.to_csv(RESULTS_DIR / "summary_by_strategy.csv", index=False)

	by_strategy_params = (
		df.groupby(["strategy", "hyperparams_key"], dropna=False)
		.agg(
			runs=("is_correct", "count"),
			accuracy=("is_correct", "mean"),
			avg_token_reduction=("token_reduction_ratio", "mean"),
		)
		.reset_index()
		.sort_values(["strategy", "accuracy"], ascending=[True, False])
	)
	by_strategy_params["accuracy"] = by_strategy_params["accuracy"] * 100.0
	by_strategy_params.to_csv(RESULTS_DIR / "summary_by_strategy_hyperparams.csv", index=False)

	by_domain = (
		df.groupby(["domain", "strategy"], dropna=False)
		.agg(
			runs=("is_correct", "count"),
			accuracy=("is_correct", "mean"),
			avg_token_reduction=("token_reduction_ratio", "mean"),
		)
		.reset_index()
		.sort_values(["domain", "accuracy"], ascending=[True, False])
	)
	by_domain["accuracy"] = by_domain["accuracy"] * 100.0
	by_domain.to_csv(RESULTS_DIR / "summary_by_domain.csv", index=False)

	by_turn_count = (
		df.groupby(["turn_count", "strategy"], dropna=False)
		.agg(
			runs=("is_correct", "count"),
			accuracy=("is_correct", "mean"),
			avg_token_reduction=("token_reduction_ratio", "mean"),
		)
		.reset_index()
		.sort_values(["turn_count", "accuracy"], ascending=[True, False])
	)
	by_turn_count["accuracy"] = by_turn_count["accuracy"] * 100.0
	by_turn_count.to_csv(RESULTS_DIR / "summary_by_turn_count.csv", index=False)

	logger.info("Saved summary tables to %s", RESULTS_DIR)
	logger.info("Top strategies by accuracy:\n%s", by_strategy.to_string(index=False))


def _plot_accuracy_by_strategy(df: pd.DataFrame) -> None:
	plot_df = (
		df.groupby("strategy", dropna=False)
		.agg(accuracy=("is_correct", "mean"))
		.reset_index()
	)
	plot_df["accuracy"] = plot_df["accuracy"] * 100.0

	sns.set_theme(style="whitegrid")
	plt.figure(figsize=(8, 5))
	ax = sns.barplot(data=plot_df, x="strategy", y="accuracy", hue="strategy", legend=False)
	ax.set_xlabel("Strategy")
	ax.set_ylabel("Accuracy (%)")
	ax.set_title("Accuracy by Compression Strategy")
	ax.set_ylim(0, 100)
	plt.tight_layout()
	plt.savefig(PLOTS_DIR / "accuracy_by_strategy.png", dpi=200)
	plt.close()


def _plot_compression_distribution(df: pd.DataFrame) -> None:
	sns.set_theme(style="whitegrid")
	plt.figure(figsize=(9, 5))
	ax = sns.histplot(
		data=df,
		x="token_reduction_ratio",
		hue="strategy",
		element="step",
		bins=25,
		stat="count",
		common_norm=False,
	)
	ax.set_xlabel("Token Reduction Ratio")
	ax.set_ylabel("Count")
	ax.set_title("Compression Ratio Distribution by Strategy")
	plt.tight_layout()
	plt.savefig(PLOTS_DIR / "compression_ratio_distribution.png", dpi=200)
	plt.close()


def _plot_tradeoff_scatter(df: pd.DataFrame) -> None:
	agg = (
		df.groupby(["strategy", "hyperparams_key"], dropna=False)
		.agg(
			accuracy=("is_correct", "mean"),
			avg_token_reduction=("token_reduction_ratio", "mean"),
		)
		.reset_index()
	)
	agg["accuracy"] = agg["accuracy"] * 100.0

	sns.set_theme(style="whitegrid")
	plt.figure(figsize=(10, 6))
	ax = sns.scatterplot(
		data=agg,
		x="avg_token_reduction",
		y="accuracy",
		hue="strategy",
		style="strategy",
		s=120,
	)
	for _, row in agg.iterrows():
		ax.text(
			row["avg_token_reduction"] + 0.002,
			row["accuracy"] + 0.2,
			row["hyperparams_key"],
			fontsize=8,
		)
	ax.set_xlabel("Average Token Reduction Ratio")
	ax.set_ylabel("Accuracy (%)")
	ax.set_title("Accuracy vs Compression Tradeoff")
	plt.tight_layout()
	plt.savefig(PLOTS_DIR / "tradeoff_accuracy_vs_reduction.png", dpi=200)
	plt.close()


def _strategy_hyperparam_table(df: pd.DataFrame) -> pd.DataFrame:
	result = (
		df.groupby(["strategy", "hyperparams_key"], dropna=False)
		.agg(
			runs=("is_correct", "count"),
			accuracy=("is_correct", "mean"),
			avg_token_reduction=("token_reduction_ratio", "mean"),
		)
		.reset_index()
	)
	result["accuracy_pct"] = result["accuracy"] * 100.0
	result["avg_token_reduction_pct"] = result["avg_token_reduction"] * 100.0
	return result.sort_values(["strategy", "avg_token_reduction_pct"])


def _find_cliff_points(summary: pd.DataFrame) -> list[CliffPoint]:
	cliffs: list[CliffPoint] = []
	for strategy, group in summary.groupby("strategy", dropna=False):
		g = group.sort_values("avg_token_reduction_pct").reset_index(drop=True)
		for i in range(1, len(g)):
			token_gain = g.loc[i, "avg_token_reduction_pct"] - g.loc[i - 1, "avg_token_reduction_pct"]
			accuracy_drop = g.loc[i - 1, "accuracy_pct"] - g.loc[i, "accuracy_pct"]
			if token_gain > 0 and accuracy_drop > token_gain:
				cliffs.append(
					CliffPoint(
						strategy=str(strategy),
						hyperparams_key=str(g.loc[i, "hyperparams_key"]),
						compression_ratio_pct=float(g.loc[i, "avg_token_reduction_pct"]),
						accuracy_pct=float(g.loc[i, "accuracy_pct"]),
						token_gain_pct=float(token_gain),
						accuracy_drop_pct=float(accuracy_drop),
					)
				)
				break
	return cliffs


def _choose_recommended_strategy(summary: pd.DataFrame) -> tuple[str, pd.DataFrame]:
	strategy_rollup = (
		summary.groupby("strategy", dropna=False)
		.agg(
			accuracy_pct=("accuracy_pct", "mean"),
			avg_token_reduction_pct=("avg_token_reduction_pct", "mean"),
		)
		.reset_index()
	)
	strategy_rollup = strategy_rollup.sort_values(
		["accuracy_pct", "avg_token_reduction_pct"],
		ascending=[False, False],
	)
	return str(strategy_rollup.iloc[0]["strategy"]), strategy_rollup


def _parse_hyperparams_key(hyperparams_key: str) -> dict[str, str]:
	parts: dict[str, str] = {}
	for item in hyperparams_key.split(", "):
		if "=" not in item:
			continue
		k, v = item.split("=", 1)
		parts[k] = v
	return parts


def _infer_kept_indices(conversation: Conversation, strategy: str, hyperparams_key: str) -> list[int]:
	parts = _parse_hyperparams_key(hyperparams_key)
	history = conversation.history

	if strategy == "sliding_window":
		window = int(parts.get("window", "1"))
		start = max(0, len(history) - window)
		return list(range(start, len(history)))

	if strategy == "summarization":
		# Summarization keeps a synthetic summary turn plus the last original turn.
		# Only the last original turn index can be mapped directly.
		return [len(history) - 1] if history else []

	if strategy == "hybrid":
		k = int(parts.get("k", "1"))
		recent_window = int(parts.get("recent_window", str(HYBRID_RECENT_WINDOW)))
		recent_n = min(recent_window, len(history))
		recent_start = len(history) - recent_n
		older = history[:recent_start]

		def _priority(turn: object, idx: int, total: int) -> tuple[float, int]:
			text = turn.content
			has_number = bool(re.search(r"\\d", text))
			has_unit = bool(re.search(r"\\b(?:usd|dollars?|%|kg|km|hours?|days?|weeks?|months?)\\b", text.lower()))
			has_id = bool(re.search(r"\\b[A-Z]{2,}-?\\d{1,}\\b", text))
			length_bonus = min(len(text) / 300.0, 1.0)
			first_third = max(total // 3, 1)
			last_third_start = total - first_third
			boundary_bonus = 1.0 if idx < first_third or idx >= last_third_start else 0.0
			score = (
				(2.0 if has_number else 0.0)
				+ (1.5 if has_unit else 0.0)
				+ (1.0 if has_id else 0.0)
				+ length_bonus
				+ boundary_bonus
			)
			return (score, idx)

		anchor_count = min(k, len(older))
		chosen_older: list[int] = []
		if anchor_count > 0:
			priorities = [(_priority(turn, idx, len(history)), idx) for idx, turn in enumerate(older)]
			priorities.sort(reverse=True)
			chosen_older = sorted(idx for _, idx in priorities[:anchor_count])

		return chosen_older + list(range(recent_start, len(history)))

	raise ValueError(f"Unsupported strategy for failure analysis: {strategy}")


def _build_failure_analysis(
	*,
	df: pd.DataFrame,
	conversations: list[Conversation],
	recommended_strategy: str,
) -> str:
	conv_map = {c.id: c for c in conversations}
	wrong = df[(df["strategy"] == recommended_strategy) & (~df["is_correct"])]
	if wrong.empty:
		return "No incorrect example found for the recommended strategy in current results."

	row = wrong.sort_values("token_reduction_ratio", ascending=False).iloc[0]
	conversation = conv_map.get(row["conversation_id"])
	if conversation is None:
		return "Unable to construct failure case because conversation metadata was missing."

	kept_indices = _infer_kept_indices(
		conversation,
		str(row["strategy"]),
		str(row["hyperparams_key"]),
	)
	missing_critical = [
		i for i in conversation.critical_turn_indices
		if i not in kept_indices
	]

	mechanism = (
		"Compression removed at least one critical early/late anchor turn required for the final derivation."
		if missing_critical
		else "Compression retained critical indices but likely lost detail fidelity during summarization or semantic selection."
	)

	return (
		f"Conversation ID: {conversation.id}\n"
		f"Strategy: {row['strategy']} ({row['hyperparams_key']})\n"
		f"Token reduction ratio: {row['token_reduction_ratio']:.3f}\n"
		f"Ground truth: {conversation.ground_truth_answer}\n"
		f"Model answer: {row['generated_answer']}\n"
		f"Critical turn indices: {conversation.critical_turn_indices}\n"
		f"Kept turn indices: {kept_indices}\n"
		f"Missing critical indices: {missing_critical}\n"
		f"Mechanism: {mechanism}\n"
		f"Judge reasoning: {row['judge_reasoning']}"
	)


def _write_assignment_findings(
	*,
	summary: pd.DataFrame,
	cliffs: list[CliffPoint],
	strategy_rollup: pd.DataFrame,
	recommended_strategy: str,
	failure_case: str,
) -> None:
	path = RESULTS_DIR / "assignment3_findings.md"
	lines: list[str] = []

	def _table_text(frame: pd.DataFrame) -> str:
		try:
			return frame.to_markdown(index=False)
		except Exception:
			return frame.to_string(index=False)

	lines.append("# Assignment 3 Findings")
	lines.append("")
	lines.append("## Token-Quality Tradeoff Summary")
	lines.append("")
	lines.append(_table_text(summary[["strategy", "hyperparams_key", "runs", "accuracy_pct", "avg_token_reduction_pct"]]))
	lines.append("")
	lines.append("## Cliff Points")
	if cliffs:
		for cliff in cliffs:
			lines.append(
				f"- {cliff.strategy}: cliff near {cliff.hyperparams_key} at {cliff.compression_ratio_pct:.2f}% reduction and {cliff.accuracy_pct:.2f}% accuracy "
				f"(token gain {cliff.token_gain_pct:.2f}pp, accuracy drop {cliff.accuracy_drop_pct:.2f}pp)."
			)
	else:
		lines.append("- No cliff detected by the configured rule (accuracy drop > token gain between adjacent compression levels).")
	lines.append("")
	lines.append("## Recommended Strategy")
	lines.append("")
	lines.append(f"Recommended strategy: {recommended_strategy}")
	lines.append("")
	lines.append("Rationale: highest mean accuracy across its hyperparameter settings, with token reduction used as tie-breaker.")
	lines.append("")
	lines.append("Strategy rollup:")
	lines.append(_table_text(strategy_rollup))
	lines.append("")
	lines.append("## One Concrete Failure Case")
	lines.append("")
	lines.append("```text")
	lines.append(failure_case)
	lines.append("```")

	path.write_text("\n".join(lines), encoding="utf-8")
	logger.info("Wrote assignment findings: %s", path)


def run_report() -> None:
	PLOTS_DIR.mkdir(parents=True, exist_ok=True)

	records = _load_eval_records(RESULTS_PATH)
	if not records:
		logger.warning("No eval records found at %s. Run src.eval first.", RESULTS_PATH)
		return

	df = _build_dataframe(records)
	conversations = load_dataset(DATASET_PATH)
	df = _attach_dataset_metadata(df)

	logger.info("Loaded %d eval records", len(df))

	_save_summary_tables(df)
	_plot_accuracy_by_strategy(df)
	_plot_compression_distribution(df)
	_plot_tradeoff_scatter(df)

	summary = _strategy_hyperparam_table(df)
	cliffs = _find_cliff_points(summary)
	recommended_strategy, strategy_rollup = _choose_recommended_strategy(summary)
	failure_case = _build_failure_analysis(
		df=df,
		conversations=conversations,
		recommended_strategy=recommended_strategy,
	)
	_write_assignment_findings(
		summary=summary,
		cliffs=cliffs,
		strategy_rollup=strategy_rollup,
		recommended_strategy=recommended_strategy,
		failure_case=failure_case,
	)

	logger.info("Saved plots to %s", PLOTS_DIR)


if __name__ == "__main__":
	run_report()
