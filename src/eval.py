"""
Evaluation engine.
Runs all conversations through all compression strategies and records results.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import litellm
import tiktoken
from pydantic import ValidationError

from src.compressors import all_compressors
from src.config import (
	CONTEXT_WINDOW_MARGIN,
	DATASET_PATH,
	EVAL_CONTEXT_WARNING_TOKENS,
	EVAL_MAX_WORKERS,
	GENERATION_CONTEXT_WINDOW,
	GENERATION_MODEL,
	GENERATION_MAX_TOKENS,
	MAX_RETRIES,
	RESULTS_PATH,
	TIKTOKEN_ENCODING,
)
from src.dataset import load_dataset
from src.models import Conversation, EvalRecord, JudgeVerdict, Turn

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
	datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


_GENERATION_SYSTEM_PROMPT = (
	"Answer the user question using only the supplied conversation context. "
	"Be concise and provide the final answer directly."
)

_NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")


def _turns_to_text(turns: list[Turn]) -> str:
	return "\n".join(f"{t.role}: {t.content}" for t in turns)


def _count_tokens(turns: list[Turn], encoder: tiktoken.Encoding) -> int:
	return len(encoder.encode(_turns_to_text(turns)))


def _serialize_hyperparams(hyperparams: dict[str, Any]) -> str:
	return json.dumps(hyperparams, sort_keys=True, separators=(",", ":"))


def _record_key(conversation_id: str, strategy: str, hyperparams: dict[str, Any]) -> tuple[str, str, str]:
	return (conversation_id, strategy, _serialize_hyperparams(hyperparams))


def _load_completed_keys(path: Path) -> set[tuple[str, str, str]]:
	completed: set[tuple[str, str, str]] = set()
	if not path.exists():
		return completed

	with path.open("r", encoding="utf-8") as fh:
		for lineno, line in enumerate(fh, start=1):
			line = line.strip()
			if not line:
				continue
			try:
				record = EvalRecord.model_validate_json(line)
				completed.add(
					_record_key(
						conversation_id=record.conversation_id,
						strategy=record.strategy,
						hyperparams=record.hyperparams,
					)
				)
			except ValidationError as exc:
				logger.warning("Skipping malformed result line %d: %s", lineno, exc.errors()[0]["msg"])
	return completed


def _litellm_call(
	*,
	model: str,
	messages: list[dict[str, str]],
	max_tokens: int,
	context_window: int,
	response_format: dict[str, str] | None = None,
) -> str:
	def _message_tokens(msgs: list[dict[str, str]], encoder: tiktoken.Encoding) -> int:
		text = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in msgs)
		return len(encoder.encode(text))

	encoder = tiktoken.get_encoding(TIKTOKEN_ENCODING)
	input_budget = context_window - max_tokens - CONTEXT_WINDOW_MARGIN
	if input_budget <= 0:
		raise RuntimeError(
			f"Invalid token budget: context_window={context_window} max_tokens={max_tokens} margin={CONTEXT_WINDOW_MARGIN}"
		)

	input_tokens = _message_tokens(messages, encoder)
	if input_tokens > input_budget:
		raise RuntimeError(
			"Input context exceeds generation budget without clipping: "
			f"input={input_tokens} budget={input_budget}. Increase GENERATION_CONTEXT_WINDOW."
		)

	for attempt in range(1, MAX_RETRIES + 1):
		try:
			kwargs: dict[str, Any] = {
				"model": model,
				"messages": messages,
				"max_tokens": max_tokens,
				"temperature": 0.1,
			}
			if response_format is not None:
				kwargs["response_format"] = response_format

			response = litellm.completion(**kwargs)
			content = response.choices[0].message.content
			if not content or not content.strip():
				raise RuntimeError("LLM returned empty content")
			return content.strip()
		except (litellm.RateLimitError, litellm.ServiceUnavailableError) as exc:
			if attempt == MAX_RETRIES:
				raise RuntimeError(f"Transient model error after {MAX_RETRIES} retries") from exc
			wait_seconds = 2 ** attempt
			logger.warning("Transient LLM error. Retry %d/%d in %ds", attempt, MAX_RETRIES, wait_seconds)
			time.sleep(wait_seconds)
		except litellm.APIError as exc:
			raise RuntimeError(f"LiteLLM API error: {exc}") from exc

	raise RuntimeError("Unexpected retry state")


def _generate_answer(messages: list[dict[str, str]]) -> str:
	request_messages = [{"role": "system", "content": _GENERATION_SYSTEM_PROMPT}, *messages]
	return _litellm_call(
		model=GENERATION_MODEL,
		messages=request_messages,
		max_tokens=GENERATION_MAX_TOKENS,
		context_window=GENERATION_CONTEXT_WINDOW,
	)


def _extract_numeric_values(text: str) -> list[Decimal]:
	values: list[Decimal] = []
	for match in _NUMBER_RE.findall(text):
		cleaned = match.replace("$", "").replace(",", "")
		try:
			values.append(Decimal(cleaned))
		except InvalidOperation:
			continue
	return values


def _numeric_verdict_against_reference(
	*,
	final_question: str,
	reference_answer: str,
	generated_answer: str,
	ground_truth_answer: str,
) -> JudgeVerdict:
	# Numeric-only evaluation: correctness is based purely on matching value, not surrounding context.
	target_values = _extract_numeric_values(ground_truth_answer)
	if not target_values:
		target_values = _extract_numeric_values(reference_answer)

	candidate_values = _extract_numeric_values(generated_answer)

	if not target_values:
		return JudgeVerdict(
			is_correct=False,
			reasoning=(
				"Could not parse a numeric target from ground truth or reference answer for question: "
				f"{final_question}"
			),
		)

	if not candidate_values:
		return JudgeVerdict(
			is_correct=False,
			reasoning="Candidate answer does not contain a parseable numeric value.",
		)

	tolerance = Decimal("0.01")
	for candidate in candidate_values:
		for target in target_values:
			if abs(candidate - target) <= tolerance:
				return JudgeVerdict(
					is_correct=True,
					reasoning=(
						"Numeric value match found between candidate and expected answer; "
						"context wording ignored by design."
					),
				)

	return JudgeVerdict(
		is_correct=False,
		reasoning=(
			f"No numeric match. Expected one of {target_values} but found {candidate_values}."
		),
	)


def _append_eval_record(path: Path, record: EvalRecord) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a", encoding="utf-8") as fh:
		fh.write(record.model_dump_json() + "\n")


def _build_messages(turns: list[Turn], final_question: str) -> list[dict[str, str]]:
	messages = [{"role": turn.role, "content": turn.content} for turn in turns]
	messages.append({"role": "user", "content": final_question})
	return messages


def _evaluate_combo(
	*,
	conversation_id: str,
	final_question: str,
	ground_truth_answer: str,
	original_tokens: int,
	compressor: Any,
	conversation: Conversation,
	reference_answer: str,
	encoder: tiktoken.Encoding,
) -> EvalRecord:
	compressed_turns = compressor.compress(conversation)
	compressed_tokens = _count_tokens(compressed_turns, encoder)
	reduction_ratio = (original_tokens - compressed_tokens) / original_tokens if original_tokens > 0 else 0.0

	generated_answer = _generate_answer(_build_messages(compressed_turns, final_question))
	verdict = _numeric_verdict_against_reference(
		final_question=final_question,
		reference_answer=reference_answer,
		generated_answer=generated_answer,
		ground_truth_answer=ground_truth_answer,
	)

	return EvalRecord(
		conversation_id=conversation_id,
		strategy=compressor.name,
		hyperparams=compressor.hyperparams,
		original_token_count=original_tokens,
		compressed_token_count=compressed_tokens,
		token_reduction_ratio=reduction_ratio,
		generated_answer=generated_answer,
		is_correct=verdict.is_correct,
		judge_reasoning=verdict.reasoning,
	)


def run_evaluation(
	*,
	strategy: str | None = None,
	max_conversations: int | None = None,
	max_combinations: int | None = None,
) -> None:
	conversations = load_dataset(DATASET_PATH)
	compressors = all_compressors()
	if strategy:
		compressors = [c for c in compressors if c.name == strategy]
		if not compressors:
			raise ValueError(f"Unknown strategy: {strategy}")

	if max_conversations is not None:
		conversations = conversations[:max_conversations]

	encoder = tiktoken.get_encoding(TIKTOKEN_ENCODING)
	completed = _load_completed_keys(RESULTS_PATH)

	total = len(conversations) * len(compressors)
	skipped = 0
	written = 0
	failed = 0

	logger.info(
		"Starting evaluation: %d conversations x %d compressors = %d combinations",
		len(conversations),
		len(compressors),
		total,
	)
	logger.info("Resume state: %d completed combinations already present", len(completed))
	if max_combinations is not None:
		logger.info("Run limit: max_combinations=%d", max_combinations)

	processed_this_run = 0
	stop_after_this_conversation = False

	for conversation in conversations:
		original_tokens = _count_tokens(conversation.history, encoder)
		if original_tokens >= EVAL_CONTEXT_WARNING_TOKENS:
			logger.info(
				"Long context detected for conversation=%s (%d tokens). Pipeline remains compatible; max_tokens caps apply to outputs only.",
				conversation.id,
				original_tokens,
			)
		reference_answer = _generate_answer(
			_build_messages(conversation.history, conversation.final_question)
		)
		pending: list[tuple[Any, tuple[str, str, str]]] = []

		for compressor in compressors:
			if max_combinations is not None and processed_this_run >= max_combinations:
				stop_after_this_conversation = True
				break

			key = _record_key(conversation.id, compressor.name, compressor.hyperparams)
			if key in completed:
				skipped += 1
				processed_this_run += 1
				continue

			pending.append((compressor, key))
			processed_this_run += 1

		if pending:
			max_workers = max(1, min(EVAL_MAX_WORKERS, len(pending)))
			with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
				future_map: dict[concurrent.futures.Future[EvalRecord], tuple[Any, tuple[str, str, str]]] = {}
				for compressor, key in pending:
					future = executor.submit(
						_evaluate_combo,
						conversation_id=conversation.id,
						final_question=conversation.final_question,
						ground_truth_answer=conversation.ground_truth_answer,
						original_tokens=original_tokens,
						compressor=compressor,
						conversation=conversation,
						reference_answer=reference_answer,
						encoder=encoder,
					)
					future_map[future] = (compressor, key)

				for future in concurrent.futures.as_completed(future_map):
					compressor, key = future_map[future]
					try:
						record = future.result()
						_append_eval_record(RESULTS_PATH, record)
						completed.add(key)
						written += 1
					except (RuntimeError, ValidationError, json.JSONDecodeError) as exc:
						failed += 1
						logger.warning(
							"Failed combo conversation=%s strategy=%s hyperparams=%s error=%s",
							conversation.id,
							compressor.name,
							compressor.hyperparams,
							exc,
						)

					processed = skipped + written + failed
					if processed % 10 == 0 or processed == total:
						logger.info(
							"Progress %d/%d | written=%d skipped=%d failed=%d",
							processed,
							total,
							written,
							skipped,
							failed,
						)

		if stop_after_this_conversation:
			logger.info("Reached max_combinations=%d. Stopping early.", max_combinations)
			logger.info(
				"Partial run complete: written=%d skipped=%d failed=%d processed=%d",
				written,
				skipped,
				failed,
				processed_this_run,
			)
			return

	logger.info("Evaluation complete: written=%d skipped=%d failed=%d total=%d", written, skipped, failed, total)
	logger.info("Results path: %s", RESULTS_PATH)


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run context compression evaluation")
	parser.add_argument(
		"--strategy",
		choices=["sliding_window", "summarization", "hybrid"],
		help="Evaluate only one strategy family",
	)
	parser.add_argument(
		"--max-conversations",
		type=int,
		help="Evaluate only the first N conversations",
	)
	parser.add_argument(
		"--max-combinations",
		type=int,
		help="Stop after processing N conversation-strategy combinations",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = _parse_args()
	run_evaluation(
		strategy=args.strategy,
		max_conversations=args.max_conversations,
		max_combinations=args.max_combinations,
	)
