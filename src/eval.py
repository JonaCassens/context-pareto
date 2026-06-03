"""
Evaluation engine.
Runs all conversations through all compression strategies and records results.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import litellm
import tiktoken
from pydantic import ValidationError

from src.compressors import all_compressors
from src.config import (
	DATASET_PATH,
	GENERATION_MODEL,
	JUDGE_MODEL,
	MAX_RETRIES,
	RESULTS_PATH,
	TIKTOKEN_ENCODING,
)
from src.dataset import load_dataset
from src.models import EvalRecord, JudgeVerdict, Turn

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

_JUDGE_SYSTEM_PROMPT = (
	"You are an evaluator. Compare the candidate answer against the ground truth. "
	"Return strict JSON with keys: is_correct (boolean) and reasoning (string). "
	"Mark is_correct true only when semantically equivalent."
)


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
	response_format: dict[str, str] | None = None,
) -> str:
	for attempt in range(1, MAX_RETRIES + 1):
		try:
			kwargs: dict[str, Any] = {
				"model": model,
				"messages": messages,
				"max_tokens": 32768,
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
	return _litellm_call(model=GENERATION_MODEL, messages=request_messages)


def _judge_answer(final_question: str, generated_answer: str, ground_truth_answer: str) -> JudgeVerdict:
	judge_user = (
		"Question:\n"
		f"{final_question}\n\n"
		"Ground truth answer:\n"
		f"{ground_truth_answer}\n\n"
		"Candidate answer:\n"
		f"{generated_answer}\n\n"
		"Return JSON only."
	)
	raw = _litellm_call(
		model=JUDGE_MODEL,
		messages=[
			{"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
			{"role": "user", "content": judge_user},
		],
		response_format={"type": "json_object"},
	)
	return JudgeVerdict.model_validate(json.loads(raw))


def _judge_against_reference(
	*,
	final_question: str,
	reference_answer: str,
	generated_answer: str,
	ground_truth_answer: str,
) -> JudgeVerdict:
	judge_user = (
		"Question:\n"
		f"{final_question}\n\n"
		"Reference answer from full uncompressed context:\n"
		f"{reference_answer}\n\n"
		"Ground truth answer:\n"
		f"{ground_truth_answer}\n\n"
		"Candidate answer:\n"
		f"{generated_answer}\n\n"
		"Decide correctness based primarily on semantic equivalence to the reference answer. "
		"Use the ground truth to disambiguate format differences. Return JSON only."
	)
	raw = _litellm_call(
		model=JUDGE_MODEL,
		messages=[
			{"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
			{"role": "user", "content": judge_user},
		],
		response_format={"type": "json_object"},
	)
	return JudgeVerdict.model_validate(json.loads(raw))


def _append_eval_record(path: Path, record: EvalRecord) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a", encoding="utf-8") as fh:
		fh.write(record.model_dump_json() + "\n")


def _build_messages(turns: list[Turn], final_question: str) -> list[dict[str, str]]:
	messages = [{"role": turn.role, "content": turn.content} for turn in turns]
	messages.append({"role": "user", "content": final_question})
	return messages


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

	for conversation in conversations:
		original_tokens = _count_tokens(conversation.history, encoder)
		reference_answer = _generate_answer(
			_build_messages(conversation.history, conversation.final_question)
		)

		for compressor in compressors:
			if max_combinations is not None and processed_this_run >= max_combinations:
				logger.info("Reached max_combinations=%d. Stopping early.", max_combinations)
				logger.info(
					"Partial run complete: written=%d skipped=%d failed=%d processed=%d",
					written,
					skipped,
					failed,
					processed_this_run,
				)
				return

			key = _record_key(conversation.id, compressor.name, compressor.hyperparams)
			if key in completed:
				skipped += 1
				processed_this_run += 1
				continue

			try:
				compressed_turns = compressor.compress(conversation)
				compressed_tokens = _count_tokens(compressed_turns, encoder)
				reduction_ratio = (
					(original_tokens - compressed_tokens) / original_tokens if original_tokens > 0 else 0.0
				)

				generated_answer = _generate_answer(
					_build_messages(compressed_turns, conversation.final_question)
				)
				verdict = _judge_against_reference(
					final_question=conversation.final_question,
					reference_answer=reference_answer,
					generated_answer=generated_answer,
					ground_truth_answer=conversation.ground_truth_answer,
				)

				record = EvalRecord(
					conversation_id=conversation.id,
					strategy=compressor.name,
					hyperparams=compressor.hyperparams,
					original_token_count=original_tokens,
					compressed_token_count=compressed_tokens,
					token_reduction_ratio=reduction_ratio,
					generated_answer=generated_answer,
					is_correct=verdict.is_correct,
					judge_reasoning=verdict.reasoning,
				)
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

			processed_this_run += 1
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
