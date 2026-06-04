"""
Dataset generator for the context-compression evaluation pipeline.

Default mode is LLM generation with strict post-generation validation.
Extraction from an existing JSONL file is optional and opt-in.

Usage:
    python -m src.dataset
    python -m src.dataset --n 36 --out data/conversations.jsonl
    python -m src.dataset --extract-from-source --source data/conversations.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import time

import litellm
import tiktoken
from pydantic import ValidationError

from src.config import (
    BATCH_SIZE,
    CONTEXT_WINDOW_MARGIN,
    DATASET_CHECKPOINT_EVERY,
    DATASET_PATH,
    DATASET_GENERATION_CONTEXT_WINDOW,
    DATASET_GENERATION_MAX_TOKENS,
    DATASET_SIZE,
    GENERATION_MODEL,
    GENERATION_DOMAINS,
    LLM_REQUEST_TIMEOUT_SECONDS,
    MAX_RETRIES,
    RETRY_BASE_DELAY_SECONDS,
    RETRY_JITTER_SECONDS,
    RETRY_MAX_DELAY_SECONDS,
    TURN_COUNTS,
)
from src.models import Conversation, GeneratedDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")
_DATASET_TEMPERATURE = 0.4

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a synthetic-data engineer building an evaluation dataset for LLM \
context-compression research. Your output MUST be valid JSON that matches the \
schema provided exactly — no extra commentary, no markdown fences.\
"""

_BATCH_PROMPT_TEMPLATE = """\
Generate exactly {n} synthetic multi-turn conversations in the domain of \
"{domain}". Each conversation must satisfy ALL of the following hard rules:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRUCTURAL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Each conversation must have EXACTLY {target_turns} history turns. Roles
    alternate user / assistant, starting with user.
2. The conversation ends with a `final_question` posed by the user that is
   NOT part of `history`.
3. `ground_truth_answer` is the single, unambiguous correct answer to
   `final_question`. For numeric answers include the number AND unit.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEPENDENCY CHAIN RULES  ← most important
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. `critical_turn_indices` MUST contain exactly two indices.
5. One critical index MUST be in turns 0-2 (inclusive): first required anchor.
6. One critical index MUST be in turns N-3 to N-1 (inclusive), where N is
    len(history): second required anchor.
7. Both anchors must be necessary to compute the final answer. The final
    question must be unanswerable with only one anchor.
8. Include misleading values in non-critical turns. Misleading values are also
    allowed in critical turns, but they must not remove two-anchor necessity.
9. The `final_question` MUST NOT contain numeric literals.
10. Do NOT place the final computed answer value in any history turn.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUALITY RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
12. Use distinct named entities (people, products, projects) across all {n}
   conversations — do not reuse "Alice", "Widget A", or any other name more
   than once across the batch.
13. Vary dependency style across the batch: multiplication, percentage
    adjustment, lookup with modifier, and unit conversion.
14. Each `id` must be a unique UUID v4 string.
15. Set `domain` to "{domain}".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKED EXAMPLE (do NOT copy this — only use it as a structural template)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
history (6 turns):
  [0] user: "We received a shipment of 340 units of part #K-77 this morning."
  [1] assistant: "Got it — 340 units of K-77 logged."
  [2] user: "Also, the warehouse team flagged that bay 4 needs repairs. Cost?"
  [3] assistant: "Bay 4 repair estimate is $2,400 for labor and $800 for parts."
  [4] user: "Separately, what's the standard unit cost for K-77 these days?"
  [5] assistant: "Current unit cost for K-77 is $18.50."

final_question: "What is the total inventory value of the K-77 units we received today?"
ground_truth_answer: "$6,290.00"   (340 × $18.50)
critical_turn_indices: [0, 5]      ← turn 0 has the quantity; turn 5 has the price

The bay-4 repair info (turns 2–3) is an intentional distractor.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Now generate {n} NEW conversations following all rules above.
Return ONLY a JSON object matching this schema:

{{
  "conversations": [
    {{
      "id": "<uuid4>",
      "domain": "{domain}",
      "history": [
        {{"role": "user" | "assistant", "content": "<text>"}}
      ],
      "final_question": "<question text>",
      "ground_truth_answer": "<exact answer with units if numeric>",
      "critical_turn_indices": [<int>, ...]
    }}
  ]
}}
"""
def build_batch_prompt(
    n: int,
    domain: str,
    target_turns: int,
) -> str:
    """Return the user-turn prompt for generating ``n`` conversations in ``domain``."""
    return _BATCH_PROMPT_TEMPLATE.format(
        n=n,
        domain=domain,
        target_turns=target_turns,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_dependency_spread(conv: Conversation) -> bool:
    """
    Secondary check that one critical anchor exists in turns 0-2 and one
    critical anchor exists in turns N-3..N-1.
    """
    n = len(conv.history)
    early_window = set(range(0, min(3, n)))
    late_window = set(range(max(0, n - 3), n))
    first, second = conv.critical_turn_indices
    early_count = int(first in early_window) + int(second in early_window)
    late_count = int(first in late_window) + int(second in late_window)
    return early_count == 1 and late_count == 1


def _extract_numeric_tokens(text: str) -> set[str]:
    """Extract normalized numeric tokens from text for lightweight leakage checks."""
    out: set[str] = set()
    for raw in _NUMBER_RE.findall(text):
        token = raw.replace("$", "").replace(",", "").strip()
        if token:
            out.add(token)
    return out


def _normalize_for_match(text: str) -> str:
    """Normalize text for simple containment checks."""
    lowered = text.lower().strip()
    lowered = lowered.replace("$", "")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def _answer_leaked_verbatim(answer: str, turn_text: str) -> bool:
    """Return True only when the full answer appears verbatim in a turn."""
    normalized_answer = _normalize_for_match(answer)
    if not normalized_answer:
        return False
    return normalized_answer in _normalize_for_match(turn_text)


def validate_required_dependency(conv: Conversation) -> tuple[bool, str]:
    """
    Enforce simplified two-anchor requirements for numeric tasks.

    Rules:
    1) final_question must not contain numeric literals (prevents leaking operands).
    2) Exactly one critical anchor must be early (0-2) and one late (N-3..N-1).
    3) At least two distinct numeric operands must appear across critical turns.
    4) Keep misleading values possible while preserving two-anchor necessity.
    """
    question_numbers = _extract_numeric_tokens(conv.final_question)
    if question_numbers:
        return (False, "final_question leaks numeric literals; answer can become single-anchor solvable")

    if not validate_dependency_spread(conv):
        return (False, "critical_turn_indices must include one early (0-2) and one late (N-3..N-1) index")

    critical_number_sets: dict[int, set[str]] = {
        idx: _extract_numeric_tokens(conv.history[idx].content)
        for idx in conv.critical_turn_indices
    }
    combined_critical_numbers: set[str] = set().union(*critical_number_sets.values())

    if len(combined_critical_numbers) < 2:
        return (
            False,
            "critical turns do not provide at least two distinct numeric operands",
        )

    return (True, "ok")


def evaluate_conversation_quality(conv: Conversation) -> tuple[bool, list[str]]:
    """
    Evaluate whether a generated conversation is admissible for this benchmark.

    Returns ``(is_valid, reasons)`` where ``reasons`` contains all failed checks.
    """
    reasons: list[str] = []

    # Enforce alternating user/assistant turns starting with user.
    expected_roles = ["user" if i % 2 == 0 else "assistant" for i in range(len(conv.history))]
    actual_roles = [turn.role for turn in conv.history]
    if actual_roles != expected_roles:
        reasons.append("history roles must alternate user/assistant starting with user")

    dep_ok, dep_reason = validate_required_dependency(conv)
    if not dep_ok:
        reasons.append(dep_reason)

    if not validate_dependency_spread(conv):
        reasons.append("critical_turn_indices must include one early and one late anchor")

    if not _extract_numeric_tokens(conv.ground_truth_answer):
        reasons.append("ground_truth_answer must contain at least one numeric value")

    question_numbers = _extract_numeric_tokens(conv.final_question)
    if question_numbers:
        reasons.append("final_question must not contain numeric literals")

    # Avoid exact final-answer leakage in conversation history.
    for idx, turn in enumerate(conv.history):
        if _answer_leaked_verbatim(conv.ground_truth_answer, turn.content):
            reasons.append(f"history turn {idx} leaks final answer value")
            break

    return (len(reasons) == 0, reasons)


def _sanitize_question_no_numbers(question: str) -> str:
    """Remove numeric literals from a question while preserving readability."""
    sanitized = _NUMBER_RE.sub("", question)
    sanitized = re.sub(r"\s+", " ", sanitized).strip(" ,.;:")
    if sanitized and not sanitized.endswith("?"):
        sanitized = f"{sanitized}?"
    return sanitized


def validate_or_repair_conversation(conv: Conversation) -> tuple[Conversation | None, list[str]]:
    """
    Validate a conversation and apply lightweight deterministic repair when safe.

    Current repair policy:
    - If only question-number leakage is present, remove numeric literals from
      final_question and re-validate.
    """
    is_valid, reasons = evaluate_conversation_quality(conv)
    if is_valid:
        return conv, []

    leakage_markers = {
        "final_question must not contain numeric literals",
        "final_question leaks numeric literals; answer can become single-anchor solvable",
    }
    non_leakage_reasons = [r for r in reasons if r not in leakage_markers]

    if not non_leakage_reasons:
        fixed_question = _sanitize_question_no_numbers(conv.final_question)
        if fixed_question:
            repaired = conv.model_copy(update={"final_question": fixed_question})
            repaired_ok, repaired_reasons = evaluate_conversation_quality(repaired)
            if repaired_ok:
                return repaired, []
            return None, repaired_reasons

    return None, reasons


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def _call_litellm(
    prompt: str,
    model: str,
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    """
    Call litellm with JSON response format and return the parsed dict.

    Retries on rate-limit (429) errors with exponential backoff.
    Raises RuntimeError on API or JSON-parse failures after all retries.
    """
    encoder = tiktoken.get_encoding("cl100k_base")
    input_budget = (
        DATASET_GENERATION_CONTEXT_WINDOW
        - DATASET_GENERATION_MAX_TOKENS
        - CONTEXT_WINDOW_MARGIN
    )
    if input_budget <= 0:
        raise RuntimeError(
            "Invalid dataset-generation token budget. Increase DATASET_GENERATION_CONTEXT_WINDOW "
            "or reduce DATASET_GENERATION_MAX_TOKENS/CONTEXT_WINDOW_MARGIN."
        )

    prompt_tokens = len(encoder.encode(prompt))
    if prompt_tokens > input_budget:
        raise RuntimeError(
            "Dataset-generation prompt exceeds context budget without clipping: "
            f"prompt={prompt_tokens} budget={input_budget}. Increase DATASET_GENERATION_CONTEXT_WINDOW."
        )

    def _parse_llm_json(raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start:end + 1])

        raise json.JSONDecodeError("No JSON object found", cleaned, 0)

    for attempt in range(1, max_retries + 1):
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=_DATASET_TEMPERATURE,
                max_tokens=DATASET_GENERATION_MAX_TOKENS,
                timeout=LLM_REQUEST_TIMEOUT_SECONDS,
            )
            raw = response.choices[0].message.content
            try:
                return _parse_llm_json(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"LLM returned non-JSON content: {raw[:200]}") from exc

        except (litellm.RateLimitError, litellm.ServiceUnavailableError) as exc:
            wait = min(RETRY_MAX_DELAY_SECONDS, RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)))
            wait = wait + random.uniform(0.0, RETRY_JITTER_SECONDS)
            logger.warning(
                "Transient API error (attempt %d/%d). Waiting %.1fs before retry. %s",
                attempt, max_retries, wait, exc,
            )
            if attempt == max_retries:
                raise RuntimeError(f"Transient API error persisted after {max_retries} retries.") from exc
            time.sleep(wait)

        except litellm.APIError as exc:
            raise RuntimeError(f"LiteLLM API error: {exc}") from exc


# ---------------------------------------------------------------------------
# Batch generation with retry logic
# ---------------------------------------------------------------------------


def generate_batch(
    domain: str,
    n: int = BATCH_SIZE,
    model: str = GENERATION_MODEL,
    max_retries: int = MAX_RETRIES,
    target_turns: int = 8,
) -> list[Conversation]:
    """
    Generate ``n`` conversations for the given ``domain``.

    Retries individual conversations that fail Pydantic validation (e.g.,
    dependency-spread rule) or have the wrong turn count up to ``max_retries``
    times per conversation.  Returns however many valid conversations were produced.
    """
    logger.info(
        "Generating batch: domain=%r n=%d turns=%d model=%s",
        domain,
        n,
        target_turns,
        model,
    )
    prompt = build_batch_prompt(
        n=n,
        domain=domain,
        target_turns=target_turns,
    )

    raw_dict: dict[str, Any] = {}
    for attempt in range(1, max_retries + 1):
        try:
            raw_dict = _call_litellm(prompt, model)
            break
        except RuntimeError as exc:
            logger.warning("Attempt %d/%d failed: %s", attempt, max_retries, exc)
            if attempt == max_retries:
                logger.error("All attempts exhausted for domain %r. Skipping batch.", domain)
                return []

    raw_conversations: list[dict] = raw_dict.get("conversations", [])
    valid: list[Conversation] = []
    invalid_raws: list[dict] = []

    for raw_conv in raw_conversations:
        # Ensure a UUID is present even if the LLM omitted it.
        if not raw_conv.get("id"):
            raw_conv["id"] = str(uuid.uuid4())
        try:
            conv = Conversation.model_validate(raw_conv)
            if len(conv.history) != target_turns:
                logger.warning(
                    "  ✗ Wrong turn count: expected %d, got %d",
                    target_turns, len(conv.history),
                )
                invalid_raws.append(raw_conv)
            else:
                repaired_conv, reasons = validate_or_repair_conversation(conv)
                if repaired_conv is None:
                    logger.warning("  ✗ Quality checks failed: %s", "; ".join(reasons))
                    invalid_raws.append(raw_conv)
                    continue

                valid.append(repaired_conv)
                logger.debug("  ✓ %s  (critical=%s)", repaired_conv.id, repaired_conv.critical_turn_indices)
        except ValidationError as exc:
            logger.warning("  ✗ Validation failed for a conversation: %s", exc.errors()[0]["msg"])
            invalid_raws.append(raw_conv)

    # Retry each invalid conversation individually.
    if invalid_raws:
        logger.info("  Re-generating %d failed conversations …", len(invalid_raws))
        retry_prompt = build_batch_prompt(
            n=len(invalid_raws),
            domain=domain,
            target_turns=target_turns,
        )
        for attempt in range(1, max_retries + 1):
            try:
                retry_dict = _call_litellm(retry_prompt, model)
                break
            except RuntimeError:
                if attempt == max_retries:
                    logger.error("Retry exhausted; accepting partial batch.")
                    return valid
        for raw_conv in retry_dict.get("conversations", []):
            if not raw_conv.get("id"):
                raw_conv["id"] = str(uuid.uuid4())
            try:
                conv = Conversation.model_validate(raw_conv)
                repaired_conv, _ = validate_or_repair_conversation(conv)
                if repaired_conv is not None:
                    valid.append(repaired_conv)
            except ValidationError:
                pass  # Accept partial results after retry.

    logger.info("  Batch complete: %d/%d valid conversations.", len(valid), n)
    return valid


def generate_conversations(
    total: int = DATASET_SIZE,
    model: str = GENERATION_MODEL,
    out_path: Path | None = None,
    checkpoint_every: int = DATASET_CHECKPOINT_EVERY,
) -> list[Conversation]:
    """
    Generate conversations across configured domains and turn counts.

    Turn count is constrained to configured values in TURN_COUNTS (6-10).
    """
    all_conversations: list[Conversation] = []
    occupied_ids: set[str] = set()

    combinations: list[tuple[str, int]] = [
        (domain, turns) for domain in GENERATION_DOMAINS for turns in TURN_COUNTS
    ]
    if not combinations:
        logger.error("No generation combinations configured.")
        return []

    slot_idx = 0
    while len(all_conversations) < total:
        domain, target_turns = combinations[slot_idx % len(combinations)]
        slot_idx += 1

        conv: Conversation | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            batch = generate_batch(
                domain=domain,
                n=1,
                model=model,
                target_turns=target_turns,
            )
            if not batch:
                logger.warning(
                    "Generation failed: domain=%s turns=%d attempt=%d/%d",
                    domain,
                    target_turns,
                    attempt,
                    MAX_RETRIES,
                )
                continue

            candidate = batch[0]
            if candidate.id in occupied_ids:
                candidate = candidate.model_copy(update={"id": str(uuid.uuid4())})

            repaired_conv, reasons = validate_or_repair_conversation(candidate)
            if repaired_conv is None:
                logger.warning(
                    "Quality failure: domain=%s turns=%d reason=%s",
                    domain,
                    target_turns,
                    "; ".join(reasons),
                )
                continue

            conv = repaired_conv
            break

        if conv is None:
            logger.error(
                "Could not generate valid conversation: domain=%s turns=%d",
                domain,
                target_turns,
            )
            break

        occupied_ids.add(conv.id)
        all_conversations.append(conv)

        if out_path is not None and (
            len(all_conversations) % max(1, checkpoint_every) == 0
            or len(all_conversations) >= total
        ):
            save_dataset(all_conversations, out_path)

    # If custom ``total`` is requested, trim to that size while preserving ordering.
    trimmed = all_conversations[:total]
    logger.info("Total valid conversations generated: %d / %d", len(trimmed), total)
    return trimmed


def extract_conversations(
    source: Path,
    total: int = DATASET_SIZE,
) -> list[Conversation]:
    """
    Deterministically extract validated conversations from an existing JSONL file.

    Conversations are selected in deterministic order after validation.
    """
    if not source.exists():
        logger.error("Source dataset file not found: %s", source)
        return []

    candidates = load_dataset(source)
    if not candidates:
        logger.error("No valid conversations found in source: %s", source)
        return []

    # Stable ordering keeps extraction deterministic across runs.
    candidates = sorted(candidates, key=lambda c: c.id)

    def _normalize_duplicate_ids(conversations: list[Conversation]) -> list[Conversation]:
        """Keep IDs unique so downstream eval keys remain collision-free."""
        seen: dict[str, int] = {}
        normalized: list[Conversation] = []

        for conv in conversations:
            count = seen.get(conv.id, 0)
            seen[conv.id] = count + 1
            if count == 0:
                normalized.append(conv)
                continue

            # Deterministic suffix ensures stable IDs across repeated runs.
            new_id = f"{conv.id}__dup{count + 1}"
            normalized.append(conv.model_copy(update={"id": new_id}))

        duplicate_count = sum(v - 1 for v in seen.values() if v > 1)
        if duplicate_count:
            logger.warning(
                "Normalized %d duplicate conversation IDs during extraction.",
                duplicate_count,
            )
        return normalized

    if total != DATASET_SIZE:
        extracted = candidates[:total]
        extracted = _normalize_duplicate_ids(extracted)
        logger.info("Extracted %d/%d conversations from %s", len(extracted), total, source)
        return extracted

    selected = _normalize_duplicate_ids(candidates[:DATASET_SIZE])
    logger.info("Extracted %d/%d conversations from %s", len(selected), total, source)
    return selected[:total]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_dataset(conversations: list[Conversation], path: Path) -> None:
    """Write conversations to a JSONL file (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for conv in conversations:
            fh.write(conv.model_dump_json() + "\n")
    logger.info("Saved %d conversations → %s", len(conversations), path)


def load_dataset(path: Path) -> list[Conversation]:
    """Read and validate conversations from a JSONL file."""
    conversations: list[Conversation] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                conv = Conversation.model_validate_json(line)
                is_valid, reasons = evaluate_conversation_quality(conv)
                if not is_valid:
                    logger.warning(
                        "Line %d: quality validation error — %s",
                        lineno,
                        "; ".join(reasons),
                    )
                    continue
                conversations.append(conv)
            except ValidationError as exc:
                logger.warning("Line %d: validation error — %s", lineno, exc.errors()[0]["msg"])
    logger.info("Loaded %d conversations from %s", len(conversations), path)
    return conversations


# ---------------------------------------------------------------------------
# Quick-inspect utility
# ---------------------------------------------------------------------------


def inspect_dataset(conversations: list[Conversation]) -> None:
    """Print a human-readable summary to stdout for quick manual review."""
    print(f"\n{'='*70}")
    print(f"Dataset summary — {len(conversations)} conversations")
    print(f"{'='*70}")

    domain_counts: dict[str, int] = {}
    for conv in conversations:
        domain_counts[conv.domain] = domain_counts.get(conv.domain, 0) + 1

    print("\nDomain distribution:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain:<45} {count:>3} conversations")

    turn_counts: dict[int, int] = {}
    for conv in conversations:
        n = len(conv.history)
        turn_counts[n] = turn_counts.get(n, 0) + 1
    print("\nTurn-count distribution:")
    for turns, count in sorted(turn_counts.items()):
        print(f"  {turns:>2} turns   {count:>3} conversations")

    print("\nSample conversations (first 3):")
    for conv in conversations[:3]:
        n = len(conv.history)
        early = [i for i in conv.critical_turn_indices if i <= 2]
        late = [i for i in conv.critical_turn_indices if i >= n - 3]
        print(f"\n  ID       : {conv.id}")
        print(f"  Domain   : {conv.domain}")
        print(f"  Turns    : {n}  (critical: {conv.critical_turn_indices})")
        print(f"  Spread   : early={early}  late={late}  ✓")
        print(f"  Question : {conv.final_question[:90]}")
        print(f"  Answer   : {conv.ground_truth_answer}")
        print(f"  Turn 0   : {conv.history[0].content[:80]}")
        print(f"  Turn {n-1:<2}  : {conv.history[-1].content[:80]}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the evaluation dataset.")
    parser.add_argument(
        "--n",
        type=int,
        default=DATASET_SIZE,
        help=f"Number of conversations to generate (default: {DATASET_SIZE}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATASET_PATH,
        help=f"Output JSONL path (default: {DATASET_PATH}).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DATASET_PATH,
        help=(
            "Input JSONL source used for optional deterministic extraction "
            f"(default: {DATASET_PATH})."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=GENERATION_MODEL,
        help=f"litellm model string (default: {GENERATION_MODEL}).",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print a summary of the generated dataset to stdout.",
    )
    parser.add_argument(
        "--extract-from-source",
        action="store_true",
        help="Use deterministic extraction from --source instead of LLM generation.",
    )
    args = parser.parse_args(argv)

    if args.extract_from_source:
        conversations = extract_conversations(source=args.source, total=args.n)
    else:
        conversations = generate_conversations(
            total=args.n,
            model=args.model,
            out_path=args.out,
            checkpoint_every=DATASET_CHECKPOINT_EVERY,
        )

    if not conversations:
        logger.error("No valid conversations generated. Exiting.")
        sys.exit(1)

    if args.n != DATASET_SIZE and len(conversations) < args.n:
        logger.warning(
            "Only %d / %d conversations passed validation.",
            len(conversations),
            args.n,
        )

    save_dataset(conversations, args.out)

    if args.inspect:
        inspect_dataset(conversations)


if __name__ == "__main__":
    main()
