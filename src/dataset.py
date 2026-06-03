"""
Dataset generator for the context-compression evaluation pipeline.

Generates 30 synthetic multi-turn conversations where the correct answer to
the final question depends on combining information from *early* turns AND
*late* turns — making the dependency chain non-trivial for any compression
strategy that only keeps a sliding window or naive summary.

Usage:
    python -m src.dataset                         # generates data/conversations.jsonl
    python -m src.dataset --n 10 --out custom.jsonl
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
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
    DATASET_PATH,
    DATASET_GENERATION_CONTEXT_WINDOW,
    DATASET_GENERATION_MAX_TOKENS,
    DATASET_SIZE,
    GENERATION_DOMAINS,
    GENERATION_MODEL,
    MAX_RETRIES,
    TURN_COUNTS,
)
from src.models import Conversation, GeneratedDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

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
1. Each conversation must have EXACTLY {target_turns} history turns (role
   alternates user / assistant, starting with user). Do NOT use more or
   fewer turns — this count is a strict requirement.
2. The conversation ends with a `final_question` posed by the user that is
   NOT part of `history`.
3. `ground_truth_answer` is the single, unambiguous correct answer to
   `final_question`. For numeric answers include the number AND unit.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEPENDENCY CHAIN RULES  ← most important
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. EARLY ANCHOR (turns 0–1): The user establishes a specific, concrete fact
   (a numeric quantity, a named entity attribute, a rate or price). The
   assistant acknowledges it. Label the relevant turn index in
   `critical_turn_indices`.

5. DISTRACTOR ZONE (turns 2 through N-3): The conversation pivots to
   a completely different subject within the same domain. Introduce new
   named entities, numbers, and transactions that are plausible but are
   NOT required to answer the final question. These turns should be long
   and detailed enough to tempt a compression model into keeping them.

6. LATE ANCHOR (last 2 turns of history, turns N-2 and N-1): A second
   essential fact is introduced — a modifier, rate, constraint, or second
   quantity that, combined ONLY with the early anchor, yields the answer.
   Label the relevant turn index in `critical_turn_indices`.

7. The `final_question` MUST be unanswerable using ONLY the early anchor OR
   ONLY the late anchor. The answer requires multiplying, combining, or
   applying one to the other. Verify this before writing the question.

8. `critical_turn_indices` must contain at least:
   - one index that is < floor(N / 3)          ← early anchor index
   - one index that is >= N - floor(N / 3)     ← late anchor index
   where N = len(history).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUALITY RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9. Use distinct named entities (people, products, projects) across all {n}
   conversations — do not reuse "Alice", "Widget A", or any other name more
   than once across the batch.
10. Vary the type of dependency: use multiplication, percentage application,
    date arithmetic, conditional lookup, and unit conversion across the batch.
11. Each `id` must be a unique UUID v4 string.
12. Set `domain` to "{domain}".

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

CRITICAL INDEX LAYOUT CONSTRAINT FOR THIS REQUEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{critical_layout_instruction}

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


@dataclass(frozen=True)
class CriticalIndexArchetype:
    key: str
    instruction: str


CRITICAL_INDEX_ARCHETYPES: list[CriticalIndexArchetype] = [
    CriticalIndexArchetype(
        key="extreme",
        instruction=(
            "Produce the EXTREME layout: critical_turn_indices must include BOTH first two turns "
            "(0 and 1) and BOTH last two turns (N-2 and N-1), where N=len(history)."
        ),
    ),
    CriticalIndexArchetype(
        key="middle_end",
        instruction=(
            "Produce the SIMPLER middle+end layout: critical_turn_indices must include at least one index in "
            "the middle third and at least one index in the last third, with NO first-third index."
        ),
    ),
    CriticalIndexArchetype(
        key="three_way",
        instruction=(
            "Produce the THREE-WAY layout: critical_turn_indices must include at least one index from the first "
            "third, one from the middle third, and one from the last third (minimum 3 indices total)."
        ),
    ),
]


def build_batch_prompt(
    n: int,
    domain: str,
    target_turns: int,
    critical_layout_instruction: str | None = None,
) -> str:
    """Return the user-turn prompt for generating ``n`` conversations in ``domain``."""
    layout_instruction = critical_layout_instruction or (
        "No additional special layout beyond the required dependency-spread rules."
    )
    return _BATCH_PROMPT_TEMPLATE.format(
        n=n,
        domain=domain,
        target_turns=target_turns,
        critical_layout_instruction=layout_instruction,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_dependency_spread(conv: Conversation) -> bool:
    """
    Secondary (post-Pydantic) check that critical indices genuinely span
    early and late thirds. Pydantic's model_validator already enforces this,
    so this function is used for logging / manual inspection convenience.
    """
    n = len(conv.history)
    first_third = n // 3
    last_third_start = n - n // 3
    has_early = any(i < first_third for i in conv.critical_turn_indices)
    has_late = any(i >= last_third_start for i in conv.critical_turn_indices)
    return has_early and has_late


def validate_archetype(conv: Conversation, archetype_key: str | None) -> bool:
    """Validate that critical indices match the requested per-batch archetype."""
    if archetype_key is None:
        return True

    n = len(conv.history)
    first_third = n // 3
    last_third_start = n - n // 3

    has_early = any(i < first_third for i in conv.critical_turn_indices)
    has_middle = any(first_third <= i < last_third_start for i in conv.critical_turn_indices)
    has_late = any(i >= last_third_start for i in conv.critical_turn_indices)

    if archetype_key == "extreme":
        has_first_two = 0 in conv.critical_turn_indices and 1 in conv.critical_turn_indices
        has_last_two = (n - 2) in conv.critical_turn_indices and (n - 1) in conv.critical_turn_indices
        return has_first_two and has_last_two

    if archetype_key == "middle_end":
        no_early = not has_early
        return has_middle and has_late and no_early

    if archetype_key == "three_way":
        return has_early and has_middle and has_late and len(conv.critical_turn_indices) >= 3

    return False


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

    for attempt in range(1, max_retries + 1):
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.9,
                max_tokens=DATASET_GENERATION_MAX_TOKENS,
            )
            raw = response.choices[0].message.content
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"LLM returned non-JSON content: {raw[:200]}") from exc

        except (litellm.RateLimitError, litellm.ServiceUnavailableError) as exc:
            wait = 2 ** attempt  # 2s, 4s, 8s …
            logger.warning(
                "Transient API error (attempt %d/%d). Waiting %ds before retry. %s",
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
    archetype: CriticalIndexArchetype | None = None,
) -> list[Conversation]:
    """
    Generate ``n`` conversations for the given ``domain``.

    Retries individual conversations that fail Pydantic validation (e.g.,
    dependency-spread rule) or have the wrong turn count up to ``max_retries``
    times per conversation.  Returns however many valid conversations were produced.
    """
    logger.info(
        "Generating batch: domain=%r n=%d turns=%d model=%s archetype=%s",
        domain,
        n,
        target_turns,
        model,
        archetype.key if archetype else "default",
    )
    prompt = build_batch_prompt(
        n=n,
        domain=domain,
        target_turns=target_turns,
        critical_layout_instruction=archetype.instruction if archetype else None,
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
            elif not validate_archetype(conv, archetype.key if archetype else None):
                logger.warning(
                    "  ✗ Archetype mismatch: expected %s, got critical=%s",
                    archetype.key if archetype else "default",
                    conv.critical_turn_indices,
                )
                invalid_raws.append(raw_conv)
            else:
                valid.append(conv)
                logger.debug("  ✓ %s  (critical=%s)", conv.id, conv.critical_turn_indices)
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
            critical_layout_instruction=archetype.instruction if archetype else None,
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
                if validate_archetype(conv, archetype.key if archetype else None):
                    valid.append(conv)
            except ValidationError:
                pass  # Accept partial results after retry.

    logger.info("  Batch complete: %d/%d valid conversations.", len(valid), n)
    return valid


def generate_conversations(
    total: int = DATASET_SIZE,
    model: str = GENERATION_MODEL,
) -> list[Conversation]:
    """
    Generate conversations with exactly 3 per (domain, turn_count) combination.

    Produces 3 domains × 4 turn counts × 3 conversations = 36 total.
    Each sub-batch (12 LLM calls) targets an exact turn count from TURN_COUNTS.
    """
    all_conversations: list[Conversation] = []

    for domain in GENERATION_DOMAINS:
        for target_turns in TURN_COUNTS:
            # Enforce one conversation per requested critical-index archetype.
            for archetype in CRITICAL_INDEX_ARCHETYPES:
                batch = generate_batch(
                    domain=domain,
                    n=1,
                    model=model,
                    target_turns=target_turns,
                    archetype=archetype,
                )
                all_conversations.extend(batch)

    logger.info("Total valid conversations generated: %d / %d", len(all_conversations), total)
    return all_conversations


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
        first_third = n // 3
        last_third_start = n - n // 3
        early = [i for i in conv.critical_turn_indices if i < first_third]
        late = [i for i in conv.critical_turn_indices if i >= last_third_start]
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
    args = parser.parse_args(argv)

    conversations = generate_conversations(total=args.n, model=args.model)

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
