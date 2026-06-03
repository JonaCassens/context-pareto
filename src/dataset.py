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
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

import litellm
from pydantic import ValidationError

from src.config import (
    BATCH_SIZE,
    DATASET_PATH,
    DATASET_SIZE,
    GENERATION_DOMAINS,
    GENERATION_MODEL,
    MAX_RETRIES,
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
1. Each conversation has between 5 and 9 history turns (role alternates
   user / assistant, starting with user).
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


def build_batch_prompt(n: int, domain: str) -> str:
    """Return the user-turn prompt for generating ``n`` conversations in ``domain``."""
    return _BATCH_PROMPT_TEMPLATE.format(n=n, domain=domain)


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


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def _call_litellm(prompt: str, model: str) -> dict[str, Any]:
    """
    Call litellm with JSON response format and return the parsed dict.
    Raises RuntimeError on API or JSON-parse failures.
    """
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.9,   # High enough for variety across 10 conversations
        max_tokens=8192,
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM returned non-JSON content: {raw[:200]}") from exc


# ---------------------------------------------------------------------------
# Batch generation with retry logic
# ---------------------------------------------------------------------------


def generate_batch(
    domain: str,
    n: int = BATCH_SIZE,
    model: str = GENERATION_MODEL,
    max_retries: int = MAX_RETRIES,
) -> list[Conversation]:
    """
    Generate ``n`` conversations for the given ``domain``.

    Retries individual conversations that fail Pydantic validation (e.g.,
    dependency-spread rule) up to ``max_retries`` times per conversation.
    Returns however many valid conversations were produced.
    """
    logger.info("Generating batch: domain=%r  n=%d  model=%s", domain, n, model)
    prompt = build_batch_prompt(n=n, domain=domain)

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
            valid.append(conv)
            logger.debug("  ✓ %s  (critical=%s)", conv.id, conv.critical_turn_indices)
        except ValidationError as exc:
            logger.warning("  ✗ Validation failed for a conversation: %s", exc.errors()[0]["msg"])
            invalid_raws.append(raw_conv)

    # Retry each invalid conversation individually.
    if invalid_raws:
        logger.info("  Re-generating %d failed conversations …", len(invalid_raws))
        retry_prompt = build_batch_prompt(n=len(invalid_raws), domain=domain)
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
    Generate ``total`` conversations spread evenly across all configured domains.

    Uses one LLM call per domain (3 calls × 10 conversations = 30 total).
    """
    all_conversations: list[Conversation] = []
    per_domain = total // len(GENERATION_DOMAINS)
    remainder = total % len(GENERATION_DOMAINS)

    for i, domain in enumerate(GENERATION_DOMAINS):
        n = per_domain + (1 if i < remainder else 0)
        batch = generate_batch(domain=domain, n=n, model=model)
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
