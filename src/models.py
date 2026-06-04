"""
Shared Pydantic models used across the entire pipeline.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Conversation primitives
# ---------------------------------------------------------------------------


class Turn(BaseModel):
    """A single turn in a multi-turn conversation."""

    role: Literal["user", "assistant"]
    content: str


class Conversation(BaseModel):
    """
    One fully-annotated multi-turn conversation.

    ``critical_turn_indices`` lists the 0-based indices into ``history`` that
    contain information *required* to correctly answer ``final_question``.
    Exactly two anchors are required: one in turns 0-2 and one in the final
    three turns of ``history``.
    """

    id: str
    domain: str = Field(description="High-level domain label (e.g. 'inventory', 'finance').")
    history: list[Turn] = Field(min_length=6, max_length=10)
    final_question: str
    ground_truth_answer: str = Field(
        description=(
            "The exact, unambiguous correct answer. "
            "For numeric questions this must be the numeric result with units."
        )
    )
    critical_turn_indices: list[int] = Field(
        description=(
            "0-based indices into 'history' whose content is necessary to "
            "answer final_question. Must include one early and one late anchor."
        ),
        min_length=2,
        max_length=2,
    )

    @model_validator(mode="after")
    def validate_dependency_spread(self) -> "Conversation":
        n = len(self.history)
        if len(self.critical_turn_indices) != 2:
            raise ValueError("critical_turn_indices must contain exactly two indices.")

        early_window = set(range(0, min(3, n)))
        late_window_start = max(0, n - 3)
        late_window = set(range(late_window_start, n))

        first, second = self.critical_turn_indices
        if first == second:
            raise ValueError("critical_turn_indices must reference two distinct turns.")

        early_count = int(first in early_window) + int(second in early_window)
        late_count = int(first in late_window) + int(second in late_window)

        if early_count != 1:
            raise ValueError(
                "critical_turn_indices must contain exactly one index in turns 0-2."
            )
        if late_count != 1:
            raise ValueError(
                "critical_turn_indices must contain exactly one index in the final three turns."
            )

        for idx in self.critical_turn_indices:
            if not (0 <= idx < n):
                raise ValueError(
                    f"critical_turn_index {idx} is out of range for history of length {n}."
                )
        return self


# ---------------------------------------------------------------------------
# LLM structured-output envelopes
# ---------------------------------------------------------------------------


class GeneratedDataset(BaseModel):
    """
    Envelope returned by the LLM during dataset generation.
    Each batch produces 10 conversations.
    """

    conversations: list[Conversation] = Field(min_length=1, max_length=10)


class JudgeVerdict(BaseModel):
    """
    Structured response from the LLM-as-a-judge in eval.py.
    """

    is_correct: bool = Field(
        description="True if the generated answer is semantically equivalent to the ground truth."
    )
    reasoning: str = Field(
        description="One or two sentence explanation of the judgement."
    )


# ---------------------------------------------------------------------------
# Evaluation result record (written to JSONL by eval.py)
# ---------------------------------------------------------------------------


class EvalRecord(BaseModel):
    """One row in the evaluation results table."""

    conversation_id: str
    strategy: Literal["sliding_window", "summarization", "hybrid"]
    hyperparams: dict  # e.g. {"window": 4} or {"k": 3}
    original_token_count: int
    compressed_token_count: int
    token_reduction_ratio: float  # (original - compressed) / original
    generated_answer: str
    is_correct: bool
    judge_reasoning: str
