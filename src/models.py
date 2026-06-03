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
    At least one index must fall in the last third of ``history`` and at least
    one index must fall before the last third. This preserves long-range
    dependency pressure while allowing different index-layout archetypes.
    """

    id: str
    domain: str = Field(description="High-level domain label (e.g. 'inventory', 'finance').")
    history: list[Turn] = Field(min_length=6, max_length=12)
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
            "answer final_question. Must span both early and late turns."
        ),
        min_length=2,
    )

    @model_validator(mode="after")
    def validate_dependency_spread(self) -> "Conversation":
        n = len(self.history)
        last_third_boundary = n - n // 3

        has_pre_late = any(i < last_third_boundary for i in self.critical_turn_indices)
        has_late = any(i >= last_third_boundary for i in self.critical_turn_indices)

        if not has_pre_late:
            raise ValueError(
                f"critical_turn_indices {self.critical_turn_indices} must include "
                f"at least one index before the last third of history (< {last_third_boundary})."
            )
        if not has_late:
            raise ValueError(
                f"critical_turn_indices {self.critical_turn_indices} must include "
                f"at least one index in the last third of history (>= {last_third_boundary})."
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
