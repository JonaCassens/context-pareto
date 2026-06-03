"""
Context compression strategies.
Implementations: SlidingWindowCompressor, SummarizationCompressor, HybridCompressor.
"""

from __future__ import annotations

import logging
import time
import re
from abc import ABC, abstractmethod
from threading import Lock

import litellm
import tiktoken

from src.config import (
    CONTEXT_WINDOW_MARGIN,
    HYBRID_K_VALUES,
    HYBRID_RECENT_WINDOW,
    HYBRID_SUMMARY_TOKEN_LIMIT,
    MAX_RETRIES,
    SUMMARIZATION_CONTEXT_WINDOW,
    SUMMARIZATION_MAX_TOKENS,
    SUMMARIZATION_MODEL,
    SUMMARY_TOKEN_LIMITS,
    TIKTOKEN_ENCODING,
    WINDOW_SIZES,
)

from src.models import Conversation, Turn


_SUMMARY_SYSTEM_PROMPT = (
    "You compress conversation context for downstream QA. "
    "Return a compact summary that preserves entities, quantities, constraints, and facts needed "
    "to answer a future user question."
)

logger = logging.getLogger(__name__)

_SUMMARY_CACHE: dict[tuple[str, int, str], str] = {}
_SUMMARY_CACHE_LOCK = Lock()


def _turns_to_text(turns: list[Turn]) -> str:
    return "\n".join(f"{turn.role}: {turn.content}" for turn in turns)


class Compressor(ABC):
    """Base class for all context compression strategies."""

    @abstractmethod
    def compress(self, conversation: Conversation) -> list[Turn]:
        """
        Return a compressed version of ``conversation.history``.
        The final_question is NOT included — callers append it.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier used in eval records."""
        ...

    @property
    @abstractmethod
    def hyperparams(self) -> dict:
        """Hyperparameter dict logged in the eval record."""
        ...


class SlidingWindowCompressor(Compressor):
    def __init__(self, window: int):
        if window <= 0:
            raise ValueError("window must be > 0")
        self.window = window

    def compress(self, conversation: Conversation) -> list[Turn]:
        return conversation.history[-self.window :]

    @property
    def name(self) -> str:
        return "sliding_window"

    @property
    def hyperparams(self) -> dict:
        return {"window": self.window}


class SummarizationCompressor(Compressor):
    def __init__(self, token_limit: int):
        if token_limit <= 0:
            raise ValueError("token_limit must be > 0")
        self.token_limit = token_limit
        self._encoder = tiktoken.get_encoding(TIKTOKEN_ENCODING)

    def _count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def _build_summary_prompt(self, text: str) -> str:
        return (
            "Summarize the following conversation turns into a concise memory block. "
            "Preserve critical quantities, entities, timelines, constraints, and decisions. "
            "Do not include analysis or meta commentary.\n\n"
            f"Conversation turns:\n{text}"
        )

    def _summary_input_budget(self) -> int:
        return SUMMARIZATION_CONTEXT_WINDOW - SUMMARIZATION_MAX_TOKENS - CONTEXT_WINDOW_MARGIN

    def _summarize_text_once(self, text: str) -> str:
        prompt = self._build_summary_prompt(text)
        input_budget = self._summary_input_budget()
        if input_budget <= 0:
            raise RuntimeError(
                "Invalid summarization token budget. Increase SUMMARIZATION_CONTEXT_WINDOW "
                "or reduce SUMMARIZATION_MAX_TOKENS/CONTEXT_WINDOW_MARGIN."
            )

        if self._count_tokens(prompt) > input_budget:
            raise RuntimeError("Summarization prompt exceeds context window budget.")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = litellm.completion(
                    model=SUMMARIZATION_MODEL,
                    messages=[
                        {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=SUMMARIZATION_MAX_TOKENS,
                    temperature=0.2,
                )
                content = response.choices[0].message.content
                if not content or not content.strip():
                    raise RuntimeError("Summarization response was empty.")
                return content.strip()
            except (litellm.RateLimitError, litellm.ServiceUnavailableError) as exc:
                if attempt == MAX_RETRIES:
                    raise RuntimeError("Summarization failed after retries.") from exc
                wait_seconds = 2 ** attempt
                time.sleep(wait_seconds)
            except litellm.APIError as exc:
                raise RuntimeError(f"Summarization API error: {exc}") from exc

        raise RuntimeError("Unexpected summarization retry state.")

    def _summarize_prefix(self, prefix_turns: list[Turn]) -> str:
        prefix_text = _turns_to_text(prefix_turns)
        cache_key = (prefix_text, self.token_limit, SUMMARIZATION_MODEL)
        with _SUMMARY_CACHE_LOCK:
            cached = _SUMMARY_CACHE.get(cache_key)
        if cached is not None:
            return cached

        if self._count_tokens(self._build_summary_prompt(prefix_text)) > self._summary_input_budget():
            raise RuntimeError(
                "Single-pass summarization exceeds context budget. Increase summarization context window."
            )

        normalized = self._summarize_text_once(prefix_text)

        with _SUMMARY_CACHE_LOCK:
            _SUMMARY_CACHE[cache_key] = normalized
        return normalized

    def compress(self, conversation: Conversation) -> list[Turn]:
        if len(conversation.history) <= 1:
            return conversation.history

        prefix = conversation.history[:-1]
        last_turn = conversation.history[-1]

        prefix_text = _turns_to_text(prefix)
        prefix_tokens = self._count_tokens(prefix_text)
        if prefix_tokens <= self.token_limit:
            return conversation.history

        try:
            summary = self._summarize_prefix(prefix)
        except RuntimeError as exc:
            # Preserve full context if single-pass summarization cannot fit the model context window.
            logger.warning(
                "Summarization skipped; preserving full context for compatibility. reason=%s",
                exc,
            )
            return conversation.history

        summary_turn = Turn(role="assistant", content=summary)
        return [summary_turn, last_turn]

    @property
    def name(self) -> str:
        return "summarization"

    @property
    def hyperparams(self) -> dict:
        return {"token_limit": self.token_limit}


class HybridCompressor(Compressor):
    def __init__(self, k: int):
        if k <= 0:
            raise ValueError("k must be > 0")
        self.k = k
        self.recent_window = HYBRID_RECENT_WINDOW
        self.summary_token_limit = HYBRID_SUMMARY_TOKEN_LIMIT
        self._encoder = tiktoken.get_encoding(TIKTOKEN_ENCODING)

    def _count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def _turn_priority(self, turn: Turn, idx: int, total: int) -> tuple[float, int]:
        text = turn.content
        has_number = bool(re.search(r"\\d", text))
        has_unit = bool(re.search(r"\\b(?:usd|dollars?|%|kg|km|hours?|days?|weeks?|months?)\\b", text.lower()))
        has_id = bool(re.search(r"\\b[A-Z]{2,}-?\\d{1,}\\b", text))
        length_bonus = min(len(text) / 300.0, 1.0)

        # Encourage preserving both early and late anchors without using unseen future question.
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

    def _summarize_older(self, older_turns: list[Turn]) -> Turn | None:
        if not older_turns:
            return None
        older_text = _turns_to_text(older_turns)
        if self._count_tokens(older_text) <= self.summary_token_limit:
            return None
        summary = SummarizationCompressor(self.summary_token_limit)._summarize_prefix(older_turns)
        return Turn(role="assistant", content=summary)

    def compress(self, conversation: Conversation) -> list[Turn]:
        history = conversation.history
        if not history:
            return []

        recent_n = min(self.recent_window, len(history))
        recent_start = len(history) - recent_n
        recent_turns = history[recent_start:]
        older_turns = history[:recent_start]

        anchor_count = min(self.k, len(older_turns))
        selected_older: list[Turn] = []
        if anchor_count > 0:
            priorities = [
                (self._turn_priority(turn, idx, len(history)), idx)
                for idx, turn in enumerate(older_turns)
            ]
            priorities.sort(reverse=True)
            chosen = sorted(idx for _, idx in priorities[:anchor_count])
            selected_older = [older_turns[i] for i in chosen]

        summary_turn = self._summarize_older(older_turns)

        out: list[Turn] = []
        if summary_turn is not None:
            out.append(summary_turn)
        out.extend(selected_older)
        out.extend(recent_turns)
        return out

    @property
    def name(self) -> str:
        return "hybrid"

    @property
    def hyperparams(self) -> dict:
        return {
            "k": self.k,
            "recent_window": self.recent_window,
            "summary_token_limit": self.summary_token_limit,
        }


def all_compressors() -> list[Compressor]:
    compressors: list[Compressor] = []
    compressors.extend(SlidingWindowCompressor(w) for w in WINDOW_SIZES)
    compressors.extend(SummarizationCompressor(tl) for tl in SUMMARY_TOKEN_LIMITS)
    compressors.extend(HybridCompressor(k) for k in HYBRID_K_VALUES)
    return compressors
