"""
Context compression strategies.
Implementations: SlidingWindowCompressor, SummarizationCompressor, HybridCompressor.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import litellm
import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

from src.config import (
    HYBRID_K_VALUES,
    MAX_RETRIES,
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


def _turns_to_text(turns: list[Turn]) -> str:
    return "\n".join(f"{turn.role}: {turn.content}" for turn in turns)


def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    matrix_norm = np.linalg.norm(matrix, axis=1)
    denom = np.maximum(query_norm * matrix_norm, 1e-12)
    return np.dot(matrix, query) / denom


_EMBEDDER: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


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

    def _summarize_prefix(self, prefix_turns: list[Turn]) -> str:
        prompt = (
            "Summarize the following conversation turns into a concise memory block. "
            "Preserve critical quantities, entities, timelines, constraints, and decisions. "
            "Do not include analysis or meta commentary.\n\n"
            f"Conversation turns:\n{_turns_to_text(prefix_turns)}"
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = litellm.completion(
                    model=SUMMARIZATION_MODEL,
                    messages=[
                        {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=32768,
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

    def compress(self, conversation: Conversation) -> list[Turn]:
        if len(conversation.history) <= 1:
            return conversation.history

        prefix = conversation.history[:-1]
        last_turn = conversation.history[-1]

        prefix_text = _turns_to_text(prefix)
        prefix_tokens = self._count_tokens(prefix_text)
        if prefix_tokens <= self.token_limit:
            return conversation.history

        summary = self._summarize_prefix(prefix)
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

    def compress(self, conversation: Conversation) -> list[Turn]:
        history = conversation.history
        if not history:
            return []

        k = min(self.k, len(history))
        embedder = _get_embedder()

        query_embedding = np.asarray(
            embedder.encode([conversation.final_question], convert_to_numpy=True)[0],
            dtype=np.float32,
        )
        turn_texts = [f"{t.role}: {t.content}" for t in history]
        turn_embeddings = np.asarray(
            embedder.encode(turn_texts, convert_to_numpy=True),
            dtype=np.float32,
        )
        scores = _cosine_similarity(query_embedding, turn_embeddings)

        top_indices = np.argpartition(scores, -k)[-k:]
        ordered_indices = sorted(int(i) for i in top_indices)
        return [history[i] for i in ordered_indices]

    @property
    def name(self) -> str:
        return "hybrid"

    @property
    def hyperparams(self) -> dict:
        return {"k": self.k}


def all_compressors() -> list[Compressor]:
    compressors: list[Compressor] = []
    compressors.extend(SlidingWindowCompressor(w) for w in WINDOW_SIZES)
    compressors.extend(SummarizationCompressor(tl) for tl in SUMMARY_TOKEN_LIMITS)
    compressors.extend(HybridCompressor(k) for k in HYBRID_K_VALUES)
    return compressors
