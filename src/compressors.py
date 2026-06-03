"""
Context compression strategies.
Implementations: SlidingWindowCompressor, SummarizationCompressor, HybridCompressor.
(Milestone 2 — not yet implemented)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import Conversation, Turn


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
