"""
Central configuration for all pipeline hyperparameters and paths.
All values can be overridden via environment variables (loaded from .env).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = ROOT_DIR / Path(os.getenv("PLOTS_DIR", "plots"))

DATASET_PATH = ROOT_DIR / os.getenv("DATASET_PATH", "data/conversations.jsonl")
RESULTS_PATH = ROOT_DIR / os.getenv("RESULTS_PATH", "results/eval_results.jsonl")

# Ensure output directories exist at import time.
for _dir in (DATA_DIR, RESULTS_DIR, PLOTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# LLM model identifiers (passed directly to litellm.completion)
# ---------------------------------------------------------------------------

GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "gpt-4o")
JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "gpt-4o")
SUMMARIZATION_MODEL: str = os.getenv("SUMMARIZATION_MODEL", "gpt-4o-mini")

# Tiktoken encoding used everywhere in the pipeline.
TIKTOKEN_ENCODING: str = "cl100k_base"


# ---------------------------------------------------------------------------
# Evaluation hyperparameter grid
# ---------------------------------------------------------------------------

# SlidingWindowCompressor: number of turns to retain
WINDOW_SIZES: list[int] = [2, 4, 6]

# SummarizationCompressor: target token budget for the summary block
SUMMARY_TOKEN_LIMITS: list[int] = [100, 200, 400]

# HybridCompressor: top-K turns to retrieve via semantic similarity
HYBRID_K_VALUES: list[int] = [2, 4, 6]


# ---------------------------------------------------------------------------
# Dataset generation settings
# ---------------------------------------------------------------------------

DATASET_SIZE: int = 30            # Total conversations to generate
BATCH_SIZE: int = 10              # Conversations per LLM call
MAX_RETRIES: int = 3              # Max re-generation attempts per failed conversation

GENERATION_DOMAINS: list[str] = [
    "inventory and logistics",
    "personal finance and scheduling",
    "project management and resource allocation",
]
