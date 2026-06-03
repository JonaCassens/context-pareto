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

GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "gemini/gemini-2.5-flash")
JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "gemini/gemini-2.5-flash")
SUMMARIZATION_MODEL: str = os.getenv("SUMMARIZATION_MODEL", "gemini/gemini-2.5-flash")

# Tiktoken encoding used everywhere in the pipeline.
TIKTOKEN_ENCODING: str = "cl100k_base"


# ---------------------------------------------------------------------------
# Evaluation hyperparameter grid
# ---------------------------------------------------------------------------

# SlidingWindowCompressor: number of turns to retain
WINDOW_SIZES: list[int] = [2, 4, 6]

# SummarizationCompressor: target token budget for the summary block
SUMMARY_TOKEN_LIMITS: list[int] = [100, 200, 400]

# Online HybridCompressor: anchors sampled from older context
HYBRID_K_VALUES: list[int] = [2, 4, 6]

# Online HybridCompressor: recent turns kept verbatim
HYBRID_RECENT_WINDOW: int = int(os.getenv("HYBRID_RECENT_WINDOW", "2"))

# Online HybridCompressor: summary budget for older context before anchors are selected
HYBRID_SUMMARY_TOKEN_LIMIT: int = int(os.getenv("HYBRID_SUMMARY_TOKEN_LIMIT", "160"))


# ---------------------------------------------------------------------------
# Dataset generation settings
# ---------------------------------------------------------------------------

DATASET_SIZE: int = 36            # 3 domains × 4 turn values × 3 conversations
BATCH_SIZE: int = 3               # Conversations per turn-value sub-batch
MAX_RETRIES: int = 3              # Max re-generation attempts per failed conversation
TURN_COUNTS: list[int] = [6, 8, 10, 12]  # Exact history turn counts; 3 conversations each

GENERATION_DOMAINS: list[str] = [
    "inventory and logistics",
    "personal finance and scheduling",
    "project management and resource allocation",
]
