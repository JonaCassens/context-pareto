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

# Context windows used for input-budgeting safeguards.
# Defaults are intentionally >= 1000 tokens plus margin for observed workloads.
GENERATION_CONTEXT_WINDOW: int = int(os.getenv("GENERATION_CONTEXT_WINDOW", "4096"))
SUMMARIZATION_CONTEXT_WINDOW: int = int(os.getenv("SUMMARIZATION_CONTEXT_WINDOW", "4096"))
DATASET_GENERATION_CONTEXT_WINDOW: int = int(os.getenv("DATASET_GENERATION_CONTEXT_WINDOW", "8192"))
CONTEXT_WINDOW_MARGIN: int = int(os.getenv("CONTEXT_WINDOW_MARGIN", "256"))

# Response-token caps for faster inference and lower cost.
# These are OUTPUT token limits and do not limit input context length.
GENERATION_MAX_TOKENS: int = max(32, int(os.getenv("GENERATION_MAX_TOKENS", "256")))
SUMMARIZATION_MAX_TOKENS: int = max(64, int(os.getenv("SUMMARIZATION_MAX_TOKENS", "512")))
DATASET_GENERATION_MAX_TOKENS: int = max(512, int(os.getenv("DATASET_GENERATION_MAX_TOKENS", "4096")))

# Provider-call reliability tuning.
LLM_REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "120"))
RETRY_BASE_DELAY_SECONDS: float = float(os.getenv("RETRY_BASE_DELAY_SECONDS", "2"))
RETRY_MAX_DELAY_SECONDS: float = float(os.getenv("RETRY_MAX_DELAY_SECONDS", "30"))
RETRY_JITTER_SECONDS: float = float(os.getenv("RETRY_JITTER_SECONDS", "0.5"))

# Emit an info log when a conversation reaches this many input tokens.
EVAL_CONTEXT_WARNING_TOKENS: int = int(os.getenv("EVAL_CONTEXT_WARNING_TOKENS", "1000"))

# Parallel workers for eval combo execution.
EVAL_MAX_WORKERS: int = int(os.getenv("EVAL_MAX_WORKERS", "4"))

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

DATASET_SIZE: int = 36            # Default target sample size
BATCH_SIZE: int = 3               # Conversations per generation sub-batch
MAX_RETRIES: int = 6              # Max retries for transient provider errors
TURN_COUNTS: list[int] = [6, 7, 8, 9, 10]  # Allowed history turn counts
DATASET_CHECKPOINT_EVERY: int = max(1, int(os.getenv("DATASET_CHECKPOINT_EVERY", "1")))

GENERATION_DOMAINS: list[str] = [
    "inventory and logistics",
    "personal finance and scheduling",
    "project management and resource allocation",
]
