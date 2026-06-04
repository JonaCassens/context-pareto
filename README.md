# Assignment 3 - Context Compression Pareto Frontier

This repo tests long-context compression for multi-turn conversations and shows the accuracy vs token trade-off across three strategies:

1. Sliding window baseline
2. Summarization baseline
3. Hybrid strategy (summary memory + selected anchors + recent verbatim turns)

The full pipeline is reproducible with a standard Docker build and direct `docker run` commands.

## Why I Picked This Assignment

I picked this project because I enjoy optimisation problems where you push efficiency until performance breaks. I have done work in RAG already, so I wanted to explore a different angle: how much context can be compressed before answer quality drops off.

The assignment also suited the way I like to work: test multiple compressor designs on the same benchmark, measure token savings, and understand why failures happen rather than just chasing one headline number.

## Decisions and Alternatives

### Setup choices

- Used VS Code agent workflow over Cursor because I am more familiar with it and it keeps costs down.
- Used Gemini-backed calls across generation and eval for a simple, consistent setup.
- Used Docker so the project can be run on one machine without unusual setup.

### Benchmark design choices

- Built a synthetic benchmark of 36 conversations (>=30 requirement met).
- Domains: inventory/logistics, finance/scheduling, project/resource allocation.
- Turn counts: 6, 7, 8, 9, 10 to stress different memory lengths.
- Enforced two required critical anchors in each conversation:
  - one early anchor (turns 0-2)
  - one late anchor (turns N-3 to N-1)
- Prevented final-question numeric leakage and injected distractor values in non-critical turns.

These constraints are enforced in dataset validation so final questions depend on full-history information rather than short-context shortcuts.

### Strategy design choices

- **Sliding window**: keep last `N` turns only.
- **Summarization**: summarize older turns when prefix token budget is exceeded.
- **Hybrid**: keep recent turns verbatim, keep top-priority anchor turns from older context, and optionally add summary memory for older context.

Alternative ruled out: an earlier hybrid idea that selected turns using final-question similarity. I dropped that because it is not realistic for live compression; compression should happen as the conversation grows, not after seeing the final question.

## What Was Built for the Assignment Requirements

### 1) Three compression strategies on the same input

Implemented and evaluated across the same conversation set:

- Sliding window: `window=2,4,6`
- Summarization: `token_limit=100,200,400`
- Hybrid: `k=2,4,6` with fixed recent window and summary limit

### 2) Multi-turn benchmark with known final answers

- 36 total conversations
- Each has a known `ground_truth_answer`
- Each final question depends on required early + late anchors

### 3) Two metrics per strategy

- Token reduction vs uncompressed context
- Final-question correctness

Evaluation flow:

1. Generate reference answer from uncompressed context.
2. Generate candidate answer from compressed context.
3. Score correctness with deterministic numeric matching against ground truth/reference numeric value.

Note on judge framing: this is not a pure free-form LLM-as-judge pipeline. The current setup uses LLM generation for answers and deterministic numeric matching for final scoring.

Important interpretation note: the best near-full-context setting is summarization with `token_limit=400`, which reaches 61.1% accuracy and 0.0% average token reduction. That makes it effectively a full-context control rather than a compression win, so the accuracy ceiling here is mainly an end-to-end model/task limit, not a compression limit.

### 4) Token-quality trade-off and cliff analysis

Trade-off tables are produced under `results/*.csv`, with assignment findings in `results/assignment3_findings.md`.

Cliff rule result: **no cliff detected** by the strict adjacent-point rule used in reporting.

In practice, each strategy still has fall-off points (details below).

### 5) Recommendation plus concrete loss case

Recommended strategy is **hybrid** based on the best mean accuracy retention among tested settings, with a caveat that tokens can occasionally increase.

A concrete failure case is documented in `results/assignment3_findings.md`.

## Headline Numbers

From `results/summary_by_strategy.csv`:

| Strategy | Runs | Accuracy | Avg token reduction |
|---|---:|---:|---:|
| Hybrid | 108 | 58.3% | 13.8% |
| Summarization | 108 | 47.2% | 25.7% |
| Sliding window | 108 | 5.6% | 46.9% |

Key hyperparameter points from `results/summary_by_strategy_hyperparams.csv`:

- Hybrid `k=6`: 63.9% accuracy, **-3.8% token reduction** (token expansion)
- Hybrid `k=2`: 58.3% accuracy, 35.1% token reduction
- Summarization `token_limit=400`: 61.1% accuracy, 0.0% token reduction
- Summarization `token_limit=100`: 30.6% accuracy, 57.7% token reduction
- Sliding window `window=6`: 16.7% accuracy, 21.1% token reduction
- Sliding window `window=2/4`: 0.0% accuracy with high compression

## Why Accuracy Is Below 70%

The main issues were not only "missing tokens" from compression. The error profile points to a mix of generation and task factors:

1. **Incomplete/truncated numeric answers**
	- Some failures had non-parseable numeric outputs or incomplete final values.
	- These are scored as incorrect even when parts of the reasoning look sensible.

2. **Numeric hallucination / wrong operand selection**
	- Candidate answers sometimes picked distractor numbers or intermediate values.
	- This is especially harmful in synthetic tasks that include intentional distractor values.

3. **Structural lower bound of sliding windows**
	- With required early+late anchor dependency, hard cutoffs inevitably lose critical long-range context.
	- This explains the near-zero scores at tighter windows.

4. **Dataset/task composition limits**
	- Current synthetic set is heavily numeric and relatively compact.
	- It tests arithmetic retrieval well, but does not yet fully test text-heavy semantic recall.

## Why Accuracy Spread Persists Despite Token Changes

Accuracy variance remains high even when token differences look small, because strategy behaviour and conversation structure interact in uneven ways:

1. **Non-monotonic trade-off**
	- Keeping more context does not always lead to better final answers.
	- Example: hybrid settings show quality shifts that do not map linearly to token reduction.

2. **Turn-count sensitivity**
	- From `results/summary_by_turn_count.csv`, hybrid accuracy swings strongly by turn count:
	  - 7 turns: 38.1%
	  - 8 turns: 81.0%
	  - 9 turns: 33.3%
	  - 10 turns: 85.7%
	- This suggests the pattern is sensitive to structure rather than following one clean token frontier.

3. **Domain sensitivity**
	- From `results/summary_by_domain.csv`, hybrid ranges from 50.0% to 63.6% across domains.
	- Different domain phrasing and distractor patterns change retrieval and calculation reliability.

4. **Hybrid anchor+summary interactions**
	- Hybrid can preserve critical indices but still lose detail in compressed memory phrasing.
	- In some settings, retained anchors plus summary overhead can even increase tokens.

## One Concrete Failure Mechanism

From `results/assignment3_findings.md`:

- Conversation ID: `1e7c9f6d-4a1b-4e0f-9c2d-8b3a5f7e1d0c`
- Strategy: hybrid (`k=2`, `recent_window=2`, `summary_token_limit=160`)
- Ground truth: `$91,800.00`
- Model answer: non-numeric refusal-like output
- Critical indices were retained, but answer still failed.

Mechanism: compression can preserve anchor turns while still hurt exact answerability if summary memory or response generation drops key numeric detail during decoding.

## Recommendation

**Recommend hybrid** for this benchmark, with clear caveats:

- It delivers the best average accuracy among tested strategies.
- It avoids the severe long-range information loss of strict sliding windows.
- It still needs stricter budget controls because some settings expand tokens.

## One Thing I Would Do Differently With Another Week

If I had another week, I would prioritise evaluation robustness over adding more strategy variants:

1. Improve correctness evaluation to better distinguish true semantic correctness from formatting artefacts.
2. Add a richer text-heavy benchmark (not only numeric dependency chains).
3. Add stricter hybrid budget enforcement to prevent token expansion.
4. Expand per-failure diagnostics (which operands were missed, where they were dropped, and whether retrieval or arithmetic failed).

## Reproducibility

### Prerequisites

- Docker Desktop (or Docker Engine)
- API credentials via `.env` or host environment variables

### Repo setup

If you are starting from GitHub, clone the repository first and then run the Docker commands from the repo root:

```bash
git clone <repo-url>
cd <repo-folder>
```

### API key setup

Option 1 (recommended): create a `.env` file in the repo root with at least one key:

```bash
GOOGLE_API_KEY=...
# or
ANTHROPIC_API_KEY=...
```

Option 2: set host environment variables before running Docker:

```powershell
$env:GOOGLE_API_KEY="your_key_here"
```

### Quick run

```bash
docker build -t lec-eval:latest .
docker run --rm --env-file .env -e GOOGLE_API_KEY -e ANTHROPIC_API_KEY -v "${PWD}:/work" -w /work lec-eval:latest
```

By default, the image runs `python -m src.report`, so this command regenerates plots from the current results.

### Regeneration options

Regenerate summaries and findings:

```bash
docker run --rm --env-file .env -e GOOGLE_API_KEY -e ANTHROPIC_API_KEY -v "${PWD}:/work" -w /work lec-eval:latest python -m src.report --regen-summaries --regen-findings
```

Regenerate eval (with quick bounded run args example):

```bash
docker run --rm --env-file .env -e GOOGLE_API_KEY -e ANTHROPIC_API_KEY -v "${PWD}:/work" -w /work lec-eval:latest python -m src.eval --max-conversations 5 --max-combinations 30
```

Regenerate dataset and eval:

```bash
docker run --rm --env-file .env -e GOOGLE_API_KEY -e ANTHROPIC_API_KEY -v "${PWD}:/work" -w /work lec-eval:latest python -m src.dataset
docker run --rm --env-file .env -e GOOGLE_API_KEY -e ANTHROPIC_API_KEY -v "${PWD}:/work" -w /work lec-eval:latest python -m src.eval
```

Deterministic extraction mode from existing source:

```bash
docker run --rm --env-file .env -e GOOGLE_API_KEY -e ANTHROPIC_API_KEY -v "${PWD}:/work" -w /work lec-eval:latest python -m src.dataset --extract-from-source --source data/conversations.jsonl --n 36
```

Full rebuild sequence:

```bash
docker build -t lec-eval:latest .
docker run --rm --env-file .env -e GOOGLE_API_KEY -e ANTHROPIC_API_KEY -v "${PWD}:/work" -w /work lec-eval:latest python -m src.dataset --extract-from-source --source data/conversations.jsonl --n 36
docker run --rm --env-file .env -e GOOGLE_API_KEY -e ANTHROPIC_API_KEY -v "${PWD}:/work" -w /work lec-eval:latest python -m src.eval
docker run --rm --env-file .env -e GOOGLE_API_KEY -e ANTHROPIC_API_KEY -v "${PWD}:/work" -w /work lec-eval:latest python -m src.report --regen-summaries --regen-findings
```

### Discoverability

```bash
docker run --rm --env-file .env -e GOOGLE_API_KEY -e ANTHROPIC_API_KEY -v "${PWD}:/work" -w /work lec-eval:latest python -m src.report --help
```

### Generated outputs

- `results/eval_results.jsonl`
- `results/summary_by_strategy.csv`
- `results/summary_by_strategy_hyperparams.csv`
- `results/summary_by_turn_count.csv`
- `results/summary_by_domain.csv`
- `results/assignment3_findings.md`
- `plots/*`

### Notes for Windows

Use PowerShell or Git Bash for the `docker run` commands. If you prefer a shell with POSIX-style variable expansion, use Git Bash or WSL.
