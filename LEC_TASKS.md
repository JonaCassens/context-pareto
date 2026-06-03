# LEC Task and Context Tracker

Last updated: 2026-06-04

## Project Snapshot
- Project: LEC context compression evaluation pipeline
- Goal: compare token savings vs answer quality for sliding window, summarization, and hybrid strategies
- Root: c:/Users/jonat/OneDrive/Documents/Work/LEC

## Current Status
- Milestone 1 dataset generation: complete
- Milestone 2 compressors: complete
- Milestone 3 evaluation engine: implemented and running with resume support
- Milestone 4 reporting and plots: implemented and generating artifacts

## Runtime Progress (Current)
- Dataset size target: 36 conversations
- Compressor grid target: 9 configs
- Full combination target: 324 rows
- Rows currently present in results/eval_results.jsonl: 315
- Remaining combinations to evaluate: 9

## Implemented Components

### src/compressors.py
- SlidingWindowCompressor
- SummarizationCompressor (LLM summary with retry/backoff)
- HybridCompressor
- all_compressors() factory from config grids

### src/eval.py
- Loads validated dataset from data/conversations.jsonl
- Iterates conversation x compressor combinations
- Computes token counts and reduction ratio per run
- Generates compressed-context answer with generation model
- Judges candidate answer against full-context reference + ground truth
- Appends EvalRecord rows to results/eval_results.jsonl
- Resume-safe skipping of already-completed keys
- CLI filters:
  - --strategy
  - --max-conversations
  - --max-combinations

### src/report.py
- Loads EvalRecord rows from eval results JSONL
- Builds DataFrame and joins dataset metadata (domain, turn_count)
- Saves summary CSVs:
  - results/summary_by_strategy.csv
  - results/summary_by_strategy_hyperparams.csv
  - results/summary_by_domain.csv
  - results/summary_by_turn_count.csv
- Generates plots:
  - plots/accuracy_by_strategy.png
  - plots/compression_ratio_distribution.png
  - plots/tradeoff_accuracy_vs_reduction.png
- Detects cliff points from adjacent compression levels
- Writes assignment narrative with recommendation + failure case:
  - results/assignment3_findings.md

## Remaining Checklist

### Evaluation completion
- [ ] Run remaining eval combinations to reach 324/324 rows
- [ ] Confirm no failed combinations after final resume run

### Final reporting pass
- [ ] Re-run src.report after eval reaches full coverage
- [ ] Validate assignment recommendation on full grid output

## Known Constraints / Gotchas
- Use absolute interpreter path in PowerShell:
  - & "c:\Users\jonat\OneDrive\Documents\Work\LEC\.venv\Scripts\python.exe"
- Structured judge output depends on response_format json_object
- Gemini transient API errors are handled via exponential backoff
- First hybrid usage may trigger sentence-transformers model load/download

## Run Commands
- Compressor smoke test:
  - & "c:\Users\jonat\OneDrive\Documents\Work\LEC\.venv\Scripts\python.exe" "c:\Users\jonat\OneDrive\Documents\Work\LEC\test_compressors.py"
- Resume eval until completion:
  - & "c:\Users\jonat\OneDrive\Documents\Work\LEC\.venv\Scripts\python.exe" -m src.eval
- Optional bounded eval:
  - & "c:\Users\jonat\OneDrive\Documents\Work\LEC\.venv\Scripts\python.exe" -m src.eval --max-conversations 1 --max-combinations 3
- Regenerate report artifacts:
  - & "c:\Users\jonat\OneDrive\Documents\Work\LEC\.venv\Scripts\python.exe" -m src.report
