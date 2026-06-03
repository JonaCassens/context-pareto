# LEC Task and Context Tracker

Last updated: 2026-06-03

## Project Snapshot
- Project: LEC context compression evaluation pipeline
- Goal: compare compression quality tradeoff across sliding window, summarization, and hybrid retrieval for multi-turn conversations
- Root: c:/Users/jonat/OneDrive/Documents/Work/LEC

## Assignment 3 Goal Context
- Assignment: Context Compression, Pareto Frontier
- Problem framing: multi-turn prompt history becomes too long or expensive, so compression must reduce tokens without breaking answer quality
- Core objective: find the cliff point for each strategy where answer quality drops faster than token savings

### Required Deliverables
- Implement at least 3 strategies on the same benchmark input:
  - summarization baseline that rewrites older turns
  - sliding window baseline that keeps last N turns
  - hybrid strategy of choice
- Use a multi-turn benchmark with at least 30 conversation samples:
  - each sample has a final question
  - each final question has known ground truth
  - ground truth must depend on early and late history, not only recent turns
- Measure per strategy, with at least 30 evaluations each:
  - token reduction versus uncompressed baseline
  - answer correctness on final question using LLM judge against reference answer
- Produce a token-quality tradeoff plot or table:
  - identify fall off cliff ratio for each strategy
  - define cliff as quality dropping faster than token savings
- Make a final strategy recommendation:
  - justify choice with evidence
  - include one concrete failure case where compressed context loses information preserved by full history
  - explain failure mechanism

### Mapping to This Repo
- Three strategies map to src/compressors.py implementations
- Benchmark requirement maps to data/conversations.jsonl generation in src/dataset.py
- Token and correctness measurements map to src/eval.py output in results/eval_results.jsonl
- Tradeoff analysis and frontier map to src/report.py summaries and plots
- Recommendation plus failure analysis should be written after report outputs are generated

## Environment
- Python: 3.13
- Virtual environment: .venv
- Use this Python executable in terminal commands:
  - & "c:\Users\jonat\OneDrive\Documents\Work\LEC\.venv\Scripts\python.exe"
- API key: GOOGLE_API_KEY in .env

## Core Stack
- litellm
- pydantic v2
- tiktoken with cl100k_base
- sentence-transformers with all-MiniLM-L6-v2
- numpy

## Confirmed Config
- WINDOW_SIZES: [2, 4, 6]
- SUMMARY_TOKEN_LIMITS: [100, 200, 400]
- HYBRID_K_VALUES: [2, 4, 6]
- DATASET_SIZE: 36
- BATCH_SIZE: 3
- MAX_RETRIES: 3
- TURN_COUNTS: [6, 8, 10, 12]
- GENERATION_DOMAINS:
  - inventory and logistics
  - personal finance and scheduling
  - project management and resource allocation

## Milestone Status
- Milestone 1 dataset generation: complete
- Milestone 2 compressors: complete in code and smoke tested
- Milestone 3 evaluation engine: implemented and smoke-run validated
- Milestone 4 reporting and plots: implemented and artifact generation validated

## Progress Log
- 2026-06-03:
  - Implemented compressor concrete classes in src/compressors.py
  - Added SlidingWindowCompressor
  - Added SummarizationCompressor with token budget check, LLM summarization, and retry logic for rate limit and service unavailable
  - Added HybridCompressor with cosine similarity retrieval and chronological output order
  - Added all_compressors factory using config grids
  - Ran test_compressors.py and observed all smoke checks passing
  - Implemented src/eval.py end to end with:
    - full combinations loop over conversations x compressors
    - token counting with cl100k_base
    - token_reduction_ratio computation
    - generation call using compressed context plus final question
    - judge call with response_format json_object
    - retry and exponential backoff for transient API errors
    - append-only EvalRecord writes to results/eval_results.jsonl
    - resume skip support from existing results keys
  - Implemented src/report.py end to end with:
    - EvalRecord JSONL loading and validation
    - DataFrame construction and hyperparameter labeling
    - dataset metadata join for domain and turn count grouping
    - CSV summaries by strategy, hyperparameters, domain, and turn count
    - plot generation for strategy accuracy, compression distribution, and accuracy versus reduction tradeoff
  - Enhanced src/eval.py with:
    - full-context reference answer generation per conversation for judge comparison
    - CLI flags: --strategy, --max-conversations, --max-combinations
  - Enhanced src/report.py with:
    - cliff point detection rule (accuracy drop greater than token gain across adjacent compression levels)
    - strategy recommendation generation from aggregated metrics
    - concrete failure case extraction with mechanism analysis
    - assignment writeup output to results/assignment3_findings.md
  - Executed bounded eval run (1 conversation, 3 combinations) and produced initial results records
  - Generated reporting artifacts from current results snapshot

## Remaining Work Checklist

### Milestone 3 eval.py
- [x] Build dataset load and resume index from results/eval_results.jsonl
- [x] Iterate conversations x compressors
- [x] Count original and compressed tokens
- [x] Compute token_reduction_ratio
- [x] Build generation messages: compressed turns then final question
- [x] Call generation model with max_tokens 32768
- [x] Call judge model with response_format json_object and max_tokens 32768
- [x] Parse JudgeVerdict and write EvalRecord row per combo
- [x] Append write results incrementally for resume safety
- [x] Skip already completed keys: conversation_id + strategy + hyperparams
- [x] Execute bounded live eval smoke run and confirm writes
- [ ] Execute full eval grid and confirm resume behavior across reruns

### Milestone 4 report.py
- [x] Load eval results from JSONL
- [x] Aggregate accuracy by strategy and hyperparameters
- [x] Aggregate by domain and turn count
- [x] Compute compression statistics
- [x] Generate plots under plots directory
- [x] Save concise summary table output
- [x] Execute report against current eval output and verify generated files
- [ ] Re-run report after full eval grid to finalize recommendation confidence

## Known Gotchas
- Gemini 2.5 Flash can truncate structured output at low token ceilings, use max_tokens 32768
- Structured judge output requires response_format json_object
- Conversation validation is strict, especially critical turn spread constraints
- sentence-transformers model download happens on first hybrid usage
- Use absolute Python path in PowerShell contexts where python alias resolution can fail

## Run Commands
- Compressor smoke test:
  - & "c:\Users\jonat\OneDrive\Documents\Work\LEC\.venv\Scripts\python.exe" "c:\Users\jonat\OneDrive\Documents\Work\LEC\test_compressors.py"
- Dataset generation:
  - & "c:\Users\jonat\OneDrive\Documents\Work\LEC\.venv\Scripts\python.exe" -m src.dataset
- Eval run:
  - & "c:\Users\jonat\OneDrive\Documents\Work\LEC\.venv\Scripts\python.exe" -m src.eval
- Report run:
  - & "c:\Users\jonat\OneDrive\Documents\Work\LEC\.venv\Scripts\python.exe" -m src.report

## Next Immediate Step
- Run full src.eval grid, then regenerate report for final assignment recommendation.

## Generated Artifacts
- results/eval_results.jsonl
- results/summary_by_strategy.csv
- results/summary_by_strategy_hyperparams.csv
- results/summary_by_domain.csv
- results/summary_by_turn_count.csv
- results/assignment3_findings.md
- plots/accuracy_by_strategy.png
- plots/compression_ratio_distribution.png
- plots/tradeoff_accuracy_vs_reduction.png
