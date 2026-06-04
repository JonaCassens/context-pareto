# LEC Evaluation Pipeline

This project now supports a Docker-backed `make run` command for assessor-friendly execution.

## Prerequisites

- Docker Desktop (or Docker Engine)
- GNU Make (`make` command)
- API credentials via `.env` or host environment variables

## API Key Setup (Required)

Option 1 (recommended): use `.env`

```bash
make init-env
```

Then edit `.env` and set at least one of:
- `GOOGLE_API_KEY=...`
- `ANTHROPIC_API_KEY=...`

Option 2: set host environment variables (forwarded into Docker)

```powershell
$env:GOOGLE_API_KEY="your_key_here"
make run
```

API key preflight runs automatically for eval paths (`make eval`, `make run REGEN_EVAL=1`, and `make run-regen`). Plot-only `make run` does not require an API key.

## Quick Start

```bash
make run
```

What `make run` does:
1. Uses existing `results/eval_results.jsonl`
2. Regenerates plots only (`plots/*`)
3. Does not regenerate dataset, eval results, summary CSVs, or findings unless requested

## Optional Regeneration

Regenerate summary CSVs and assignment findings:

```bash
make run REGEN_SUMMARIES=1 REGEN_FINDINGS=1
```

Regenerate eval results first, then regenerate plots:

```bash
make run REGEN_EVAL=1 EVAL_ARGS="--max-conversations 5 --max-combinations 30"
```

Regenerate dataset and eval first:

```bash
make run REGEN_DATA=1 REGEN_EVAL=1
```

You can pass dataset options:

```bash
make run REGEN_DATA=1 REGEN_EVAL=1 DATA_ARGS="--n 36 --inspect"
```

Full rebuild convenience target:

```bash
make run-regen
```

## Common Options

Limit eval for a quick smoke run while regenerating eval output:

```bash
make run REGEN_EVAL=1 EVAL_ARGS="--max-conversations 5 --max-combinations 30"
```

Run one strategy only:

```bash
make eval STRATEGY=hybrid
```

Use deterministic extraction mode instead of LLM generation:

```bash
make run REGEN_DATA=1 REGEN_EVAL=1 DATA_ARGS="--extract-from-source --source data/conversations.jsonl --n 36"
```

## Discoverability

```bash
make help
```

## Generated Outputs

- `results/eval_results.jsonl`
- `results/*.csv`
- `results/assignment3_findings.md`
- `plots/*`

## Notes for Windows

If `make` is unavailable in PowerShell, install GNU Make (for example via Chocolatey) or run via WSL.
