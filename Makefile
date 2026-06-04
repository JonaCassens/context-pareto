IMAGE_NAME ?= lec-eval:latest
ENV_FILE ?= .env

ifeq ($(wildcard $(ENV_FILE)),)
ENV_FILE_ARG :=
else
ENV_FILE_ARG := --env-file $(ENV_FILE)
endif

DOCKER_RUN = docker run --rm $(ENV_FILE_ARG) -e GOOGLE_API_KEY -e ANTHROPIC_API_KEY -v "$(CURDIR):/work" -w /work $(IMAGE_NAME)

.PHONY: help build init-env run run-regen check-env ensure-data data eval report clean

help:
	@echo "LEC Docker Pipeline"
	@echo ""
	@echo "Targets:"
	@echo "  make build                 Build Docker image"
	@echo "  make init-env              Create .env from .env.example if missing"
	@echo "  make run                   Default: generate plots only from existing eval results"
	@echo "  make run-regen             Convenience: regenerate data + eval + summaries + findings"
	@echo "  make eval                  Run evaluation only"
	@echo "  make report                Run reporting only"
	@echo "  make data                  Regenerate dataset only"
	@echo "  make clean                 Remove generated result artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  IMAGE_NAME=<name:tag>      Docker image name (default: lec-eval:latest)"
	@echo "  ENV_FILE=<path>            Env file passed to docker --env-file (default: .env if present)"
	@echo "                             Host GOOGLE_API_KEY/ANTHROPIC_API_KEY are also forwarded if set"
	@echo "  DATA_ARGS='<args>'         Extra args for src.dataset"
	@echo "  EVAL_ARGS='<args>'         Extra args for src.eval"
	@echo "  REPORT_ARGS='<args>'       Extra args for src.report"
	@echo "  STRATEGY=<name>            Shortcut for eval strategy: sliding_window|summarization|hybrid"
	@echo "  REGEN_DATA=1               Optional: regenerate conversations before run"
	@echo "  REGEN_EVAL=1               Optional: regenerate eval_results before run"
	@echo "  REGEN_SUMMARIES=1          Optional: regenerate summary CSV tables"
	@echo "  REGEN_FINDINGS=1           Optional: regenerate assignment3_findings.md"
	@echo ""
	@echo "Examples:"
	@echo "  make run"
	@echo "  make run REGEN_SUMMARIES=1 REGEN_FINDINGS=1"
	@echo "  make run REGEN_EVAL=1 EVAL_ARGS='--max-conversations 5 --max-combinations 30'"
	@echo "  make run REGEN_DATA=1 REGEN_EVAL=1 REGEN_SUMMARIES=1 REGEN_FINDINGS=1"
	@echo "  make eval STRATEGY=hybrid"

build:
	docker build -t $(IMAGE_NAME) .

init-env:
	@if [ -f .env ]; then \
		echo ".env already exists"; \
	elif [ -f .env.example ]; then \
		cp .env.example .env; \
		echo "Created .env from .env.example - update the API key values before running"; \
	else \
		echo "No .env.example found. Create .env manually with GOOGLE_API_KEY or ANTHROPIC_API_KEY"; \
	fi

check-env:
	@if [ -n "$$GOOGLE_API_KEY" ] || [ -n "$$ANTHROPIC_API_KEY" ]; then \
		if [ "$$GOOGLE_API_KEY" = "sk-..." ] || [ "$$ANTHROPIC_API_KEY" = "sk-ant-..." ]; then \
			echo "Host environment API key appears to be a placeholder. Replace with a real key."; \
			exit 1; \
		fi; \
		echo "Using API key from host environment"; \
	elif [ -f "$(ENV_FILE)" ] && (grep -Eq '^[[:space:]]*GOOGLE_API_KEY=.+$$' "$(ENV_FILE)" || grep -Eq '^[[:space:]]*ANTHROPIC_API_KEY=.+$$' "$(ENV_FILE)"); then \
		if grep -Eq '^[[:space:]]*GOOGLE_API_KEY=sk-\.\.\.[[:space:]]*$$' "$(ENV_FILE)" || grep -Eq '^[[:space:]]*ANTHROPIC_API_KEY=sk-ant-\.\.\.[[:space:]]*$$' "$(ENV_FILE)"; then \
			echo "$(ENV_FILE) contains placeholder API key value(s). Replace with real key(s)."; \
			exit 1; \
		fi; \
		echo "Using API key from $(ENV_FILE)"; \
	else \
		echo "Missing API key. Set GOOGLE_API_KEY or ANTHROPIC_API_KEY in $(ENV_FILE) or host env."; \
		echo "Tip: run 'make init-env' to create .env from .env.example"; \
		exit 1; \
	fi

ensure-data: check-env
	@if [ -f data/conversations.jsonl ]; then \
		echo "Using existing data/conversations.jsonl"; \
	else \
		echo "Dataset missing, generating data/conversations.jsonl"; \
		$(MAKE) data; \
	fi

data: build
	$(DOCKER_RUN) python -m src.dataset $(DATA_ARGS)

eval: build ensure-data
	$(DOCKER_RUN) python -m src.eval $(if $(STRATEGY),--strategy $(STRATEGY),) $(EVAL_ARGS)

report: build
	$(DOCKER_RUN) python -m src.report $(REPORT_ARGS)

run: build
	@report_args="$(REPORT_ARGS)"; \
	if [ "$(REGEN_DATA)" = "1" ]; then \
		$(MAKE) data DATA_ARGS="$(DATA_ARGS)"; \
	fi; \
	if [ "$(REGEN_EVAL)" = "1" ]; then \
		$(MAKE) eval EVAL_ARGS="$(EVAL_ARGS)" STRATEGY="$(STRATEGY)"; \
	fi; \
	if [ "$(REGEN_SUMMARIES)" = "1" ]; then \
		report_args="$$report_args --regen-summaries"; \
	fi; \
	if [ "$(REGEN_FINDINGS)" = "1" ]; then \
		report_args="$$report_args --regen-findings"; \
	fi; \
	$(MAKE) report REPORT_ARGS="$$report_args"

run-regen: build
	$(MAKE) run \
		REGEN_DATA=1 \
		REGEN_EVAL=1 \
		REGEN_SUMMARIES=1 \
		REGEN_FINDINGS=1 \
		DATA_ARGS="$(DATA_ARGS)" \
		EVAL_ARGS="$(EVAL_ARGS)" \
		STRATEGY="$(STRATEGY)" \
		REPORT_ARGS="$(REPORT_ARGS)"

clean:
	@if [ -d results ]; then find results -mindepth 1 -maxdepth 1 -type f -delete; fi
	@if [ -d plots ]; then find plots -mindepth 1 -maxdepth 1 -type f -delete; fi
	@echo "Cleaned files in results/ and plots/"
