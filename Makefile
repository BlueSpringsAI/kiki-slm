.PHONY: install install-dev install-gpu train-sft train-alignment evaluate serve serve-mlx \
       merge-export download-data docker-build docker-up docker-down clean help

UV := uv
PYTHON := $(UV) run python

# ─── Installation ────────────────────────────────────────────────────────────

install:  ## Install core dependencies
	$(UV) pip install -e .

install-dev:  ## Install with dev dependencies
	$(UV) pip install -e ".[dev]"

install-gpu:  ## Install with GPU training dependencies
	$(UV) pip install -e ".[gpu]"

install-all:  ## Install all dependencies
	$(UV) pip install -e ".[dev,gpu,apple]"

# ─── Training ────────────────────────────────────────────────────────────────

train-sft-intent:  ## Train intent classifier SFT adapter
	$(PYTHON) scripts/train_sft.py --config configs/sft/intent_classifier.yaml

train-sft-workflow:  ## Train workflow reasoner SFT adapter
	$(PYTHON) scripts/train_sft.py --config configs/sft/workflow_reasoner.yaml

train-sft-tools:  ## Train tool caller SFT adapter
	$(PYTHON) scripts/train_sft.py --config configs/sft/tool_caller.yaml

train-sft-response:  ## Train response generator SFT adapter
	$(PYTHON) scripts/train_sft.py --config configs/sft/response_generator.yaml

train-sft: train-sft-intent train-sft-workflow train-sft-tools train-sft-response  ## Train all SFT adapters

train-dpo:  ## Run DPO alignment
	$(PYTHON) scripts/train_alignment.py --config configs/alignment/dpo.yaml

train-grpo:  ## Run GRPO alignment
	$(PYTHON) scripts/train_alignment.py --config configs/alignment/grpo.yaml

train-kto:  ## Run KTO safety alignment
	$(PYTHON) scripts/train_alignment.py --config configs/alignment/kto.yaml

train-alignment: train-dpo train-grpo train-kto  ## Run full alignment pipeline

train-all: train-sft train-alignment  ## Run full training pipeline (SFT + alignment)

# ─── Evaluation ──────────────────────────────────────────────────────────────

evaluate:  ## Run full evaluation suite
	$(PYTHON) scripts/evaluate.py --config configs/evaluation/eval_suite.yaml

# ─── Serving ─────────────────────────────────────────────────────────────────

serve:  ## Launch vLLM server with multi-LoRA
	$(PYTHON) scripts/serve.py --config configs/serving/vllm_multi_lora.yaml

serve-mlx:  ## Launch MLX server (Apple Silicon)
	$(PYTHON) scripts/serve.py --mlx

# ─── Export ──────────────────────────────────────────────────────────────────

merge-gguf:  ## Merge adapter and export to GGUF
	$(PYTHON) scripts/merge_and_export.py --adapter-path runs/grpo-v1 --format gguf --output-dir outputs/exports/kiki-gguf

merge-mlx:  ## Merge adapter and export to MLX
	$(PYTHON) scripts/merge_and_export.py --adapter-path runs/grpo-v1 --format mlx --output-dir outputs/exports/kiki-mlx

# ─── Data ────────────────────────────────────────────────────────────────────

download-data:  ## Download all training datasets
	$(PYTHON) scripts/download_datasets.py

prepare-data:  ## Run full data preparation pipeline
	$(PYTHON) scripts/prepare_data.py

# ─── Docker ──────────────────────────────────────────────────────────────────

docker-build:  ## Build Docker image
	docker build -t kiki-slm .

docker-up:  ## Start all services (vLLM + Prometheus + Grafana)
	docker compose up -d

docker-down:  ## Stop all services
	docker compose down

# ─── Utilities ───────────────────────────────────────────────────────────────

clean:  ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info .pytest_cache/

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
