# PersonaMem0

PersonaMem benchmark and analysis workspace for two-stage Mem0 evaluation:

- Stage 1: ingest user chat history into memory
- Stage 2: run benchmark questions against stored memory and evaluate answers

## Repository Layout

- `src/personamem/` core benchmark and Mem0 integration modules
- `scripts/` helper scripts for plotting and dataset overlap analysis
- `notebooks/` exploratory and analysis notebooks
- `data/inputs/` benchmark JSONL inputs used by runs
- `artifacts/benchmark_logs/` generated benchmark run outputs
- `artifacts/analysis_exports/` generated charts/tables/text exports
- `docs/` setup notes (including Neo4j configuration)

Compatibility wrappers remain at repo root (`experiment_runner.py`, `personamem_benchmark.py`, etc.) so existing notebook imports and CLI habits continue to work.

## Quick Start

This project uses `pyproject.toml` (not `requirements.txt`) for dependency management.

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install dependencies:

```bash
pip install -e .
```

Optional dev extras:

```bash
pip install -e ".[dev]"
```

3. Copy `.env.example` to `.env` and set required secrets (at minimum OpenAI key; add Neo4j values for graph mode).

## .env Setup

Create `.env` from the example:

```bash
cp .env.example .env
```

Set these values:

- `OPENAI_API_KEY`: required for Mem0/OpenAI calls.
- `NEO4J_URL`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`: required when running with graph enabled.
- `PERSONAMEM_BENCHMARK_LOGS_DIR` (optional): override output log directory.

If you use `--enable-graph`, ensure Neo4j is installed and running before launching experiments.

## Running the Benchmark

### Sanity check (run this first)

Before a full run, do a quick smoke test with small limits to verify `.env`, Neo4j, and output paths:

Text sanity check:

```bash
python run_text_experiment.py \
  --stage both \
  --enable-graph \
  --max-users 2 \
  --max-qa-per-user 5
```

Multimodal sanity check:

```bash
python run_multimodal_experiment.py \
  --stage both \
  --enable-graph \
  --max-users 2 \
  --max-qa-per-user 5
```

### Notebook-equivalent launch examples

These commands mirror the notebook-style defaults (graph enabled, `graph_retries=3`, stage1 sync behavior by default, `max_concurrent=5`, no user caps).

Text benchmark (`notebooks/experiment_personamem.ipynb` equivalent):

```bash
python run_text_experiment.py \
  --stage both \
  --enable-graph \
  --experiment-id text_gpt-4.1.-mini_graph \
  --jsonl data/inputs/personamem_benchmark_text_32k_user_bundles.jsonl \
  --graph-retries 3 \
  --max-concurrent 5
```

Multimodal benchmark (`notebooks/experiment_personamem_multi.ipynb` equivalent):

```bash
python run_multimodal_experiment.py \
  --stage both \
  --enable-graph \
  --experiment-id multi_gpt-4.1.-mini_graph-img-preserved-2 \
  --jsonl data/inputs/personamem_benchmark_multimodal_32k_user_bundles.jsonl \
  --graph-retries 3 \
  --max-concurrent 5
```

Multimodal Stage 2 resume example (as in notebook resume usage):

```bash
python run_multimodal_experiment.py \
  --stage 2 \
  --enable-graph \
  --experiment-id multi_gpt-4.1.-mini_graph-img-preserved-2 \
  --jsonl data/inputs/personamem_benchmark_multimodal_32k_user_bundles.jsonl \
  --resume-from-stream artifacts/benchmark_logs/multi_gpt-4.1.-mini_graph-img-preserved-2/stage2_RUN-2_qa_20260312_003449_stream.jsonl
```

Legacy runner (still supported):

```bash
python personamem_benchmark.py both --enable-graph
```

Use custom input JSONL (example for text):

```bash
python run_text_experiment.py --stage both --enable-graph --jsonl data/inputs/personamem_benchmark_text_32k_user_bundles.jsonl
```

Outputs are written under `artifacts/benchmark_logs/<experiment_id>/` (or a legacy logs directory if explicitly configured).

## Results and Artifacts

Main benchmark results are stored per experiment in:

- `artifacts/benchmark_logs/<experiment_id>/`

Typical files produced:

- Stage 1:
  - `stage1_RUN-*_fill_*.json` (full stage-1 log)
  - `stage1_RUN-*_errors_*.jsonl` (per-error lines)
  - `flow_stage1_RUN-*.log` (step-by-step flow log)
- Stage 2:
  - `stage2_RUN-*_qa_*.json` (full stage-2 log)
  - `stage2_RUN-*_qa_*_stream.jsonl` (streamed QA rows, resume-friendly)
  - `stage2_RUN-*_qa_*_answers.json` (answers grouped by user)
  - `flow_stage2_RUN-*.log` (step-by-step flow log)
- Additional analysis artifacts created by notebooks/scripts:
  - `checked_results.json`
  - `stage2_unified_from_logs.csv`
  - `stage2_latency_plots.png`
  - `analysis_exports/*.csv`

Cross-experiment exports from helper scripts are stored in:

- `artifacts/analysis_exports/`

## Helpful Scripts

- Question overlap Venn:
  - `python scripts/personamem_question_overlap_venn.py`
- Stage 2 latency plots:
  - `python scripts/plot_stage2_latency.py`
- Multimodal image inspection:
  - `python scripts/view_multimodal_images.py --show 1`

## Evaluation Notebook

LLM-as-a-judge evaluation notebook:

- `notebooks/llm_as_a_judge.ipynb`

It typically reads a Stage 2 QA log from `artifacts/benchmark_logs/<experiment_id>/` and writes judged outputs such as:

- `artifacts/benchmark_logs/<experiment_id>/checked_results.json`

## Notes

- `docs/NEO4J_SETUP.md` contains graph-memory setup details.
- `data/inputs/` stores source benchmark bundles; generated outputs go to `artifacts/`.
