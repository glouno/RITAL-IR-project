# RITAL IR Project: HiRAG Reproduction and Retrieval Experiments

This repository contains our implementation work for a course project on
hierarchical graph retrieval for retrieval-augmented generation (RAG). The main
code path is based on HiRAG, "Retrieval-Augmented Generation with Hierarchical
Knowledge", and extends it with local execution support, retrieval variants,
evaluation scripts, and analysis artifacts.

The project focuses on three questions:

- how HiRAG builds and queries a hierarchical knowledge graph;
- how graph retrieval compares with a dense text-only baseline on UltraDomain;
- whether modified bridge-retrieval strategies improve answer quality or reduce
  noisy graph context.

## Repository Structure

```text
.
├── main.py                       # Main indexing/query entry point
├── src/hirag/                    # Active HiRAG implementation
├── eval/                         # Dataset preparation and evaluation scripts
├── tests/                        # Unit tests for retrieval/evaluation helpers
├── docs/                         # Method notes and experiment documentation
├── artifacts/                    # Curated result summaries and selected outputs
├── imgs/                         # Figures from the original HiRAG repository
├── pyproject.toml                # Python dependencies and package metadata
└── .env.example                  # Local runtime configuration template
```

There is also a secondary package under `src/rital_ir_project/`. It is an early
scaffold used for small local experiments with simpler extraction and retrieval
logic. The experiments reported for this project use the top-level `main.py`
script and `src/hirag/`.

## Main Implementation

The active runtime is:

1. `main.py`
2. `src/hirag/hirag.py`
3. `src/hirag/_op.py`
4. `src/hirag/prompt.py`
5. `src/hirag/_storage/`

The top-level script configures HiRAG for a local or OpenAI-compatible runtime:

- chat completions use an OpenAI-compatible endpoint, such as vLLM;
- embeddings can be generated with FastEmbed for local indexing runs;
- context inputs can be plain text files or JSON lists of context strings;
- both high-capacity and low-cost model hooks are routed through the configured
  chat backend for reproducible local experiments.

## Datasets

The main experiments use UltraDomain subsets:

- Agriculture
- Mix
- Computer Science
- Legal

For graph construction and indexing-cost estimates, use the extracted
`*_unique_contexts.json` files rather than the raw QA JSONL files:

```text
eval/datasets/agriculture/agriculture_unique_contexts.json
eval/datasets/cs/cs_unique_contexts.json
eval/datasets/legal/legal_unique_contexts.json
eval/datasets/mix/mix_unique_contexts.json
```

These files are JSON arrays of unique source contexts. The helper that produced
them is `eval/extract_context.py`.

## Query Modes

The main query modes exposed by `main.py` are:

| Mode | Purpose |
|---|---|
| `hi` | Default HiRAG retrieval with local, global, and bridge context |
| `naive` | Dense text-only retrieval baseline |
| `hi_nobridge` | HiRAG retrieval without bridge context |
| `hi_local` | Local entity context only |
| `hi_global` | Community/global context only |
| `hi_bridge` | Bridge context only |
| `hi_weighted` | Query-weighted bridge scoring |
| `hi_minmax` | Minimax bridge path strategy |
| `hi_minmax_budgeted` | Budget-aware minimax bridge strategy |
| `hi_rerank` | Local candidate reranking |
| `hi_rerank_weighted` | Reranking plus weighted bridge scoring |
| `hi_mcts` | Experimental MCTS bridge traversal |

## Setup

This project uses `uv`.

```bash
uv sync
```

The project currently targets Python 3.13 through the constraints in
`pyproject.toml`.

Copy the example environment file if you prefer environment-variable
configuration:

```bash
cp .env.example .env
```

The `.env` file is intentionally ignored by Git.

## Running an Indexing Experiment

First check that the local OpenAI-compatible chat backend is available:

```bash
curl -sS http://127.0.0.1:8000/v1/models
```

Then run indexing on one unique-context dataset:

```bash
uv run python main.py \
  --context-file eval/datasets/agriculture/agriculture_unique_contexts.json \
  --knowledge-graph-path /tmp/hirag_graphs \
  --graph-vis-path /tmp/hirag_vis \
  --graph-name agriculture_full \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --chat-model nvidia/Gemma-4-31B-IT-NVFP4 \
  --embed-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --embed-dim 384 \
  --max-token-size 8192 \
  --embedding-batch-num 6 \
  --embedding-func-max-async 8 \
  --fastembed-cache-path /tmp/fastembed_cache \
  --log-level INFO
```

The graph output is written under:

```text
<knowledge-graph-path>/<graph-name>_<timestamp>/
```

Use `--query` and `--query-mode` to run a query after indexing:

```bash
uv run python main.py \
  --context-file eval/datasets/mix/mix_unique_contexts.json \
  --knowledge-graph-path /tmp/hirag_graphs \
  --graph-vis-path /tmp/hirag_vis \
  --graph-name mix_full \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --chat-model nvidia/Gemma-4-31B-IT-NVFP4 \
  --query "What is the main claim supported by the retrieved evidence?" \
  --query-mode hi
```

## Evaluation Workflow

The `eval/` directory contains scripts for:

- extracting unique contexts;
- benchmarking retrieval context composition;
- exporting and importing batch answer-generation requests;
- running pairwise answer judging;
- summarizing retrieval traces and final metrics;
- building human annotation packets and analysis figures.

The most relevant result summaries are under:

```text
artifacts/openai_full_hirag_2026-05/
artifacts/openai_full_hirag_2026-05/summary/
artifacts/openai_full_hirag_2026-05/core7_eval/summary/
artifacts/openai_full_hirag_2026-05/core7_eval/human_annotation_results/
```

Start with:

- `artifacts/openai_full_hirag_2026-05/summary/FINAL_OPENAI_FULL_HIRAG_RESULTS_2026-05-22.md`
- `artifacts/openai_full_hirag_2026-05/core7_eval/summary/CORE7_EVALUATION_SUMMARY.md`
- `artifacts/openai_full_hirag_2026-05/core7_eval/human_annotation_results/HUMAN_ANNOTATION_ANALYSIS.md`

## Main Experimental Finding

In the final Agriculture and Mix runs, the dense text-only `naive` baseline was
preferred by the pairwise judge more often than graph-based HiRAG variants. The
`hi_nobridge` ablation also outperformed default `hi` in these runs, suggesting
that bridge context can add noise when graph paths are not tightly aligned with
the question. The weighted, reranked, and budgeted bridge variants provide
useful diagnostics but did not robustly outperform the simpler baselines.

These results should be interpreted as pairwise LLM-judge preferences, not as
ground-truth factual accuracy. The human annotation analysis and retrieval trace
summaries provide additional evidence about where graph retrieval helps or
hurts.

## Tests

Run the test suite with:

```bash
uv run pytest
```

For a faster check while editing retrieval or evaluation code:

```bash
uv run pytest tests/test_hirag_retrieval_experiments.py tests/test_answer_level_evaluation.py
```

## Notes for Reviewers

Large source datasets and local model caches are not required to inspect the
code. The checked-in artifacts are selected outputs intended to make the final
experiments inspectable without rerunning the full indexing pipeline.

Full graph construction on the unique-context files can take several hours on a
local LLM backend because the source contexts are long and produce many chunks.
