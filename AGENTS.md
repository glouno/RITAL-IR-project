# Repo Handoff

## What This Repo Is

This repo mixes two different implementation tracks:

- `src/hirag/` is the paper-style HiRAG codebase and is the active runtime used by the top-level [`main.py`](/home/paulbeglin/projects/RITAL-IR-project/main.py).
- `src/rital_ir_project/` is a separate local scaffold/reproduction attempt built with regex extraction, TF-IDF embeddings, and simple hierarchy/retrieval helpers. It is useful for reading/tests, but it is not the path currently used for the real HiRAG + vLLM runs.

If the goal is to run the current fork against a local OpenAI-compatible backend, start from top-level `main.py`, not `src/rital_ir_project/cli.py`.

## Change Tracking

Use [`FORK_CHANGES.md`](/home/paulbeglin/projects/RITAL-IR-project/FORK_CHANGES.md) as the running changelog for this fork.

When making future changes:

- update `FORK_CHANGES.md` in the same change set
- summarize behavior or experiment impact, not just file edits
- keep `AGENTS.md` focused on orientation and workflow, and keep detailed fork history in `FORK_CHANGES.md`

## Active Execution Path

The current indexing/query path is:

1. [`main.py`](/home/paulbeglin/projects/RITAL-IR-project/main.py)
2. [`src/hirag/hirag.py`](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/hirag.py)
3. [`src/hirag/_op.py`](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/_op.py)
4. [`src/hirag/prompt.py`](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/prompt.py)
5. [`src/hirag/_storage/`](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/_storage)

The important fork-specific change is in top-level `main.py`:

- chat completions are routed through a local OpenAI-compatible endpoint such as vLLM
- embeddings are routed through `fastembed`
- the script accepts JSON lists of contexts directly, which is what the `*_unique_contexts.json` files are
- both `best_model_func` and `cheap_model_func` are set to the same local chat backend

## Dataset Rule

Use the `*_unique_contexts.json` files for indexing runs.

Do not use the raw `*.jsonl` QA files when the goal is graph construction / indexing cost estimation. The repo already contains extracted unique-context files under:

- `eval/datasets/agriculture/agriculture_unique_contexts.json`
- `eval/datasets/cs/cs_unique_contexts.json`
- `eval/datasets/legal/legal_unique_contexts.json`
- `eval/datasets/mix/mix_unique_contexts.json`

These files are JSON arrays of strings. `main.py` accepts them directly via `--context-file`.

The extraction helper that produced them is [`eval/extract_context.py`](/home/paulbeglin/projects/RITAL-IR-project/eval/extract_context.py).

## Current Dataset Sizes

As inspected on 2026-04-20:

- `agriculture_unique_contexts.json`: 12 contexts
- `cs_unique_contexts.json`: 10 contexts
- `legal_unique_contexts.json`: 94 contexts
- `mix_unique_contexts.json`: 61 contexts

With the active HiRAG chunking settings in [`src/hirag/hirag.py`](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/hirag.py):

- `chunk_token_size=1200`
- `chunk_overlap_token_size=100`
- tokenizer metadata uses `gpt-4o`

Approximate exact chunk counts from the current code path are:

- agriculture: 1756 chunks
- cs: 1858 chunks
- legal: 4336 chunks
- mix: 579 chunks

This matters because indexing cost scales much more with chunk count than with document count.

## What Is Custom In This Fork

The pieces that appear to be added or adapted for the local RITAL fork are:

- top-level [`main.py`](/home/paulbeglin/projects/RITAL-IR-project/main.py): new CLI wrapper around HiRAG with local vLLM + FastEmbed runtime wiring
- [`.env.example`](/home/paulbeglin/projects/RITAL-IR-project/.env.example): local runtime config template
- `eval/datasets/.../*_unique_contexts.json`: pre-extracted indexing inputs
- `src/rital_ir_project/`: separate scaffold implementation, not the active runtime for paper-style runs
- tests in [`tests/`](/home/paulbeglin/projects/RITAL-IR-project/tests) only cover the scaffold code in `src/rital_ir_project`, not the top-level `main.py` + `src/hirag` path

## Important Runtime Notes

- There is no checked-in `.env` file right now. Either copy `.env.example` or pass arguments explicitly.
- Use `uv run python ...`. Plain `python` was not available on `PATH` in this environment.
- `main.py` requires real values for `--knowledge-graph-path` and `--graph-vis-path`. If they are omitted and not supplied through env, `Path(None)` will fail.
- `--working-dir` exists, but the script actually writes graphs under `--knowledge-graph-path/<graph-name>_<timestamp>`.
- `--graph-vis-path` is currently just created; the main script does not write visualizations there.
- `main.py` can auto-discover the chat model from `/v1/models`, but passing `--chat-model` is safer for reproducibility.

## Recommended Command Pattern

Check the local model list first:

```bash
curl -sS http://127.0.0.1:8000/v1/models
```

Run indexing on one of the unique-context datasets:

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

Swap the `--context-file` and `--graph-name` values for `cs`, `legal`, or `mix` as needed.

## Query Modes

The active query modes exposed by `main.py` are:

- `hi`
- `naive`
- `hi_nobridge`
- `hi_local`
- `hi_global`
- `hi_bridge`

These map to `QueryParam.mode` in [`src/hirag/base.py`](/home/paulbeglin/projects/RITAL-IR-project/src/hirag/base.py).

## Legacy / Secondary Paths

These are present but should not be mistaken for the main runtime:

- `hi_Search_openai.py`, `hi_Search_glm.py`, `hi_Search_deepseek.py`: older single-file examples using provider-specific config
- `eval/insert_context_openai.py`, `eval/insert_context_glm.py`, `eval/insert_context_deepseek.py`: older eval helpers that assume execution from `eval/` and dataset-local paths
- `src/rital_ir_project/cli.py`: local scaffold CLI, useful for small synthetic experiments, not for the active vLLM-based HiRAG path

## Known Practical Risk

The `unique_contexts` files are large. Even though agriculture has only 12 contexts, they are very long documents. A single representative agriculture context produced 143 chunks in the active pipeline, so full-dataset runs can take many hours on a local LLM backend.
