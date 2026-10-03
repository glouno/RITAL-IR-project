# HiRAG: reproduction and retrieval experiments

A Sorbonne RITAL information-retrieval project by [Paul Béglin](https://github.com/glouno) and [BshKatrin](https://github.com/BshKatrin).

We reproduce and extend [HiRAG](https://arxiv.org/abs/2503.10150), a retrieval-augmented generation system that organizes documents into a hierarchical knowledge graph. Our question: **can query-aware graph traversal give an answer model better evidence?**

**Start here:** [Final presentation](Présentation%20HiRAG_final.pdf) · [Project summary and findings](Fiche_Rendu.md)

## What we built

Starting from the [official HiRAG implementation](https://github.com/hhy-huang/HiRAG), we added:

- **Query-aware retrieval:** weighted shortest paths, minimax and budgeted minimax bridges, entity reranking, and Monte Carlo Tree Search (MCTS).
- **A reproducible experiment pipeline:** OpenAI Batch tooling for graph construction, embeddings, answers, and judging, with retries and cost tracking; a local OpenAI-compatible LLM + FastEmbed route is also available.
- **Evidence diagnostics:** traces of retrieved entities, paths, source snippets, and context budgets.
- **Evaluation beyond a single score:** pairwise LLM judging in both answer orders and human annotation of preferences, answer relevance, and useful text spans.

## Experiments and findings

The final experiments use the **Agriculture and Mix** subsets of [UltraDomain](https://huggingface.co/datasets/TommyChien/UltraDomain), with `gpt-5.4-mini` for answers/judging and `text-embedding-3-small` for embeddings. CS and Legal inputs are included, but were not rebuilt in the final campaign. Model changes and the smaller domain coverage make this an experimental reproduction, rather than an exact replication of the paper.

Naive chunk retrieval remained a strong baseline. Our graph variants changed the paths retrieved, but their answer prompts often contained less direct source evidence; community-report budgets also affected the comparisons. Human evaluation found MCTS close to the HiRAG baseline, while naive retrieval was preferred more often. The main direction for improvement is to use the graph to select evidence while retaining the source passages needed to answer the question.

For the protocols, results, and limitations:

- [Core 7 evaluation summary](artifacts/openai_full_hirag_2026-05/core7_eval/summary/CORE7_EVALUATION_SUMMARY.md)
- [Naive vs graph retrieval analysis](artifacts/openai_full_hirag_2026-05/core7_eval/summary/NAIVE_VS_GRAPH_RETRIEVAL_ANALYSIS.md)
- [Human evaluation](artifacts/openai_full_hirag_2026-05/core7_eval/human_annotation_results/HUMAN_ANNOTATION_ANALYSIS.md)

## Setup

Requires **Python 3.13** and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
```

Runtime settings are documented in [.env.example](.env.example). Copy it to `.env` when needed and set paths/backend credentials locally. `uv sync` also installs the development dependencies used by the tests.

### Local HiRAG runtime

Top-level `main.py` uses an OpenAI-compatible chat server, such as vLLM, and local FastEmbed embeddings. Start the server separately; the example assumes it is available at `http://127.0.0.1:8000/v1`. The chat model can be discovered from the server, or specified with `--chat-model`.

Index the bundled, deduplicated Mix contexts:

```bash
uv run python main.py \
  --context-file eval/datasets/mix/mix_unique_contexts.json \
  --knowledge-graph-path .runs/graphs \
  --graph-vis-path .runs/vis \
  --graph-name mix \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY
```

`EMPTY` is a placeholder for a local server without authentication; use your server's credentials if required. Indexing long documents can take substantial time. Use the `*_unique_contexts.json` arrays for this route, rather than passing raw QA JSONL files as context text.

### OpenAI Batch experiments

The final experiment campaign uses the tools under `eval/`. Follow the [OpenAI Batch runbook](docs/OPENAI_FULL_HIRAG_RUNBOOK.md) and [Core 7 evaluation runbook](docs/CORE7_EVAL_RUNBOOK_2026-05-23.md) for graph construction, retrieval variants, answer generation, and judging. These runs require an OpenAI API key and incur API costs. Existing results can be inspected without rerunning them.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/hirag/` | Active HiRAG fork and retrieval extensions |
| `main.py` | Local LLM + FastEmbed indexing/query entry point |
| `eval/` | Dataset inputs, batch tooling, and evaluation scripts |
| `artifacts/` | Saved results, metrics, traces, and figures |
| `docs/` | Runtime guides and algorithm descriptions |
| `notebooks/` | Retrieval analysis notebooks |
| `src/rital_ir_project/` | Early regex/TF-IDF scaffold for learning and small demos |
| `tests/` | Unit tests for the scaffold, runtime helpers, and retrieval extensions |

The early scaffold remains available through `uv run python -m src.main`; it is separate from the runtime used for the final experiments. For example:

```bash
uv run python -m src.main datasets list
```

Run the unit tests with:

```bash
uv run pytest
```

## Attribution

HiRAG's original code and MIT copyright notice are retained in [LICENSE](LICENSE). See [FORK_CHANGES.md](FORK_CHANGES.md) for our changes and [README_HIRAG.md](README_HIRAG.md) for the upstream documentation.
