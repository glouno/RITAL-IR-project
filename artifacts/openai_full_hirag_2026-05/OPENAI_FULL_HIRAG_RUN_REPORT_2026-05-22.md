# Full OpenAI HiRAG Run Report - 2026-05-22

This note summarizes the Agriculture and Mix OpenAI-only graph rebuilds up to the final graph/workdir stage, plus the currently running answer-export screens.

## Current Answer Screens

Active screens:

- `rital_agriculture_answers_20260522_125822`
- `rital_mix_answers_20260522_125822`

These screens are not rebuilding the graph anymore. They are preparing answer evaluation:

1. Load the final HiRAG workdir.
2. Retrieve contexts for `30` questions x `5` variants:
   `hi`, `naive`, `hi_nobridge`, `hi_rerank_weighted`, `hi_minmax_budgeted`.
3. Save `contexts.jsonl` and `retrieval_traces.jsonl`.
4. Export OpenAI Batch answer requests.
5. Validate and submit the answer Batch.

Why this can use local compute:

- Retrieval is local: NanoVectorDB search, graph traversal, context packing, bridge path logic, and trace creation.
- Query embeddings are live OpenAI embeddings because the final vector stores are OpenAI `text-embedding-3-small` vectors.
- `hi_rerank_weighted` may load a local FastEmbed late-interaction reranker (`answerdotai/answerai-colbert-small-v1`). Depending on FastEmbed/ONNXRuntime settings, this can use accelerator resources.
- The actual answer generation is still OpenAI Batch, not local GPU inference.

Observed GPU note:

- `nvidia-smi` currently showed only `VLLM::EngineCore` using GPU memory, likely from the older `gemma4-vllm-latency` screen.
- The answer screens were using CPU heavily while exporting retrieval contexts. If GPU activity is seen during answer export, the likely local cause is the reranker, not OpenAI Batch itself.

Current answer export progress at the time of this note:

- Agriculture request export had written `20` answer requests/traces so far.
- Mix request export had written `20` answer requests/traces so far.
- Final manifests had not yet been written, so these screens were still in export/retrieval, not Batch polling.

## Cost So Far

Pricing assumptions used for this calculation:

- `gpt-5.4-mini` Batch input: `$0.375 / 1M tokens`.
- `gpt-5.4-mini` Batch output: `$2.25 / 1M tokens`.
- Direct/live `gpt-5.4-mini` micro-retry input: `$0.75 / 1M tokens`.
- Direct/live `gpt-5.4-mini` micro-retry output: `$4.50 / 1M tokens`.
- `text-embedding-3-small`: `$0.01 / 1M input tokens`.

These costs are computed from local Batch manifests and direct-output `usage` fields. They do not include the still-running answer batches, and they do not include tiny live query-embedding calls made during answer context export.

| Dataset | Chat Batch | Embedding Batch | Direct Micro-Retries | Total So Far |
|---|---:|---:|---:|---:|
| Agriculture | `$41.95` | `$0.03` | `$0.11` | `$42.10` |
| Mix | `$23.56` | `$0.01` | `$0.03` | `$23.60` |
| Total | `$65.51` | `$0.05` | `$0.14` | `$65.70` |

Token usage:

| Dataset | Input Tokens | Output Tokens | Total Tokens |
|---|---:|---:|---:|
| Agriculture | `40,942,359` | `12,420,001` | `53,362,360` |
| Mix | `20,177,092` | `7,340,114` | `27,517,206` |
| Total | `61,119,451` | `19,760,115` | `80,879,566` |

Largest cost drivers:

- Community reports were the largest stage.
- Cluster summaries were the second largest late-stage LLM cost.
- Embeddings were very cheap relative to generation.

## Final Graph State

Final workdirs:

- `.runs/openai_full_hirag_2026-05/agriculture/10_hirag_workdir_final`
- `.runs/openai_full_hirag_2026-05/mix/10_hirag_workdir_final`

Final graph/workdir counts:

| Dataset | Docs | Chunks | Nodes | Edges | Community Reports | Entity Vectors | Chunk Vectors |
|---|---:|---:|---:|---:|---:|---:|---:|
| Agriculture | `12` | `1756` | `23224` | `48428` | `4639` | `23224` | `1756` |
| Mix | `61` | `579` | `16356` | `25960` | `2695` | `16356` | `579` |

Important caveat:

- Agriculture relation extraction has `1` manual-review row. It still ended with `finish_reason="length"` at `12000` output tokens, but the parsed best-available output contained `264` relations and was used in the final graph.

## Pipeline Steps Completed

1. Entity extraction Batch.
2. Entity import and retries.
3. Relation extraction Batch from final entities.
4. Relation import and retries.
5. Base graph assembly.
6. OpenAI embedding Batch for original entities and chunks.
7. Workdir materialization to create clustered graph/community schema.
8. Cluster summary Batch.
9. Cluster summary import and retries.
10. Apply cluster-summary nodes/edges into GraphML.
11. Community report Batch.
12. Community report import, parser hardening, retries, and final merge.
13. OpenAI embedding Batch for new summary nodes.
14. Merge original and summary-node embeddings.
15. Final workdir materialization with graph summaries and LLM community reports.
16. Answer context/export screens started.

## Retry Summary

The caps below are `max_completion_tokens` output caps, not model context-window sizes.

### Entity Extraction

| Dataset | Initial Cap | Initial Issues | Retry 1 | Retry 2 | Final |
|---|---:|---:|---:|---:|---|
| Agriculture | `2048` | `119` length | `119` at `4096`, `11` still length | `11` at `8192`, all stop | `1756/1756` OK |
| Mix | `2048` | `190` length + `12` server errors | `202` at `4096`, all stop | none | `579/579` OK |

Interpretation:

- Mix had many more entity retries proportionally because its chunks produced denser entity lists: final mean `37.6` entities/chunk vs Agriculture `23.8`.
- The `12` Mix failures were OpenAI `server_error` rows, not parsing/model failures.

### Relation Extraction

| Dataset | Initial Cap | Initial Issues | Retry 1 | Retry 2 | Final |
|---|---:|---:|---:|---:|---|
| Agriculture | `4096` | `8` length | `8` at `8192`, `1` still length | `1` at `12000`, still length | `1755/1755` OK, `1` manual review |
| Mix | `4096` | `3` length | `3` at `8192`, all stop | none | `579/579` OK |

Interpretation:

- Relation prompts can explode when a chunk has many entities and many plausible pairwise relations.
- The one stubborn Agriculture row had `24` input entities and still produced `264` parsed relations at `12000`, suggesting a very dense directory/list-style chunk.

### Cluster Summaries

| Dataset | Initial Cap | Initial Issues | Retry 1 | Retry 2 | Final |
|---|---:|---:|---:|---:|---|
| Agriculture | `2048` | `31` length + `22` parse errors | `53` at `4096`, then `1` length + `3` parse errors | `4` at `8192`, all stop/parse OK | `4639/4639` OK |
| Mix | `2048` | `73` length + `17` parse errors | `90` at `4096`, then `1` length + `1` parse error | `2` at `8192`, all stop/parse OK | `2695/2695` OK |

Interpretation:

- The length rows and parse-error rows did not overlap in the initial cluster-summary stage.
- Parse errors were `no summary entity parsed`, meaning the model returned text that did not contain a parseable summary entity/relation structure.
- Mix had more initial length rows proportionally, likely because several clusters were broad/heterogeneous and generated longer summaries.

### Community Reports

| Dataset | Initial Cap | Initial Issues | Retry 1 | Direct Micro-Retry | Final |
|---|---:|---:|---:|---:|---|
| Agriculture | `2048` | `49` length + `59` strict parse errors | `49` at `4096`, all OK | `10` malformed reports via live API, all OK | `4639/4639` OK |
| Mix | `2048` | `24` length + `27` strict parse errors | `24` at `4096`, all OK | `3` malformed reports via live API, all OK | `2695/2695` OK |

Interpretation:

- The original importer initially counted these reports as OK, but some had `findings` as a string rather than a JSON list. That produced broken markdown-like `report_string` output.
- We hardened `eval/import_openai_index_community_report_batch.py` to reject malformed `findings`.
- Most strict parse errors overlapped with length rows:
  - Agriculture: `49` of `59`.
  - Mix: `24` of `27`.
- The remaining tiny parse retries were run through the new live API micro-retry helper instead of Batch:
  - Agriculture: `10` rows.
  - Mix: `3` rows.

## Were Problem Rows Usually The Same?

Within a stage, yes: retries were targeted at the same hard rows until they either stopped cleanly or reached the maximum cap.

Across different stages, not necessarily. Entity, relation, cluster-summary, and community-report prompts stress different parts of the graph:

- Entity extraction fails/truncates when a chunk contains many distinct names/concepts.
- Relation extraction fails/truncates when entity combinations are dense.
- Cluster summaries fail/truncate when a community has many nodes/edges or when the model returns an unexpected format.
- Community reports fail/truncate when the report is long or `findings` is malformed.

The strongest recurring pattern is “dense input produces long output,” not a single universal bad document.

## Why These Steps Were Needed

The final HiRAG runtime needs more than a graph:

- Entity and relation extraction create the graph.
- Clustering creates hierarchy/community structure.
- Cluster summaries add higher-level semantic nodes/edges.
- Community reports create global/community context for HiRAG retrieval.
- Embeddings allow query-time vector search to find entry points into the graph.
- Final materialization writes the exact files HiRAG needs: GraphML, KV stores, and NanoVectorDB stores.
- Answer exports now create contexts and traces so we can compare retrieval modes scientifically.

## Next Steps

1. Wait for answer-export screens to finish and submit answer Batch jobs.
2. Import answer Batch outputs.
3. Retry answer rows if any API errors or `finish_reason="length"` occur.
4. Export pairwise judge requests comparing variants against `hi`, with swapped order.
5. Import judge outputs and compute summaries.
6. Summarize retrieval traces:
   entities, communities, bridge edges, path lengths, context token budgets, and selected text units.
7. Promote final small artifacts to `artifacts/openai_full_hirag_2026-05/`.
