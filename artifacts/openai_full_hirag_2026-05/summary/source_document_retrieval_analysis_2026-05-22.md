# Source Document Retrieval Analysis - Full OpenAI HiRAG Runs

Generated on 2026-05-22 from the promoted Agriculture and Mix OpenAI-full artifacts.

## Question

Compare the original source-document chunks/snippets placed in the final answer context by `naive` and by baseline `hi`: are they the same chunks, and does HiRAG filter source evidence differently?

## Method

- `naive`: split the answer context on `--New Chunk--`, then remap each chunk to `kv_store_text_chunks.json`.
- `hi`: parse the `-----Source Documents-----` section, then remap each source snippet to `kv_store_text_chunks.json` by substring matching.
- Compare chunk-id overlap per query: intersection, HiRAG-only, naive-only, and Jaccard similarity.
- Compare snippet length against the full original chunk length to estimate how much of each retrieved chunk is actually sent.

## Aggregate Results

| Dataset | Queries | Mean hi source docs | Mean naive source docs | Mean hi source chars | Mean naive source chars | Mean hi context tokens | Mean naive context tokens | Mean chunk Jaccard | Median Jaccard | Zero-overlap queries | Mean hi-only chunks | Mean naive-only chunks | Mean hi/full chunk ratio | Mean naive/full chunk ratio | Mapping misses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| agriculture | 30 | 4.27 | 5.00 | 4020 | 29321 | 5040 | 6016 | 0.199 | 0.155 | 8 | 2.77 | 3.60 | 0.166 | 1.000 | 0/278 |
| mix | 30 | 4.30 | 6.00 | 4013 | 29644 | 3174 | 6614 | 0.477 | 0.500 | 2 | 1.10 | 3.00 | 0.208 | 1.000 | 0/309 |

## Query-Aware Snippet Smoke

This compares the old prefix snippet and query-overlap snippet on the same dev smoke query. It isolates snippet-selection behavior; it is not a full final-run comparison.

| Strategy/run | Context tokens | Input tokens | Source docs | Source chars | Context chars |
|---|---:|---:|---:|---:|---:|
| prefix | 4144 | 4403 | 4 | 4816 | 14327 |
| query_overlap | 3544 | 3799 | 4 | 2697 | 12208 |

- On this smoke, query-overlap reduced context tokens by 600 tokens (14.5%).

## Interpretation

- HiRAG is not retrieving exactly the same source chunks as naive. The chunk-id Jaccard overlap is low, meaning the graph/entity retrieval path usually leads to different source evidence than direct chunk-vector search.
- HiRAG filters source documents through retrieved entities: source chunks come from selected entity `source_id`s, then are sorted using relation overlap with one-hop neighbors. That is graph-conditioned source selection, not ordinary top-k chunk retrieval.
- In the final runs, HiRAG sends far fewer source-document characters than naive, because our answer export uses bounded query-aware snippets. However, HiRAG also sends entity descriptions, community reports, and bridge relation descriptions, so total context tokens are not simply “source chunk tokens”.
- Low source-chunk overlap does not prove HiRAG is better. The final judge results often favored naive, so direct chunk retrieval was very competitive for these datasets/questions.
- The most likely story is: HiRAG changes the evidence distribution substantially, but for these QA-style questions, direct retrieval of original chunks often gives the answer model more verbatim evidence.

## Files

- Per-query overlap table: `artifacts/openai_full_hirag_2026-05/summary/source_document_overlap_2026-05-22.csv`
- Final contexts: `artifacts/openai_full_hirag_2026-05/{agriculture,mix}/retrieval/contexts.jsonl`
- Retrieval traces: `artifacts/openai_full_hirag_2026-05/{agriculture,mix}/retrieval/retrieval_traces.jsonl`
