# Mix OpenAI Batch Indexing Requests

This folder contains the Mix graph-indexing Batch API request files generated during the GPT-5.4 Mini indexing run.

## Entity Extraction

- `mix_entity_extract_gpt54_mini_1024/`: full 579-row entity extraction batch with `max_completion_tokens=1024`.
- `mix_entity_extract_gpt54_mini_retry_2048/`: 251-row retry batch for rows from the 1024 run that ended with `finish_reason=length`, using `max_completion_tokens=2048`.

The merged imported entity result is generated locally under `.runs/openai_index_batch/mix_entity_import_gpt54_mini_merged/` by replacing the truncated 1024 rows with retry rows.

## Relation Extraction

- `mix_relation_extract_gpt54_mini_from_merged_entities_2048/`: 579-row relation extraction batch generated from the merged entity import, using `max_completion_tokens=2048`.
- `mix_relation_extract_gpt54_mini_retry_4096/`: 71-row retry batch for rows from the 2048 relation run that ended with `finish_reason=length`, using `max_completion_tokens=4096`.

The final merged relation import is generated locally under `.runs/openai_index_batch/mix_relation_import_gpt54_mini_merged_retry/` by replacing the truncated 2048 rows with retry rows.

The final flat relationship graph artifact is:

- `artifacts/mix_batch_requests_2026-05-10/OpenAI_batch_indexing_graphs/mix_base_graph_gpt54_mini_final/base_graph.graphml`
