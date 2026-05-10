# Mix GPT-5.4 Mini Batch Requests

This folder contains OpenAI Batch judge/evaluation request JSONL files regenerated from the existing Mix evaluation request files with:

- `model`: `gpt-5.4-mini`
- output token parameter: `max_completion_tokens`

These are evaluation/judge requests over existing Mix answer files. They are not fresh HiRAG retrieval-context answer-generation requests, because this workspace does not currently contain an indexed Mix HiRAG graph working directory.

Files:

- `mix_eval_gpt54_mini.jsonl`: 20 requests
- `mix_eval_hi_gpt54_mini.jsonl`: 260 requests
- `mix_eval_hi_naiveR_hi_gpt54_mini.jsonl`: 260 requests

