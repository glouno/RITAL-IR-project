# Core 7 Evaluation Summary

Scope: Agriculture + Mix, OpenAI-full HiRAG workdirs, Core 7 variants, `gpt-5.4-mini` answers and pairwise judge, `text-embedding-3-small` embeddings.

Important baseline note: `hi` is our fork baseline with query-aware source snippets enabled, not a byte-for-byte untouched upstream HiRAG baseline.

## Run Health

- Answer generation: Agriculture `210/210`, Mix `210/210`, 0 API errors, 0 empty answers.
- LLM judge: Agriculture `360/360`, Mix `360/360`, 0 API failed rows, 0 parse errors.
- Edge embeddings: imported from Batch into runtime edge cache, Agriculture `48428/48428`, Mix `25960/25960`, 0 import errors.

## Best Custom

- Selected `best_custom`: `hi_minmax_budgeted`.
- Selection rule: highest combined strict swapped-order wins vs `hi` among `hi_weighted`, `hi_minmax`, `hi_minmax_budgeted`, `hi_rerank_weighted`; ties use lower mean retrieval time, then lower mean context tokens.

## Judge Interpretation

- Raw win rate counts each judge request independently, so each query contributes two judgments with answer order swapped.
- Strict win rate counts a query only when both swapped orders agree after mapping A/B back to variants; disagreement is reported separately.
- `naive` strongly beats `hi` by the LLM judge in this run, which likely means the direct chunk baseline often gives more answerable evidence than the current graph context for these QA-style questions.

## Key Raw Win Rates vs hi

### Agriculture
- `naive`: raw win rate `0.800`, strict wins/losses/disagree `21/3/6`, mean retrieval `0.50s`, mean context tokens `6016`.
- `hi_weighted`: raw win rate `0.467`, strict wins/losses/disagree `10/12/8`, mean retrieval `2.47s`, mean context tokens `4977`.
- `hi_minmax`: raw win rate `0.383`, strict wins/losses/disagree `6/11/13`, mean retrieval `5.13s`, mean context tokens `5141`.
- `hi_minmax_budgeted`: raw win rate `0.483`, strict wins/losses/disagree `11/11/8`, mean retrieval `5.01s`, mean context tokens `5141`.
- `hi_rerank_weighted`: raw win rate `0.400`, strict wins/losses/disagree `8/14/8`, mean retrieval `7.54s`, mean context tokens `5801`.
- `hi_mcts`: raw win rate `0.500`, strict wins/losses/disagree `11/11/8`, mean retrieval `2.03s`, mean context tokens `4990`.

### Mix
- `naive`: raw win rate `0.750`, strict wins/losses/disagree `19/4/7`, mean retrieval `0.40s`, mean context tokens `6614`.
- `hi_weighted`: raw win rate `0.550`, strict wins/losses/disagree `10/7/13`, mean retrieval `1.18s`, mean context tokens `3153`.
- `hi_minmax`: raw win rate `0.517`, strict wins/losses/disagree `9/7/14`, mean retrieval `2.49s`, mean context tokens `3206`.
- `hi_minmax_budgeted`: raw win rate `0.567`, strict wins/losses/disagree `11/7/12`, mean retrieval `2.43s`, mean context tokens `3206`.
- `hi_rerank_weighted`: raw win rate `0.500`, strict wins/losses/disagree `11/11/8`, mean retrieval `5.01s`, mean context tokens `3597`.
- `hi_mcts`: raw win rate `0.450`, strict wins/losses/disagree `6/8/15`, mean retrieval `0.96s`, mean context tokens `3159`.

