# Extension Proposed: Dynamic Graph Traversal for HiRAG

## Goal

The current HiRAG retrieval pipeline is mostly static once the query enters the
graph. A query is embedded, the nearest entity seeds are selected, and fixed
local/global/bridge rules assemble the final context.

The proposed extension is to keep HiRAG's graph structure and budget discipline,
but make community and path selection dynamic:

- choose which communities to inspect based on the question;
- descend only into useful sub-communities;
- expand graph neighborhoods only when needed;
- keep a fixed token and latency budget;
- compare the result against `hi` and `hi_rerank_weighted`.

This is more defensible than free-form Cypher generation because it remains close
to HiRAG's retrieval design while adding a controlled decision layer.

## Current Retrieval Hyperparameters

The active parameters are defined in `src/hirag/base.py` in `QueryParam`.

### Shared Entry Point

HiRAG first retrieves entity seeds from the entity vector DB:

- `top_k`: number of final local seed entities, default `20`.
- `query_better_than_threshold`: NanoVectorDB cosine threshold, default `0.2`,
  configured on `HiRAG`, not `QueryParam`.
- `local_candidate_multiplier`: extra dense candidates before reranking,
  default `10`.
- `local_rerank_strategy`: `none` or `fastembed_late_interaction`.
- `local_rerank_top_n`: number of dense candidates reranked, default `100`.

There is no explicit graph-distance threshold for local retrieval. "Close" first
means semantically close to the query in embedding space. The graph is then used
to fetch one-hop relations, source chunks, communities, and bridge paths around
those seed entities.

### Local Retrieval

Local retrieval uses:

- seed entities from `entities_vdb.query(query, top_k=top_k)`;
- all one-hop edges touching those entities;
- source chunks attached to the seed entities;
- source chunks from one-hop neighbors, ranked by relation overlap;
- `max_token_for_local_context`, default `20000`;
- `max_token_for_text_unit`, default `20000`.

So local retrieval is not a radius-`d` graph traversal. It is seed entity
retrieval plus one-hop neighborhood evidence and token-budget truncation.

### Global Retrieval

Global retrieval starts from seed entities and selects community reports:

- `level`: maximum community level accepted, default `2`;
- `top_k`: seed entities used to find communities;
- community ranking: first by how often retrieved entities map to the community,
  then by community report `rating`;
- `max_token_for_community_report`, default `12500`;
- `community_single_one`: if true, keep only the best community.

In the current implementation, global retrieval does not search communities from
the top of the hierarchy downward. It finds communities attached to the seed
entities and filters/ranks them.

### Bridge Retrieval

Bridge retrieval selects key entities and connects them through graph paths:

- `top_m`: number of key entities selected per retrieved community, default `10`;
- `bridge_strategy`: `unweighted`, `query_weighted`, `minmax`, or
  `minmax_budgeted`;
- `bridge_alpha`: semantic edge-cost weight, default `1.0`;
- `bridge_beta`: hub penalty weight, default `0.15`;
- `bridge_gamma`: inverse relation-weight penalty, default `0.05`;
- `bridge_max_path_edges`: per-path budget for `minmax_budgeted`, default `120`;
- `bridge_max_total_edges`: total bridge budget, default `220`;
- `bridge_budget_fallback`: fallback strategy, default `weighted`;
- `max_token_for_bridge_knowledge`, default `12500`.

In baseline `hi`, paths are unweighted shortest paths between selected key
entities. In `hi_weighted` / `hi_rerank_weighted`, each graph edge gets a
query-dependent cost:

```text
cost =
  bridge_alpha * semantic_cost
  + bridge_beta * hub_penalty
  + bridge_gamma * inverse_weight_penalty
```

This is already a mild form of dynamic traversal, but only at the edge-cost
level. It does not decide interactively which communities or neighborhoods to
inspect next.

## Level 2: Dynamic Community Traversal

This is the most natural next step.

Instead of selecting communities only because they contain retrieved seed
entities, the retriever would traverse the community hierarchy with a small
scoring model.

Proposed flow:

1. Retrieve initial seed entities with the current vector search and optional
   reranker.
2. Collect candidate high-level communities from seed entities and possibly from
   their parent communities.
3. Ask a small model or scoring prompt to rate each community report against the
   query.
4. Descend only into sub-communities that are likely to help answer the query.
5. Stop when the token budget, depth budget, or marginal relevance threshold is
   reached.
6. Build the final global context from selected community reports and selected
   supporting entities/chunks.

Possible tool API:

```python
search_entities(query: str, top_k: int) -> list[Entity]
get_communities(entity: str, max_level: int) -> list[Community]
get_subcommunities(community_id: str) -> list[Community]
score_community(query: str, community: Community) -> float
select_context(query: str, candidates: list[ContextItem], budget: int) -> list[ContextItem]
```

The important design choice is that the LLM should not write arbitrary graph
queries. It should choose among typed candidates returned by deterministic
functions. That keeps the system reproducible and limits hallucinated filters.

Why this is attractive:

- it directly improves the local/global boundary in HiRAG;
- it stays close to the paper's hierarchical community idea;
- it can be evaluated with the same answer/judge pipeline;
- it adds only a small number of extra LLM calls if batched or cached;
- it can expose a trace: selected community, score, reason, token cost.

Main risks:

- extra inference cost at query time;
- model scoring can over-select broad communities;
- results are less deterministic unless scores and prompts are cached;
- if the community reports are weak, dynamic traversal has weak supervision.

Recommended first experiment:

- Add `mode="hi_dynamic_global"` or `mode="hi_dynamic_community"`.
- Keep `top_k=20`, `top_m=10`.
- Limit to at most 8 community scoring calls per query.
- Keep a global community report budget of `12500` tokens.
- Compare against `hi` and `hi_rerank_weighted` on Agriculture and Mix.

## Level 3: Agentic Graph Traversal

This is the more ambitious version.

Here, the retriever becomes a small agent loop. It sees the query, the current
partial context, and a set of graph tools. It decides whether to expand,
inspect, connect, rerank, or stop.

Possible tool API:

```python
search_entities(query: str, top_k: int) -> list[Entity]
get_neighbors(entity: str, relation_filter: str | None, depth: int) -> list[Entity | Edge]
get_communities(entity: str) -> list[Community]
get_paths(entity_a: str, entity_b: str, strategy: str) -> list[Path]
rate_context(query: str, candidate_context: str) -> float
expand_or_stop(current_context: str, budget: int) -> Decision
```

The LLM should not be trusted to invent arbitrary low-level filters. A safer
pattern is:

- the LLM writes an intent, such as "find causal relations about irrigation";
- a deterministic normalizer maps the intent to allowed relation categories,
  keywords, or no filter;
- the graph tool applies the normalized filter;
- the LLM receives candidates and chooses what to keep.

For `relation_filter`, practical options are:

- no filter, then rerank returned edges semantically;
- keyword filter over relation descriptions;
- embedding similarity between the query/filter text and edge descriptions;
- a small fixed taxonomy, such as `causes`, `uses`, `located_in`, `affects`,
  `part_of`, `compares_with`;
- LLM classification into that taxonomy, with fallback to no filter.

In other words, the LLM can formulate the search intent, but the code should
constrain how that intent becomes graph traversal.

Possible loop:

1. Start with dense seed entities.
2. Summarize current known context and missing evidence.
3. Choose one action from a fixed action schema.
4. Execute the graph tool deterministically.
5. Rerank candidates by query relevance and novelty.
6. Add only selected evidence to the context.
7. Stop after budget, confidence, or max steps.

Suggested hard limits:

- max 3 to 5 tool steps per query;
- max depth 2 for neighborhood expansion;
- max 3 path queries;
- max 200 candidate edges inspected;
- final answer context capped at the same token budget as `hi_rerank_weighted`.

Why this could work:

- multi-hop questions often need evidence not found by the first seed entities;
- bridge paths can be chosen based on what the answer is missing;
- dynamic traversal can avoid stuffing irrelevant community reports into the
  prompt;
- traces make retrieval errors easier to debug.

Why it may not be comparable:

- it becomes agentic retrieval rather than pure HiRAG retrieval;
- retrieval latency increases;
- stochastic tool choices complicate evaluation;
- a stronger LLM planner can hide weaknesses in the graph construction;
- cost depends on the number of loop steps.

Recommended use:

- Treat this as a second research track after dynamic community traversal.
- Do not use it as the main final comparison unless we explicitly report it as
  an agentic extension.

## Proposed Project Extension

The most defensible improvement is:

> Add dynamic community and path selection, with controlled budget, on top of
> `hi_rerank_weighted`.

This preserves the strongest result from our current project while addressing
the core weakness: fixed retrieval may pull the wrong communities or bridge
paths even when the graph is good.

Concrete implementation plan:

1. Add debug/export mode for current `hi_rerank_weighted` contexts.
2. Implement deterministic candidate collectors for communities,
   sub-communities, neighbors, and paths.
3. Add a small LLM/embedding scorer for community relevance and novelty.
4. Build `hi_dynamic_community` with the same final context format as `hi`.
5. Add optional dynamic bridge path selection only after community selection is
   stable.
6. Evaluate on Agriculture and Mix with the existing answer and judge scripts.

Success criteria:

- same or lower final context token budget than `hi_rerank_weighted`;
- better pairwise judge win rate against `hi`;
- better swapped-order agreement than current `hi_minmax_budgeted`;
- trace files showing selected communities, paths, scores, and budget use;
- no unbounded free-form graph queries.

## Open Questions

- Should community scoring use `gpt-5.4-mini`, embeddings, or both?
- Should traversal optimize answer relevance, evidence diversity, or path
  compactness?
- Should we precompute community embeddings to avoid scoring every report by LLM?
- Should the dynamic system be allowed to inspect raw chunks, or only graph
  entities/reports/edges?
- How much extra query latency is acceptable for the final benchmark?
