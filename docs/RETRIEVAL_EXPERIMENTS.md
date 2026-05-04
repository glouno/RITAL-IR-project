# Retrieval Experiment Notes

Date: 2026-05-04

This file keeps the concrete experiment design separate from the higher-level plan in `docs/NEXT_STEPS.md`.

## Baseline To Preserve

The current `hi` retrieval path in `src/hirag/_op.py` should be preserved as the control.

Behavior to compare against:

- dense entity retrieval: `entities_vdb.query(query, top_k=top_k * 10)`
- local context: first `top_k` entity nodes
- global context: community reports attached to local entities
- key entities: top retrieved entities that are members of each selected community
- bridge: one unweighted shortest-path chain through deduplicated key entities

## Experiment A: Query-Weighted Bridge

Goal:

- make bridge paths prefer relations that are semantically relevant to the query

Candidate edge text:

```text
source entity name
source entity description
edge description
target entity name
target entity description
```

Cost sketch:

```text
semantic_cost = 1 - cosine(embed(query), embed(edge_text))
hub_cost = log(1 + degree(source) + degree(target))
relation_cost = 1 / log(2 + edge_weight)
cost = alpha * semantic_cost + beta * hub_cost + gamma * relation_cost
```

Initial weights:

```text
alpha = 1.0
beta = 0.05
gamma = 0.1
```

Implementation note:

- precompute edge texts lazily per query
- use the existing embedding function for query and edge text batches
- call `nx.shortest_path(graph, source, target, weight=weight_fn)` or `nx.dijkstra_path`
- for Neo4j, prototype in NetworkX first; port Cypher/GDS later only if needed

Measurements:

- path node count
- path edge count
- mean edge semantic score
- max edge semantic cost
- number of high-degree hubs used
- context token count
- query latency

## Experiment B: Minmax Bridge

Goal:

- avoid paths where one very bad edge joins otherwise good context

Idea:

- score every candidate edge with the query-conditioned cost
- find a path minimizing the maximum edge cost

Simple implementation:

- sort unique edge costs
- binary search a threshold
- keep only edges with cost <= threshold
- find the first threshold where source and target are connected
- among threshold-valid paths, choose shortest length

Why this may help:

- bridge context quality can be damaged by a single unrelated generic edge
- minmax makes the path conservative about worst-edge relevance

## Experiment C: A* Bridge

Use only if weighted Dijkstra is too slow.

Possible heuristic:

```text
h(node, target) = 1 - cosine(embed(query), embed(node_description + target_description))
```

Caution:

- this heuristic is probably not admissible
- that is acceptable for retrieval ranking, but we should not claim classical optimality

## Experiment D: MCTS Bridge

Do not implement first.

Possible action space:

- expand from current node to one neighbor
- stop when target reached or budget exhausted

Reward:

- query relevance of edge text
- reaching target
- path compactness
- penalty for generic hubs

Why it is risky now:

- many hyperparameters
- expensive per query
- hard to evaluate with the current query budget

Keep this for discussion or future work.

## Experiment E: Local Retrieval Reranking

Goal:

- improve the top local entities that seed community and bridge retrieval

Candidate generation:

- keep existing dense entity retrieval with `top_k * 10` or `top_k * 20`

Reranking options:

- ColBERT late interaction over entity text
- cross-encoder reranker
- FastEmbed reranker if dependency friction is lower

Entity text:

```text
{entity_name}
type: {entity_type}
description: {description}
```

Metrics:

- overlap with baseline top entities
- source chunk coverage
- answer/context quality on the same query set
- added latency

## Experiment F: Bridge Coverage Diagnostic

The paper reports token-level recall between bridge context and local/global context. We can implement a cheaper diagnostic:

- tokenize local entity descriptions
- tokenize community reports
- tokenize bridge relation descriptions
- compute token overlap ratios after stopword filtering

This is not a full quality metric, but it helps verify whether the bridge is actually connecting both sides rather than adding unrelated path text.

## Small Query Set

Start with:

- 10 original agriculture queries if the QA JSONL is present
- 5 hand-written graph-inspection queries

Run every retrieval variant on exactly the same query set.

Recommended first outputs:

- one JSONL record per query/variant
- one markdown qualitative report with 3-5 representative cases
- one CSV summary for path/context metrics

