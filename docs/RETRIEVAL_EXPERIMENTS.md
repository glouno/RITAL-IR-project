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

Implemented control note:

- `hi` remains the baseline full-context mode.
- The only intentional control-path cleanup is deterministic ordered deduplication for key entities.

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
beta = 0.15
gamma = 0.05
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

Implemented v1:

- `fastembed.LateInteractionTextEmbedding`
- default model: `answerdotai/answerai-colbert-small-v1`
- dense entity search remains candidate generation
- reranked entities feed the same local/global/bridge context builder
- if model initialization or embedding fails, retrieval logs a warning and preserves dense ordering

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

## Implemented Modes

The active experimental modes are:

- `hi`: baseline HiRAG.
- `hi_weighted`: baseline local retrieval plus query-weighted bridge paths.
- `hi_minmax`: baseline local retrieval plus minmax bridge paths.
- `hi_rerank`: FastEmbed late-interaction local reranking plus baseline bridge.
- `hi_rerank_weighted`: reranked local retrieval plus query-weighted bridge paths.

## Smoke Results

Command:

```bash
uv run python eval/retrieval_context_benchmark.py \
  --working-dir .runs/2026-04-22-agri-resume2/graphs/benchmark_ultra_lean_run1_20260422_204326 \
  --query-file eval/datasets/agriculture/agriculture_query.jsonl \
  --query-limit 10 \
  --output-dir .runs/retrieval_eval/dev_retrieval_smoke
```

Outputs:

- `contexts.jsonl`
- `metrics.csv`
- `summary.json`
- `summary.md`

Aggregate observations from the first smoke:

- `hi`: mean latency 0.12s, bridge/query token overlap 0.63.
- `hi_weighted`: mean latency 13.78s because the first query builds the edge embedding cache; bridge/query overlap rose to 0.67.
- `hi_minmax`: mean latency 8.24s, longer paths, bridge/query overlap rose to 0.72, max edge cost was lower than weighted Dijkstra.
- `hi_rerank`: mean latency 0.76s and changed local entities enough to reduce bridge edge count.
- `hi_rerank_weighted`: mean latency 1.01s after caches were warm, with similar bridge/query overlap to rerank-only.

Interpretation:

- `hi_minmax` is the most promising bridge-only direction by deterministic context metrics, but it expands path/context size.
- `hi_rerank` changes the seed entities substantially and should be checked manually on representative queries.
- The first weighted query is still expensive; a future optimization should compute edge costs only on a candidate subgraph or persist edge embeddings.

## Cluster-Balance Smoke

Command:

```bash
uv run python eval/analyze_cluster_balance.py \
  --graphml artifacts/agriculture_graphs_2026-04-23/graphs/agriculture_ultra_lean.graphml \
  --community-reports artifacts/agriculture_graphs_2026-04-23/metrics/agriculture_ultra_lean_community_reports.json \
  --output-dir .runs/cluster_balance/dev_retrieval_smoke
```

Outputs:

- `community_balance.csv`
- `oversized_communities.json`
- `split_plan.json`

Smoke observation:

- 1,742 communities exceeded the default size/entropy thresholds.
- The largest selected candidate had 842 nodes.
- The default split plan embeds one candidate and caps embedded members at 100 for smoke-test runtime.
- The first candidate proposed one local GMM cluster, which suggests type entropy alone is not enough to decide that a community is semantically splittable.
