# HiRAG Next Steps

Date: 2026-05-04

This note summarizes what looks most worth doing next after reading the paper, the active repo path, prior `.runs/` logs, and the exported agriculture artifacts.

## Current State

The active runtime is the top-level `main.py` using `src/hirag/`, not the scaffold in `src/rital_ir_project/`.

We have a working local-LLM fork with:

- local OpenAI-compatible chat calls through vLLM
- `fastembed` embeddings
- prompt regimes: `baseline`, `baseline_no_glean`, `lean`, `ultra_lean`
- preflight, telemetry, and curated agriculture graph exports
- query modes: `hi`, `hi_nobridge`, `hi_local`, `hi_global`, `hi_bridge`, `naive`

The strongest completed artifact is agriculture ultra-lean:

- 12 source contexts
- 1,756 chunks in the completed run
- 13,179 entities
- 15,604 relations
- 2,282 community reports
- about 61,197 seconds of indexing time
- about 16.0M total local model tokens

The exported agriculture communities are mostly small, but the tail is large:

- `ultra_lean`: median 5 nodes per community, p90 20, p99 151, max 842
- this is good evidence for testing size/entropy-based rebalancing or second-pass clustering on only the worst communities

## Paper And Implementation Gap

The paper describes HiRetrieval as:

1. retrieve top query-relevant entities as local context
2. retrieve communities containing those entities as global context
3. select top query-related entities from those communities
4. connect those selected key entities with shortest paths in the hierarchical KG
5. feed local, global, and bridge contexts to the generator

The current code follows that shape in `src/hirag/_op.py`, but with important implementation choices:

- local retrieval is pure entity-vector search through `entities_vdb.query`
- community selection is based on overlap between retrieved entities and stored `clusters`
- communities are sorted by overlap count and report rating
- bridge retrieval creates one ordered chain through deduplicated key entities
- path finding is unweighted `nx.shortest_path` or Neo4j `shortestPath`
- only after the path is found are path edges ranked/truncated

This means the bridge is currently query-aware when choosing entities, but not query-aware when choosing the route between entities.

## Recommended Priority

### 1. Build A Retrieval Evaluation Harness First

Do this before another full agriculture indexing run.

Reason:

- indexing is too expensive to use as the inner loop
- the next ideas mostly change retrieval, not extraction
- existing exported graphs and community reports are enough to prototype path selection and context construction

Minimum harness:

- load an existing graph directory or curated GraphML + community reports
- run a fixed set of queries
- emit contexts with `only_need_context=True` where possible
- compare `hi`, `hi_nobridge`, `hi_bridge`, and experimental bridge variants
- measure context token counts, path length, number of bridge edges, community count, and latency

Use a small query set first:

- 10 agriculture queries from the original QA file if available
- 5 hand-written diagnostic queries targeting known graph topics
- later extend to CS or Mix because agriculture is unusually long and slow

### 2. Query-Conditioned Shortest Path

This is the best immediate research improvement.

It directly matches the colleague note: condition shortest-path retrieval by the query by reweighting edges.

Current weakness:

- all edges have equal traversal cost during path search
- high-degree generic hubs can dominate
- an edge that is topologically short may be semantically irrelevant to the query

Suggested first implementation:

```text
edge_cost(q, u, v) =
  alpha * semantic_distance(q, edge_description + u_description + v_description)
  + beta * hub_penalty(u, v)
  + gamma * inverse_relation_weight
  + delta * generic_type_penalty
```

Start with Dijkstra via NetworkX weighted shortest path. It is simpler and easier to debug than A*.

Then test variants:

- `bridge_unweighted`: current behavior
- `bridge_weighted_query`: Dijkstra with query-conditioned edge costs
- `bridge_minmax`: path minimizing the worst edge cost, useful when one bad bridge edge contaminates the chain
- `bridge_astar`: later, if Dijkstra is too slow

A* is worth considering, but only after we have a good cost function. A non-admissible semantic heuristic can still rank paths, but then it is no longer guaranteed to be optimal in the classical sense.

Monte Carlo tree search is probably not first. It has too many knobs for our current data size and evaluation budget.

### 3. Improve Local Retrieval With Late Interaction Or Reranking

The current local retrieval step is pure dense entity retrieval. That is a brittle first domino because both global and bridge contexts are downstream of the retrieved entities.

ColBERT is a good direction because it uses late interaction:

- encode query tokens separately
- encode document/entity text tokens separately
- score by token-level max similarity rather than one pooled dense vector

Practical first step:

- keep dense retrieval as candidate generation
- rerank the top 100 or top 200 entity candidates with a ColBERT-style scorer
- feed only the reranked top 20 into the existing HiRetrieval pipeline

Candidate text for reranking:

- entity name
- entity type
- entity description
- optionally source chunk snippets

If ColBERT dependency/runtime is painful, use a cross-encoder reranker or FastEmbed reranker as a fallback. The experimental claim can still be "local retrieval reranking" first, then ColBERT if integration is clean.

### 4. Second-Pass Clustering With Entropy And Confidence

This is the best indexing-side improvement, but it is more expensive than retrieval changes.

Motivation:

- the artifact community tail is very large
- current GMM can produce unstable clusters
- a baseline run with gleaning failed in GMM due to ill-defined covariance, which suggests we need more robust clustering anyway

Metrics to add:

- cluster size
- normalized entropy of cluster membership/topic distribution
- GMM confidence: mean max probability
- GMM ambiguity: mean gap between top-1 and top-2 probabilities
- embedding variance inside the cluster

Possible rule:

- split if `size > max_size` or entropy/variance is high
- merge if `size < min_size` and nearest sibling is close enough
- keep unchanged if size and confidence are in range

This is the "self-balancing tree" idea, but for semantic density rather than key order.

Do not rebuild the whole hierarchy at first. Run second-pass clustering only on the top 1-5% worst communities/clusters, then measure whether retrieval contexts become less noisy.

### 5. Stabilize GMM Before Bigger Runs

The failed baseline-with-gleaning run hit:

```text
ValueError: Fitting the mixture model failed because some components have ill-defined empirical covariance
```

Near-term hardening:

- pass `reg_covar` to `GaussianMixture`
- ensure embeddings are `float64` for GMM
- retry with fewer components when covariance fails
- log BIC candidates and selected cluster count

This is not the most interesting research contribution, but it is important engineering hygiene before expensive reruns.

## What Not To Prioritize Yet

RL/bandits should not be the next move.

Reason:

- action space is not obvious
- reward would probably need an LLM judge
- the current query budget is around 100 queries, which is too small and noisy
- it would be hard to distinguish learning from reward noise

Keep it as a discussion/future-work idea, not the first implementation.

Full dynamic self-balancing updates should also wait.

Reason:

- incremental insertion/rebalancing would affect node IDs, cached paths, and community reports
- a local second-pass rebalancer gives most of the research signal with much less blast radius

## Suggested Experiment Order

1. Retrieval context harness on existing agriculture artifacts.
2. Weighted query-conditioned bridge path on current graph.
3. Local retrieval reranking before HiRetrieval.
4. Evaluate bridge and reranking variants on 10-20 queries.
5. Add GMM hardening.
6. Prototype entropy/confidence second-pass clustering on the worst large communities.
7. Only then consider a new full indexing run.

## Success Criteria

For retrieval experiments:

- shorter or equal bridge paths with higher query relevance
- fewer generic bridge edges
- no major latency explosion
- better answer/context preference in small manual or LLM-judge evaluation

For clustering experiments:

- fewer huge communities
- better cluster confidence/entropy distribution
- community reports become more focused
- retrieval contexts include fewer unrelated topics

## Best Research Story

The strongest project story is:

HiRAG introduces a hierarchical graph and three-level retrieval, but its bridge path is still mostly topology-driven. We improve HiRetrieval by making the bridge query-conditioned and by strengthening the local entity retrieval that seeds both global and bridge context. As a second direction, we make the hierarchy more robust by detecting oversized or low-confidence clusters and rebalancing only those regions.

