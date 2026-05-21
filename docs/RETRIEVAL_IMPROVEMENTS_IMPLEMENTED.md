# Retrieval Improvements Implemented In This Repo

Date: 2026-05-21

This document explains what was actually implemented in this repository to improve retrieval, focusing on the active runtime path:

- [main.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/main.py)
- [src/hirag/hirag.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/hirag.py)
- [src/hirag/\_op.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/_op.py)
- [src/hirag/base.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/base.py)

It does not treat `src/rital_ir_project/` as the main runtime, because the repository handoff notes explicitly say the active retrieval path is the top-level HiRAG fork under `src/hirag/`.

## Executive Summary

The retrieval work in this repo improves HiRAG in five main ways:

1. The baseline `hi` retrieval path was preserved as a control while adding new retrieval modes.
2. Bridge retrieval was made query-aware with weighted shortest paths.
3. A stronger bridge variant was added that minimizes the worst edge on the path instead of only the total path cost.
4. Local entity retrieval was improved with late-interaction reranking after dense candidate generation.
5. The repo added supporting retrieval-quality work: bounded query-focused source snippets, bridge diagnostics, and community-balance analysis for improving the global context side.

In short:

- baseline HiRAG retrieved good entities, but bridge path selection was mostly graph-topology driven
- this fork makes the bridge itself query-conditioned
- it also improves the seed entities through reranking
- and it adds tooling to measure whether those changes really improve the retrieved context

---

## 1. Baseline HiRAG Retrieval Kept As The Control

Relevant code:

- [src/hirag/hirag.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/hirag.py)
- [src/hirag/\_op.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/_op.py)

### Intuition

Before changing retrieval, the repo first kept the original `hi` mode intact. That matters because otherwise it would be impossible to tell whether a new retrieval strategy is actually better or just different.

### Full explanation

The baseline hierarchical retrieval pipeline is:

1. Retrieve query-relevant entities from the entity vector store.
2. Keep the top local entities.
3. Retrieve communities containing those entities.
4. Select key entities inside those communities.
5. Connect those key entities through graph paths.
6. Build final context from:
   - local entities
   - global community reports
   - bridge relations
   - supporting source chunks

This structure is preserved in the shared helper [src/hirag/\_op.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/_op.py) via `_build_hierarchical_query_context_common(...)`.

### Mathematical view

Let:

- `q` be the query
- `E` be the set of graph entities
- `C` be the set of communities
- `G = (V, E_g)` be the knowledge graph

The baseline local retrieval is approximately:

$$
\mathcal{E}_{local}(q) = \operatorname{TopK}_{e \in E} \; s_{dense}(q, e)
$$

where `s_dense` is the vector-space similarity produced by the entity vector database.

The final retrieval context is then:

$$
\mathcal{R}(q) = \mathcal{L}(q) \cup \mathcal{G}(q) \cup \mathcal{B}(q) \cup \mathcal{S}(q)
$$

with:

- `L(q)` = local entity context
- `G(q)` = global community context
- `B(q)` = bridge path context
- `S(q)` = source text units

---

## 2. New Retrieval Modes Were Added

Relevant code:

- [src/hirag/base.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/base.py)
- [src/hirag/hirag.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/hirag.py)

### Intuition

Instead of replacing the baseline, the repo added explicit experimental modes so each retrieval idea could be evaluated in isolation.

### Implemented modes

The query modes now include:

- `hi`: baseline HiRAG
- `hi_weighted`: query-weighted bridge path
- `hi_minmax`: bridge path that minimizes the worst edge cost
- `hi_minmax_budgeted`: minmax bridge path with explicit path-length budgets
- `hi_rerank`: local late-interaction reranking + baseline bridge
- `hi_rerank_weighted`: local reranking + query-weighted bridge

These are wired in [src/hirag/hirag.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/hirag.py) by `_resolve_query_param(...)`.

### Why this is important

This is a real retrieval improvement, not just a CLI change:

- it turns retrieval strategy into an explicit controlled variable
- it lets the repo compare bridge-only improvements against local-retrieval improvements
- it enables hybrid variants combining both

---

## 3. Bridge Retrieval Was Made Query-Aware

Relevant code:

- [src/hirag/\_op.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/_op.py)

Main helpers:

- `_build_query_edge_costs(...)`
- `_weighted_dijkstra_path(...)`
- `_path_between(...)`

### Intuition

Baseline shortest path says:

"Use the graph-shortest route between key entities."

The improved version says:

"Use the route whose edges are most relevant to the query, not just the route with the fewest hops."

This matters because many short graph paths are semantically weak. A graph hub can connect many concepts while being useless for the actual question.

### Full explanation

For each graph edge, the code builds an edge text from:

- source entity name
- source entity description
- edge description
- target entity name
- target entity description

Then it embeds:

- the query
- all edge texts

and computes a query-conditioned edge cost. Those costs are then used in Dijkstra shortest path instead of unit edge weights.

So the bridge is no longer:

$$
\arg\min_{P} |P|
$$

It becomes:

$$
\arg\min_{P} \sum_{e \in P} c(e \mid q)
$$

where `c(e | q)` depends on the query.

### Formula implemented in the repo

The code computes:

$$
	ext{semantic\_score}(q, e) = \cos(\mathbf{q}, \mathbf{e})
$$

$$
	ext{semantic\_cost}(q, e) = \max(0, 1 - \cos(\mathbf{q}, \mathbf{e}))
$$

$$
	ext{hub\_penalty}(e) = \frac{\deg(u) + \deg(v)}{\max_{(i,j)\in E_g}(\deg(i)+\deg(j))}
$$

for edge `e = (u, v)`, and:

$$
	ext{inverse\_weight\_penalty}(e) = \frac{1}{1 + w_e}
$$

Then the total edge cost is:

$$
c(e \mid q) =
\alpha \cdot \text{semantic\_cost}(q, e)
+ \beta \cdot \text{hub\_penalty}(e)
+ \gamma \cdot \text{inverse\_weight\_penalty}(e)
$$

with defaults exposed in [src/hirag/base.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/base.py):

- `alpha = 1.0`
- `beta = 0.15`
- `gamma = 0.05`

### Why this improves retrieval

This changes bridge retrieval from purely topological search into semantic path selection. The effect is:

- query-relevant bridge edges become cheaper
- generic hub edges become more expensive
- weak relations are penalized
- the bridge context better reflects the user question

---

## 4. A Minmax Bridge Objective Was Added

Relevant code:

- [src/hirag/\_op.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/_op.py)

Main helpers:

- `_minmax_path(...)`
- `_minmax_budgeted_path(...)`

### Intuition

Weighted Dijkstra minimizes the total cost of a path. That is good, but it can still accept one very bad edge if the rest of the path is cheap.

The minmax idea instead says:

"Find a path whose worst edge is as good as possible."

This is useful for bridge retrieval because one irrelevant bridge edge can contaminate the whole reasoning chain.

### Full explanation

For a path `P = (e_1, ..., e_m)`, weighted Dijkstra optimizes:

$$
\min_P \sum_{k=1}^{m} c(e_k \mid q)
$$

The minmax strategy optimizes:

$$
\min_P \max_{e \in P} c(e \mid q)
$$

The implementation does this by:

1. Collecting all distinct edge costs.
2. Binary searching over a threshold `\tau`.
3. Keeping only edges with cost `<= \tau`.
4. Checking whether source and target are connected.
5. Taking the shortest threshold-valid path.

So the optimization is effectively:

$$
	au^* = \min \left\{ \tau : \exists P \text{ from } s \text{ to } t \text{ with } c(e \mid q) \le \tau \; \forall e \in P \right\}
$$

and then:

$$
P^* = \arg\min_{P \in \mathcal{P}_{\tau^*}(s,t)} |P|
$$

### Why this improves retrieval

For bridge retrieval, the worst edge often matters more than the average edge:

- one irrelevant jump can make the bridge explanation misleading
- minmax is conservative against that failure mode
- it tends to produce cleaner semantic chains, even if they are sometimes longer

---

## 5. A Budgeted Version Of Minmax Was Added

Relevant code:

- [src/hirag/base.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/base.py)
- [src/hirag/\_op.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/_op.py)

### Intuition

Pure minmax can produce better bridges, but sometimes they get too long and too expensive. Since retrieval is constrained by token budget and latency, the repo added explicit bridge budgets.

### Full explanation

The budgeted variant adds:

- `bridge_max_path_edges`
- `bridge_max_total_edges`
- `bridge_length_penalty`
- `bridge_budget_fallback`

The path search first tries to find a minmax path that stays within the per-segment budget:

$$
|P| - 1 \le B_{path}
$$

Across the whole multi-segment bridge chain, it also enforces:

$$
\sum_i (|P_i| - 1) \le B_{total}
$$

If the constrained minmax path fails, the code can fall back to:

- weighted path
- unweighted shortest path

depending on configuration.

### Why this improves retrieval

This is the practical version of minmax:

- it preserves the semantic-cleanliness idea
- but it keeps context size from exploding
- and it turns bridge quality vs cost into an explicit controllable tradeoff

---

## 6. Local Entity Retrieval Was Improved With Late-Interaction Reranking

Relevant code:

- [src/hirag/\_op.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/_op.py)
- [src/hirag/hirag.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/hirag.py)

Main helpers:

- `_fetch_entity_node_datas(...)`
- `_rerank_entity_node_datas(...)`
- `_late_interaction_score(...)`

### Intuition

The entire HiRAG pipeline depends on the first retrieved entities:

- they seed local context
- they decide which communities are visited
- they decide which key entities will later be bridged

So if local retrieval is weak, everything downstream is weakened too.

The implemented improvement keeps dense retrieval for recall, then reranks the candidates using a token-level late-interaction model.

### Full explanation

The pipeline becomes:

1. Dense retrieval gets a broad candidate set.
2. The candidate set size is expanded using:

$$
K_{cand} = \max(K, K \cdot m)
$$

where:

- `K = top_k`
- `m = local_candidate_multiplier`

3. The top `N` candidates are reranked using FastEmbed late interaction.
4. The reranked top `K` become the local entities used by the rest of HiRAG.

The entity text used for reranking is:

```text
entity name
type: entity type
description: entity description
```

### Mathematical formula

This reranker is ColBERT-like late interaction. If the query token embeddings are:

$$
Q = \{\mathbf{q}_1, \dots, \mathbf{q}_n\}
$$

and document/entity token embeddings are:

$$
D = \{\mathbf{d}_1, \dots, \mathbf{d}_m\}
$$

then the score is:

$$
\operatorname{LI}(Q, D) = \sum_{i=1}^{n} \max_{1 \le j \le m} \cos(\mathbf{q}_i, \mathbf{d}_j)
$$

That is exactly the structure implemented in `_late_interaction_score(...)`: token-level cosine similarity matrix, max over document tokens, then sum over query tokens.

### Why this improves retrieval

Dense single-vector retrieval compresses the whole query and entity into one vector, which can blur fine-grained term matching.

Late interaction helps because:

- each query token can find its best matching document token
- rare but important query terms matter more
- entity descriptions with the right local phrase can outrank generic high-similarity entities

This is especially useful when the user query has multi-part constraints.

---

## 7. Retrieval Order Stability Was Improved

Relevant code:

- [src/hirag/\_op.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/_op.py)

### Intuition

Some retrieval changes are not glamorous, but still important. If key entities are deduplicated in a non-deterministic order, bridge construction can vary for reasons unrelated to relevance.

### Full explanation

The repo added `_ordered_unique(...)` and uses it when selecting key entities and chaining bridge segments.

That preserves the first retrieval order instead of converting through an unordered set.

### Why this improves retrieval

It improves evaluation quality and reproducibility:

- the same query is less likely to produce a different bridge just because of dedup ordering
- retrieval differences more cleanly reflect the intended strategy change

---

## 8. Source-Chunk Retrieval Was Made More Budget-Aware And Query-Focused

Relevant code:

- [src/hirag/base.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/base.py)
- [src/hirag/\_op.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/_op.py)

Main helpers:

- `_text_unit_context_content(...)`
- `_query_overlap_snippet(...)`

### Intuition

Even if retrieval finds the right chunks, feeding whole long chunks into the answer model wastes budget. The repo added bounded snippets, plus a strategy that chooses the snippet window with the highest lexical overlap with the query.

### Full explanation

Two snippet strategies exist:

- `prefix`: keep the beginning of the chunk
- `query_overlap`: scan candidate windows and keep the one with best query overlap

The query-overlap scoring is essentially:

$$
	ext{score}(w, q) = \left(
|\operatorname{tok}(w) \cap \operatorname{tok}(q)|,\;
\frac{|\operatorname{tok}(w) \cap \operatorname{tok}(q)|}{|\operatorname{tok}(w)|},\;
-|w|
\right)
$$

and the best window is selected lexicographically.

### Why this improves retrieval

This does not change which chunk is retrieved, but it improves what part of the chunk reaches the answer model:

- more query-relevant evidence per token
- less wasted context budget
- less dilution from long irrelevant chunk prefixes

So it is a retrieval-to-generation quality improvement.

---

## 9. Edge Embedding Caching Was Added To Make Query-Aware Bridge Retrieval Practical

Relevant code:

- [src/hirag/\_op.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/_op.py)

Main helpers:

- `_edge_embedding_cache_file(...)`
- `_load_edge_embedding_disk_cache(...)`
- `_write_edge_embedding_disk_cache(...)`

### Intuition

Query-weighted bridge search is expensive because it needs edge embeddings. On a large graph, recomputing them every query would make the method too slow to use.

### Full explanation

The repo caches edge embeddings both:

- in memory
- on disk

The cache key depends on:

- graph identity
- graph size
- embedding model
- embedding dimension
- hash of the edge text

### Why this matters for retrieval improvement

This is not a ranking formula by itself, but it makes the new bridge methods operational:

- the first weighted query pays the embedding cost
- later queries can reuse edge embeddings
- that makes query-aware bridge retrieval usable in benchmarks and real experiments

---

## 10. Shared Hierarchical Context Construction Was Refactored

Relevant code:

- [src/hirag/\_op.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/_op.py)

### Intuition

When several retrieval variants exist, duplicated code becomes dangerous because "variant A vs variant B" might accidentally compare multiple hidden differences at once.

### Full explanation

The repo introduced `_build_hierarchical_query_context_common(...)` so:

- local retrieval
- community retrieval
- bridge path construction
- debug instrumentation
- final context formatting

all flow through one shared retrieval implementation.

### Why this improves retrieval experiments

It reduces confounding factors:

- bridge modes mostly differ by bridge strategy
- rerank modes mostly differ by reranking
- evaluations are more trustworthy because variant logic is centralized

---

## 11. Retrieval Diagnostics And Benchmarks Were Added

Relevant files:

- [eval/retrieval_context_benchmark.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/eval/retrieval_context_benchmark.py)
- [docs/RETRIEVAL_EXPERIMENTS.md](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/docs/RETRIEVAL_EXPERIMENTS.md)
- [docs/RETRIEVAL_RESULTS_AND_NEXT_IDEAS.md](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/docs/RETRIEVAL_RESULTS_AND_NEXT_IDEAS.md)

### Intuition

Improving retrieval is not just adding algorithms. It also requires measuring whether the retrieved context is actually more useful.

### Implemented diagnostics

The repo added benchmark outputs such as:

- latency
- context token count
- bridge edge count
- bridge path length
- mean edge score
- max edge cost
- bridge vs local token recall
- bridge vs global token recall
- bridge vs query token overlap

### Mathematical formulas

For token sets `A` and `B`, the overlap-style recall used in the benchmark is:

$$
\operatorname{Recall}(A, B) = \frac{|A \cap B|}{|B|}
$$

For example:

$$
	ext{bridge\_vs\_query\_overlap}
= \frac{|\operatorname{tok}(\text{bridge}) \cap \operatorname{tok}(q)|}{|\operatorname{tok}(q)|}
$$

### Why this matters

These diagnostics are not the retrieval algorithm, but they are part of the implemented retrieval-improvement loop:

- they reveal whether a bridge is actually query-aligned
- they make path-quality vs cost visible
- they help compare weighted, minmax, and rerank variants on the same graph

---

## 12. Community-Balance Analysis Was Added To Improve Global Retrieval Quality

Relevant files:

- [src/hirag/\_cluster_utils.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/_cluster_utils.py)
- [eval/analyze_cluster_balance.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/eval/analyze_cluster_balance.py)

### Intuition

HiRAG global retrieval depends on community reports. If communities are too large or too semantically mixed, the global context becomes noisy. That hurts retrieval even if local entity retrieval is good.

So the repo also added work to analyze and stabilize community structure.

### What was implemented

1. GMM clustering was hardened:
   - embeddings are converted to `float64`
   - `reg_covar` is passed to GaussianMixture
   - failed cluster counts are skipped
   - the algorithm retries with fewer components when fitting fails
   - if all fits fail, it falls back to one cluster

2. A cluster-balance analyzer was added:
   - flags oversized communities
   - flags high-entropy communities
   - proposes local splits using entity embeddings

### Mathematical formulas

The analyzer computes normalized entropy of entity-type distributions:

$$
H(p) = -\sum_{i=1}^{K} p_i \log p_i
$$

and normalized entropy:

$$
H_{norm}(p) = \frac{H(p)}{\log K}
$$

where `K` is the number of distinct entity types in the community.

High entropy means the community is semantically mixed; very large communities also tend to produce diffuse community reports.

The GMM side uses BIC-based model selection:

$$
\operatorname{BIC}(M_k) = -2 \log \hat{L}_k + p_k \log n
$$

where:

- `\hat{L}_k` is the fitted likelihood for `k` mixture components
- `p_k` is the number of model parameters
- `n` is the number of observations

### Why this counts as retrieval improvement

This improves the indexing side of retrieval:

- better communities produce better global reports
- better global reports produce better HiRAG global context
- more balanced communities also give better key-entity selection and cleaner bridge seeds

So this is an indirect but meaningful retrieval improvement.

---

## What Was Implemented, Distilled

If we strip away the tooling and keep only the retrieval changes that directly alter what context is returned, this repo implemented:

1. Query-conditioned bridge edge scoring.
2. Weighted shortest-path bridge retrieval.
3. Minmax bridge retrieval.
4. Budgeted minmax bridge retrieval with fallbacks.
5. Late-interaction reranking for local entity retrieval.
6. Deterministic ordered key-entity chaining.
7. Query-focused bounded source snippets.

If we include supporting work that improves retrieval quality or makes those methods usable, it also implemented:

1. Edge embedding caches for weighted bridge search.
2. Shared hierarchical retrieval helpers to keep variants comparable.
3. Retrieval diagnostics and benchmark scripts.
4. GMM hardening and cluster-balance analysis for improving global retrieval quality.

---

## Bottom Line

The main retrieval idea implemented in this fork is:

$$
	extbf{make HiRAG retrieval more query-aware at both ends}
$$

That happened in two places:

1. Upstream, by reranking local entities more precisely.
2. Downstream, by making bridge path selection depend on semantic relevance rather than only graph distance.

So the repo's retrieval improvements are not one isolated trick. They are a coherent set of changes around the two most important bottlenecks in hierarchical graph retrieval:

- choosing the right seed entities
- choosing the right connecting path between them
