# Bridge Retrieval And Source Snippets

Date: 2026-05-23

This note is a report-ready summary of the retrieval mechanisms implemented in the active HiRAG fork:

- bounded source-document snippets
- query-weighted edge costs
- `hi_weighted` with Dijkstra
- `hi_minmax` and `hi_minmax_budgeted`

Implementation references:

- [src/hirag/base.py](/home/bshkatrin/RITAL-IR-project/src/hirag/base.py)
- [src/hirag/hirag.py](/home/bshkatrin/RITAL-IR-project/src/hirag/hirag.py)
- [src/hirag/_op.py](/home/bshkatrin/RITAL-IR-project/src/hirag/_op.py)
- [eval/eval_utils.py](/home/bshkatrin/RITAL-IR-project/eval/eval_utils.py)

## 1. Bounded Source-Document Snippets

### Goal

HiRAG retrieves source text units (chunks) as evidence. Some chunks are very long, so sending the full chunk to the answer model wastes context budget. The fork therefore adds a bounded-snippet mechanism that keeps the retrieved chunk but only passes a short excerpt of it.

Important implementation detail:

- chunk retrieval still happens at the chunk level
- the snippet is applied only when building the final prompt context
- if `text_unit_snippet_chars` is `None`, the full chunk is kept
- in compact answer-generation scripts, the repo typically sets `text_unit_snippet_chars = 1200`

### Step 1: retrieve source chunks under a token budget

The source chunks are first selected by the standard retrieval logic and then truncated as a list by `max_token_for_text_unit`.

If the retrieved chunk list is

$$
\mathcal{T}(q) = \{t_1, t_2, \dots, t_n\},
$$

the implementation keeps only the longest prefix whose total tokenized content stays under the text-unit budget:

$$
\sum_{t \in \mathcal{T}'(q)} \operatorname{Tok}(t) \le B_{\text{text}}.
$$

So there are two levels of control:

1. chunk-list budget: keep only a budgeted set of retrieved chunks
2. per-chunk snippet budget: optionally shorten each kept chunk before insertion into the final prompt

### Step 2: bound each chunk to at most `snippet_chars`

For a chunk text $x$ and a character budget $L$:

$$
\operatorname{snippet}(x) =
\begin{cases}
x & \text{if } |x| \le L \\
x_{1:L} \text{ + " ..."} & \text{for prefix mode}
\end{cases}
$$

This is the `prefix` strategy.

### Step 3: query-overlap snippet selection

The more important strategy is `query_overlap`. Instead of always taking the beginning of the chunk, the code searches for the bounded window that overlaps most with the query.

Let:

- $Q$ = set of normalized query tokens
- $W$ = candidate snippet window
- $T(W)$ = set of normalized tokens in window $W$

Normalization in code means:

- lowercase
- regex tokenization with `[a-z0-9]+`
- remove one-character tokens
- remove a small stopword list

The chunk is split into sentence-like pieces with punctuation and newline boundaries. If one piece is still too long, it is further split into overlapping subwindows of length $L$ with stride about $L/2$.

The algorithm then builds candidate windows whose total length stays within `snippet_chars`, and scores each candidate lexicographically by:

$$
\operatorname{overlap}(Q, W) = |Q \cap T(W)|
$$

$$
\operatorname{density}(Q, W) = \frac{|Q \cap T(W)|}{\max(1, |T(W)|)}
$$

$$
\operatorname{score}(W) =
\Big(
\operatorname{overlap}(Q, W),
\operatorname{density}(Q, W),
-|W|
\Big)
$$

The selected snippet is

$$
W^* = \arg\max_W \operatorname{score}(W).
$$

So the implementation prefers:

1. the window with the most query-token overlap
2. among ties, the densest window
3. among ties, the shorter window

### Intuition

This design solves a practical retrieval problem: the retrieved chunk may be correct, but its beginning may be irrelevant boilerplate. Query-overlap snippets keep evidence from the same retrieved chunk while spending the budget on the most query-relevant part of that chunk.

In short:

- we do not change which chunk was retrieved
- we change which part of that chunk is shown to the answer model

---

## 2. Edge Cost Definition

### Goal

The original unweighted bridge path uses graph topology only. That can produce short but semantically weak bridges. The fork therefore defines a query-conditioned edge cost and uses it in path search.

### Edge text used for scoring

For each graph edge $e = (u,v)$, the embedded text is built from:

1. source entity name
2. source entity description
3. relation description
4. target entity name
5. target entity description

So the semantic comparison is not just between the query and a relation label, but between the query and a richer description of the whole edge context.

### Formula

Let $\mathbf{q}$ be the query embedding and $\mathbf{e}$ the edge-text embedding.

The code computes:

$$
\operatorname{semantic\_score}(q,e) = \cos(\mathbf{q}, \mathbf{e})
$$

$$
\operatorname{semantic\_cost}(q,e) = \max\big(0,\ 1 - \operatorname{semantic\_score}(q,e)\big)
$$

For edge $e=(u,v)$, let $\deg(\cdot)$ be graph degree. Then:

$$
\operatorname{hub\_penalty}(e)
=
\frac{\deg(u)+\deg(v)}
{\max_{(i,j)\in E}\big(\deg(i)+\deg(j)\big)}
$$

If the graph edge has stored weight $w_e$, the code also computes:

$$
\operatorname{inverse\_weight\_penalty}(e) = \frac{1}{1+w_e}
$$

The final cost is:

$$
c(e \mid q)
=
\alpha \cdot \operatorname{semantic\_cost}(q,e)
+
\beta \cdot \operatorname{hub\_penalty}(e)
+
\gamma \cdot \operatorname{inverse\_weight\_penalty}(e)
$$

with defaults:

$$
\alpha = 1.0,\quad \beta = 0.15,\quad \gamma = 0.05
$$

### Why these terms were chosen

The choice is intuitive and asymmetric on purpose:

- semantic cost is the main term, because relevance to the query should dominate bridge selection
- hub penalty discourages generic hub nodes that connect many things but often add weak explanatory value
- inverse weight penalty favors stronger graph relations when semantic evidence is similar

The default coefficients reflect this priority:

- `alpha` is much larger than `beta` and `gamma`
- semantic relevance drives the path
- topology and relation strength act as regularizers

### Interpretation

An edge is cheap when:

- its textual meaning matches the query well
- it does not pass through a highly generic hub
- it corresponds to a stronger graph relation

So lower cost means "better bridge evidence for this specific query."

---

## 3. `hi_weighted`: Query-Weighted Dijkstra

### Objective

`hi_weighted` uses Dijkstra on the same graph, but replaces unit edge weights with the query-conditioned cost above.

For a path

$$
P = (e_1, e_2, \dots, e_m),
$$

the objective is:

$$
P^* = \arg\min_P \sum_{k=1}^{m} c(e_k \mid q)
$$

This is different from ordinary shortest path:

$$
\arg\min_P |P|
$$

because a slightly longer path can now win if its edges are much more relevant to the query.

### Algorithm steps

For each consecutive pair of required key entities:

1. Build query-conditioned costs for all graph edges.
2. Use Dijkstra with weight function $c(e \mid q)$.
3. Return the minimum-total-cost path between source and target.
4. Concatenate the paths found between consecutive key-entity pairs.

In pseudocode:

$$
\text{dist}(v) = \min_{P:s\to v} \sum_{e\in P} c(e \mid q)
$$

and the chosen bridge path to target $t$ is the path that minimizes `dist(t)`.

### Intuition

Weighted Dijkstra is a strong first improvement because it is:

- deterministic
- easy to inspect
- globally optimal for additive path cost

Its limitation is also clear: it optimizes the sum of costs, so it may still accept one very bad edge if the remaining edges are cheap enough.

---

## 4. `hi_minmax`: Minimize The Worst Edge

Terminology note:

- `hi_minmax` is the non-budgeted minimax mode
- `hi_minmax_budgeted` is the budgeted minimax mode

### Objective

Instead of minimizing total path cost, `hi_minmax` minimizes the worst edge on the path:

$$
P^* = \arg\min_P \max_{e \in P} c(e \mid q)
$$

This is useful when a single bad bridge edge can damage the interpretability of the whole reasoning path.

### How the implementation solves it

The code does not directly run a specialized minimax shortest-path routine. Instead, it uses threshold search over the observed edge costs.

Let the distinct edge costs be:

$$
\tau_1 < \tau_2 < \dots < \tau_r
$$

For a threshold $\tau$, define the filtered graph:

$$
G_{\tau} = (V, E_{\tau}),
\qquad
E_{\tau} = \{e \in E : c(e \mid q) \le \tau\}
$$

Then:

- if source and target are connected in $G_{\tau}$, there exists a path whose worst edge cost is at most $\tau$
- if they are disconnected, no such path exists

The implementation binary-searches the sorted thresholds and asks whether a path exists in $G_{\tau}$.

### Algorithm steps

For each source-target pair:

1. Compute all distinct edge costs.
2. Sort them.
3. Binary-search a threshold $\tau$.
4. Keep only edges with cost $\le \tau$.
5. Test whether source and target are connected in the filtered graph.
6. If yes, try a smaller threshold.
7. If no, try a larger threshold.
8. Once the smallest feasible threshold is found, return a shortest path inside that filtered graph.

### Intuition

The minimax objective is stricter than weighted Dijkstra:

- Dijkstra accepts a path with low total cost
- minimax accepts a path only if its worst edge is as good as possible

This is often a better fit for bridge retrieval, because one semantically irrelevant edge can make the whole bridge less convincing.

---

## 5. `hi_minmax_budgeted`: Minimax With Explicit Budgets

### Objective

`hi_minmax_budgeted` keeps the same minimax objective, but adds explicit length budgets.

For a path $P$:

$$
\min_P \max_{e \in P} c(e \mid q)
\quad
\text{subject to}
\quad
|P| - 1 \le L
$$

where:

- $|P|-1$ is the number of edges in the path
- $L = \texttt{bridge\_max\_path\_edges}$

The defaults are:

- `bridge_max_path_edges = 120`
- `bridge_max_total_edges = 220`
- `bridge_budget_fallback = weighted`

### Per-segment budget

The same threshold-search idea is used, but now a candidate path is accepted only if:

$$
|P| - 1 \le L
$$

In the code this is implemented as `len(candidate) <= max_nodes`, where:

$$
\texttt{max\_nodes} = L + 1
$$

So a minimax path is feasible only if it is both:

1. under the current threshold
2. short enough

### Total bridge budget across segments

The bridge is often built by connecting several consecutive key-entity pairs:

$$
(v_1 \to v_2),\ (v_2 \to v_3),\ \dots,\ (v_{k-1} \to v_k)
$$

If the returned subpaths are $P_1, P_2, \dots, P_{k-1}$, the code also tracks the cumulative edge count:

$$
\sum_i (|P_i|-1) \le B_{\text{total}}
$$

where:

$$
B_{\text{total}} = \texttt{bridge\_max\_total\_edges}
$$

If adding the next segment would exceed this total budget, the algorithm stops and records a `budget_stop`.

### Fallback behavior

If no minimax path satisfies the per-segment budget, the implementation falls back as follows:

1. try query-weighted Dijkstra
2. keep it only if it also satisfies the same path-length limit
3. otherwise fall back to unweighted shortest path

So the fallback policy is:

$$
\text{budgeted minimax}
\rightarrow
\text{weighted Dijkstra}
\rightarrow
\text{unweighted shortest path}
$$

depending on feasibility.

### Intuition

Pure minimax improves bridge quality, but it can also produce long paths. Long paths are risky because they:

- consume more bridge tokens
- increase latency
- add more opportunities for irrelevant edges

The budgeted variant preserves the minimax idea while making it compatible with fixed retrieval and prompt budgets.

---

## 6. Practical Summary

The implemented retrieval logic can be summarized as follows.

### Source evidence

- retrieved source chunks are first selected under a chunk-level token budget
- optionally, each retained chunk is replaced by a bounded snippet
- `query_overlap` chooses the best local window inside the chunk instead of always taking the prefix

### Bridge scoring

- every graph edge receives a query-conditioned cost
- the cost is dominated by semantic mismatch
- hubness and weak relation strength are added as smaller penalties

### Bridge search variants

- `hi_weighted`: minimize the sum of edge costs
- `hi_minmax`: minimize the worst edge cost
- `hi_minmax_budgeted`: minimize the worst edge cost subject to explicit path-edge budgets, with fallbacks

### High-level intuition

These changes all move retrieval in the same direction:

- less topology-only reasoning
- more query-conditioned bridge selection
- better control of prompt budget
- stronger evidence density in the final context
