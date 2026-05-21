# Monte Carlo Tree Search For HiRAG Bridge Retrieval

Date: 2026-05-21

This note explains how Monte Carlo Tree Search (MCTS) could fit the active retrieval path in this repo, why it might be better than the currently implemented bridge methods in some cases, what else should be added around it, and which design choices and parameters are most important if we want it to work in practice.

Active runtime path:

- [main.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/main.py)
- [src/hirag/hirag.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/hirag.py)
- [src/hirag/base.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/base.py)
- [src/hirag/\_op.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/_op.py)

Current bridge methods already implemented:

- `hi`: unweighted shortest path
- `hi_weighted`: query-weighted Dijkstra
- `hi_minmax`: minimize worst edge cost
- `hi_minmax_budgeted`: minmax with bridge budgets

## Short Answer

MCTS is worth considering here because the current bridge methods are still basically shortest-path methods with fixed objectives. They are good when the objective is simple:

- shortest path
- minimum total edge cost
- minimum worst edge cost

But bridge retrieval in HiRAG is not always that simple. What we actually want is closer to:

- reach the right target entities
- stay query-relevant along the way
- avoid generic hubs
- avoid wasting budget
- cover different parts of the question
- stop when the bridge is already good enough

That is a sequential decision problem, and MCTS is designed exactly for that kind of problem.

The main caution is also simple:

- MCTS is not obviously better by default
- it will need careful pruning, caching, rollout design, and budget control
- without that, it can be slower and noisier than the current weighted/minmax methods

So the right framing is:

> MCTS should be added as an experimental bridge-search policy on top of the current query-weighted edge scoring, not as an immediate replacement for weighted or minmax search.

---

## 1. What Problem MCTS Would Solve

### Intuition

The current bridge methods solve path problems. MCTS would solve a path-construction policy problem.

That difference matters.

Weighted Dijkstra asks:

> Among all complete paths, which one has the smallest total cost?

Minmax asks:

> Among all complete paths, which one has the best worst edge?

MCTS asks:

> If I grow a bridge step by step, which next move is most promising under a long-term reward?

That lets us optimize things that are awkward for shortest-path methods.

### Why the current methods are limited

The current implementations in [src/hirag/\_op.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/_op.py) are strong baselines, but they have structural limits:

1. Weighted shortest path assumes additivity.
   A path is scored as the sum of edge costs.

2. Minmax assumes worst-edge dominance.
   A path is scored by its worst edge.

3. Both mostly reason over complete paths between a source and target.
   They are not naturally designed for:
   - partial success
   - multi-part query coverage
   - novelty/diversity of evidence
   - adaptive stopping
   - explicit exploration vs exploitation

4. They do not naturally learn that one early edge is good because it opens many later query-relevant options.

That last point is especially important. In retrieval, a locally mediocre step can be globally excellent if it leads into the right region of the graph.

---

## 2. Why MCTS Could Be Better Than What Is Already Implemented

### Case A: The best bridge is not the shortest good path

Sometimes the best bridge is not:

- the shortest route
- the lowest-sum-cost route
- the lowest-worst-edge route

It may instead be the route that:

- reaches the right topic region
- collects complementary evidence
- avoids spending too much budget on redundant edges

MCTS can prefer that because reward can be path-level, not just edge-level.

### Case B: The query has multiple subgoals

Suppose the query implicitly asks:

- one causal link
- one domain constraint
- one concrete entity relation

Shortest-path methods do not explicitly reward covering multiple subgoals. MCTS can.

For example, if the query token set is split into semantic facets:

$$
F(q) = \{f_1, \dots, f_m\}
$$

then a path reward can include facet coverage:

$$
\text{facet\_coverage}(P, q) = \frac{|\{f_i : \exists e \in P \text{ matching } f_i\}|}{m}
$$

That is hard to encode cleanly in Dijkstra, but easy to add to an MCTS reward.

### Case C: We want adaptive stopping

Current methods find a path and then later truncate or budget the resulting bridge.

MCTS can stop when:

- reward has plateaued
- target reached with enough relevance
- token budget is nearly exhausted
- added edges become redundant

This is closer to the real retrieval problem:

> not "find a path at any cost"

but

> "build enough bridge evidence to help the final answer model"

### Case D: We want to optimize final bridge usefulness, not only path geometry

If we think the real objective is:

$$
\text{usefulness}(P \mid q) =
\text{query relevance}
+ \text{target connectivity}
+ \text{novel evidence}
- \text{hub noise}
- \text{token cost}
$$

then MCTS is a much more natural optimizer than shortest-path algorithms.

---

## 3. Why MCTS Might Still Fail

This should be explicit, because MCTS is tempting to over-sell.

MCTS may be worse if:

1. The graph is too large and branching is not aggressively controlled.
2. The rollout reward is noisy.
3. The value estimate is too weak.
4. The search budget is too small.
5. The query is simple enough that weighted Dijkstra already solves it.

In this repo, weighted and minmax search are already fairly strong. So MCTS should be treated as a specialized method for harder multi-hop bridge construction, not as the new default for every query.

---

## 4. What MCTS Should Optimize In This Repo

The most important design choice is the reward.

If the reward is bad, MCTS will just be expensive random wandering.

### Recommended path reward

For a path \(P = (v_0, v_1, \dots, v_t)\), define:

$$
R(P \mid q) =
\lambda_1 \cdot \text{target\_hit}(P)
+ \lambda_2 \cdot \text{edge\_relevance}(P, q)
+ \lambda_3 \cdot \text{facet\_coverage}(P, q)
+ \lambda_4 \cdot \text{novelty}(P)
- \lambda_5 \cdot \text{hub\_penalty}(P)
- \lambda_6 \cdot \text{length\_cost}(P)
- \lambda_7 \cdot \text{token\_cost}(P)
$$

where:

- `target_hit(P)` rewards reaching the required next key entity
- `edge_relevance(P, q)` is based on the existing query-weighted edge scoring
- `facet_coverage(P, q)` rewards covering different parts of the query
- `novelty(P)` rewards non-redundant bridge content
- `hub_penalty(P)` penalizes generic graph hubs
- `length_cost(P)` penalizes long paths
- `token_cost(P)` penalizes large final bridge context

### Recommended component definitions

Edge relevance:

$$
\text{edge\_relevance}(P, q)
= \frac{1}{|P|-1} \sum_{e \in P} \cos(\mathbf{q}, \mathbf{e})
$$

Hub penalty:

$$
\text{hub\_penalty}(P)
= \frac{1}{|P|-1} \sum_{(u,v)\in P}
\frac{\deg(u)+\deg(v)}{Z_{\text{hub}}}
$$

Length cost:

$$
\text{length\_cost}(P) = \max(0, |P|-1)
$$

Token cost:

$$
\text{token\_cost}(P) \approx
\sum_{e \in P} \operatorname{tok}(e)
$$

Novelty:

$$
\text{novelty}(P)
= \frac{|\text{unique informative tokens in bridge}(P)|}{|\text{all informative bridge tokens}(P)|}
$$

That novelty term is important because one failure mode of graph retrieval is:

- semantically similar repeated edges
- repetitive entity descriptions
- bridge inflation without added evidence

---

## 5. Best Way To Integrate MCTS Here

Do not run MCTS on the full graph.

That is the biggest design decision.

### Recommended pipeline

1. Run the current local retrieval as usual.
2. Select communities and key entities as usual.
3. Build a candidate subgraph around those entities.
4. Run MCTS only inside that candidate subgraph.
5. Fall back to weighted or budgeted-minmax if MCTS produces no good bridge.

### Candidate subgraph construction

The candidate subgraph should be the union of:

- top local entities
- entities appearing in selected communities
- neighbors within 1 or 2 hops of key entities
- maybe top edge-scored neighbors under the existing query-weighted cost

For example:

$$
G_{cand}(q) = G[N_r(K(q))]
$$

where:

- `K(q)` = key entities
- `N_r` = radius-`r` neighborhood around them

A good first version is:

- radius `r = 2`
- then keep at most `B` outgoing neighbors per node by query-weighted edge score

This keeps MCTS from exploding.

---

## 6. State, Action, Transition, Terminal Condition

### State

A search state should contain:

- current node
- target node
- path so far
- visited-node set
- visited-edge set
- remaining edge budget
- accumulated bridge token estimate
- query facet coverage state

In notation:

$$
s_t = (v_t, v_{\text{target}}, P_t, V_t, E_t, b_t, \tau_t, C_t)
$$

### Action

An action is:

- move from current node to one neighboring node
- or optionally `STOP`

So:

$$
a_t \in \mathcal{N}(v_t) \cup \{\text{STOP}\}
$$

### Transition

Applying action extends the path:

$$
P_{t+1} = P_t \oplus a_t
$$

and updates:

- remaining budget
- token estimate
- visited sets
- facet coverage

### Terminal conditions

A trajectory should end if any of the following is true:

1. Target entity reached.
2. Edge-depth limit reached.
3. Total bridge token budget exceeded.
4. Repeated-node or cycle threshold exceeded.
5. `STOP` selected.

---

## 7. Selection Rule

Use standard UCT first.

For child \(i\) of node \(s\):

$$
\text{UCT}(s, i) =
\bar{Q}(s, i)
+ c \sqrt{\frac{\ln N(s)}{N(s, i)}}
$$

where:

- `\bar{Q}(s, i)` = average reward of child `i`
- `N(s)` = visits to parent
- `N(s, i)` = visits to child
- `c` = exploration constant

### Recommended initial setting

- `c = 1.2` or `1.4`

This is a good first default in this repo because we already have heuristic edge scores. That means we do not need extremely aggressive exploration.

### Better option after v1

Bias the prior using the existing query-weighted edge score:

$$
\text{prior}(a \mid s) \propto \exp(-\eta \cdot c(e_a \mid q))
$$

Then use a PUCT-style rule:

$$
\text{PUCT}(s,i) =
\bar{Q}(s,i)
+ c_{puct} \cdot P(s,i)\frac{\sqrt{N(s)}}{1+N(s,i)}
$$

That would make MCTS much more sample-efficient than pure UCT.

---

## 8. Expansion Policy

Do not expand all neighbors.

Use progressive widening.

### Why

Graph nodes can have huge degree. If every visit expands all neighbors, MCTS becomes unusable.

### Progressive widening rule

Allow at most:

$$
K(s) = c_{pw} \cdot N(s)^{\alpha}
$$

expanded children at state `s`.

Recommended starting values:

- `c_pw = 2`
- `\alpha = 0.5`

This means early visits only consider a few strong neighbors, and more neighbors are unlocked gradually.

### Which neighbors to expand first

Order candidate neighbors by:

1. existing query-weighted edge cost
2. target-entity similarity
3. low hub penalty

That lets MCTS spend most of its budget where it matters.

---

## 9. Rollout Policy

Pure random rollout is a bad fit here.

The graph is too large and semantic rewards are too sparse.

### Recommended rollout policy

Use a cheap heuristic rollout:

1. From current node, score candidate neighbors by a local heuristic.
2. Sample from the top `k` neighbors.
3. Continue until:
   - target reached
   - depth exhausted
   - token budget exhausted

### Heuristic rollout score

For candidate next edge `e=(u,v)`:

$$
\text{rollout\_score}(e \mid q) =
\rho_1 \cdot \cos(\mathbf{q}, \mathbf{e})
+ \rho_2 \cdot \cos(\mathbf{t}, \mathbf{v})
- \rho_3 \cdot \text{hub}(v)
- \rho_4 \cdot \text{repeat}(v)
$$

where:

- `\mathbf{t}` is an embedding of the target entity text
- `repeat(v)` penalizes revisits or near-duplicate bridge content

### Practical recommendation

For v1:

- top rollout candidates: `k = 3`
- rollout depth after tree expansion: `3` to `5`
- epsilon-greedy sampling with `\epsilon = 0.1`

That keeps rollouts cheap but not fully deterministic.

---

## 10. Backpropagation

Backpropagate:

- final path reward
- optional intermediate shaping reward

The simplest version is:

$$
Q(s,a) \leftarrow Q(s,a) + \frac{R - Q(s,a)}{N(s,a)}
$$

where `R` is the terminal rollout reward.

### Reward shaping

Reward shaping will help a lot in this repo because full target hits may be sparse.

Good intermediate signals:

- improved target similarity
- added query facet coverage
- reduced remaining distance to target in the candidate subgraph

For example:

$$
r_t =
\mu_1 \Delta \text{target\_sim}
+ \mu_2 \Delta \text{facet\_coverage}
- \mu_3 \Delta \text{token\_cost}
- \mu_4 \Delta \text{hub\_cost}
$$

and:

$$
R = \sum_{t=1}^{T} \gamma^{t-1} r_t + r_{\text{terminal}}
$$

with discount `\gamma`.

Recommended:

- `\gamma = 0.95`

---

## 11. Concrete Parameter Choices I Would Start With

If we implement `hi_mcts` in this repo, I would start with these defaults.

### Search budget

- `mcts_iterations = 256`
- `mcts_max_depth_edges = 6`
- `mcts_rollout_depth = 4`
- `mcts_time_budget_ms = 750` per key-entity pair

Why:

- enough to beat trivial search
- still bounded enough for batch evaluation

### Tree policy

- `mcts_selection = uct`
- `mcts_exploration_c = 1.4`
- later try `puct`

### Progressive widening

- `mcts_progressive_widening = true`
- `mcts_pw_c = 2.0`
- `mcts_pw_alpha = 0.5`
- `mcts_expand_top_neighbors = 12`

### Candidate subgraph

- `mcts_candidate_hops = 2`
- `mcts_candidate_neighbors_per_node = 16`
- `mcts_candidate_max_nodes = 250`

### Reward weights

A good first reward:

$$
R =
2.5 \cdot \text{target\_hit}
+ 1.0 \cdot \text{mean\_edge\_relevance}
+ 0.6 \cdot \text{facet\_coverage}
+ 0.4 \cdot \text{novelty}
- 0.5 \cdot \text{hub\_penalty}
- 0.25 \cdot \text{path\_length}
- 0.15 \cdot \text{token\_cost\_norm}
$$

### Rollout

- `mcts_rollout_top_k = 3`
- `mcts_rollout_epsilon = 0.1`
- `mcts_allow_stop = true`
- `mcts_stop_margin = 0.02`

### Safety

- `mcts_max_revisits = 1`
- `mcts_cycle_penalty = 0.75`
- `mcts_fallback = weighted`

---

## 12. What Else We Should Add Around MCTS

MCTS alone is not enough. If we add it, we should also add the surrounding system that makes it stable and measurable.

### 12.1 Candidate-subgraph pruning

This is mandatory.

Without it, MCTS will spend most of its effort exploring irrelevant neighborhoods. Reuse the current query-weighted edge scoring and local/key-entity selection to define a small candidate graph first.

### 12.2 Query facet extraction

MCTS becomes much more valuable if it can optimize coverage, not only relevance.

We should add a cheap query decomposition step, for example:

- noun phrases
- entity spans
- relation phrases
- hand-built token groups after stopword removal

Even a lightweight lexical facet extractor is enough for v1.

### 12.3 Path novelty / redundancy control

Current bridge context can already grow long. MCTS would benefit from explicit redundancy control.

Add:

- repeated relation-description penalties
- repeated entity-type penalties
- near-duplicate token penalties

### 12.4 Bridge compression after search

Even a good MCTS path can be too long for the answer model.

So after search we should still:

1. rank bridge edges by contribution
2. compress or trim the bridge
3. keep path connectivity evidence explicit

### 12.5 Better debug instrumentation

The repo already stores useful `debug_info` for retrieval variants. MCTS will need more:

- iterations used
- number of expanded states
- rollout count
- best path reward
- mean reward of selected branch
- fallback reason
- search termination reason

### 12.6 Offline cached node/edge features

To make MCTS affordable, cache:

- query embedding
- target embedding
- edge embedding
- node text embedding
- node degree / hub penalty
- token counts for edge descriptions

The weighted bridge cache already helps with part of this.

### 12.7 A query router

MCTS should not run on every query.

A cheap classifier could route:

- simple factual queries -> `hi_weighted`
- multi-hop explanatory queries -> `hi_mcts`
- entity-heavy disambiguation queries -> `hi_rerank_weighted`

That is probably better than trying to make one retrieval mode dominate all query types.

---

## 13. Suggested New Query Modes

If we add this to the active runtime, I would keep it explicit and experimental.

Suggested modes:

- `hi_mcts`
- `hi_rerank_mcts`
- maybe later `hi_mcts_budgeted`

Recommended mapping:

- `hi_mcts`: dense local retrieval + MCTS bridge
- `hi_rerank_mcts`: late-interaction local reranking + MCTS bridge

I would not add more than these two initially.

---

## 14. Evaluation Plan

The existing repo already has a good evaluation scaffold. MCTS should plug into that, not invent a new loop.

### Context-level metrics

Reuse and extend:

- latency
- context tokens
- bridge path edges
- bridge edge count
- bridge vs local recall
- bridge vs global recall
- bridge vs query overlap
- mean edge relevance
- max edge cost

Add MCTS-specific metrics:

- search iterations used
- candidate graph size
- rollout success rate
- target-hit rate
- early-stop rate
- fallback rate

### Answer-level metrics

Reuse:

- `eval/answer_generation_benchmark.py`
- `eval/pairwise_answer_judge.py`
- `eval/build_human_annotation_packet.py`

This is important because MCTS might improve bridge "reasoning quality" in ways that token-overlap proxies only partially capture.

### Comparisons that matter

The most important baselines are:

- `hi`
- `hi_weighted`
- `hi_minmax_budgeted`
- `hi_rerank_weighted`

Those are stronger comparisons than old unweighted-only baselines.

---

## 15. My Recommendation

If the goal is a practical and believable next experiment in this repo, the right choice is:

1. Keep the existing query-weighted edge scoring.
2. Build a candidate subgraph first.
3. Add `hi_mcts` as a budgeted bridge-search policy.
4. Use heuristic rollouts, not random rollouts.
5. Use progressive widening.
6. Add strong fallbacks to `hi_weighted` or `hi_minmax_budgeted`.

In other words:

> MCTS should be a search policy layered on top of the current weighted bridge machinery, not a totally separate retrieval system.

That gives it the best chance to outperform the current methods on genuinely difficult multi-hop queries while remaining testable and cheap enough to benchmark.

---

## 16. Recommended First Implementation Scope

To keep the project realistic, v1 should be narrow.

### Implement

- one new bridge strategy: `mcts`
- candidate subgraph pruning
- UCT search
- heuristic rollout
- path-length and token-budget limits
- fallback to weighted path
- debug instrumentation

### Do not implement yet

- learned policy network
- learned value network
- end-to-end RL training
- full-graph MCTS
- multi-agent or parallel tree search

That narrower version is much more likely to produce interpretable experimental results.

---

## 17. Bottom Line

MCTS would be interesting here because it can optimize what the current bridge methods only approximate:

- long-term path usefulness
- partial query coverage
- adaptive stopping
- non-additive bridge reward

It could be better than weighted/minmax search when the bridge problem is genuinely sequential and multi-objective.

But for this repo, it will only work well if we also add:

- candidate-subgraph pruning
- heuristic rollouts
- progressive widening
- strong reward shaping
- budget guards
- fallback paths
- evaluation against the current strong baselines

If we do implement it, the best initial story is not:

> MCTS replaces weighted/minmax retrieval.

It is:

> MCTS is a higher-capacity bridge search policy for the hard queries where shortest-path-style objectives are too rigid.
