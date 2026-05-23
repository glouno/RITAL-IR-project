# MCTS Implementation In This Repo

Date: 2026-05-22

This document explains the exact Monte Carlo Tree Search bridge retrieval implementation currently added to the active HiRAG runtime in this repository.

It describes what the code does now, not the broader design space from [MCTS_BRIDGE_RETRIEVAL_IDEAS.md](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/docs/MCTS_BRIDGE_RETRIEVAL_IDEAS.md).

Active code path:

- [src/hirag/base.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/base.py)
- [src/hirag/hirag.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/hirag.py)
- [src/hirag/_op.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/_op.py)
- [src/hirag/mcts.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/mcts.py)

---

## Executive Summary

Implemented MCTS variant is:

1. query-weighted
2. candidate-subgraph restricted
3. UCT-based
4. progressively widened
5. heuristic-rollout driven
6. token-budget aware
7. path-length aware
8. protected by weighted Dijkstra fallback

Mode exposed to users:

- `hi_mcts`

Bridge strategy exposed internally:

- `mcts`

In short:

1. repo first computes query-aware edge costs exactly like weighted bridge retrieval
2. MCTS then searches only inside source/target candidate subgraph
3. path reward favors low-cost query-relevant edges, reaching target, shorter paths, lower token cost
4. if MCTS does not beat weighted fallback under this reward, repo returns weighted path instead

---

## 1. Runtime Integration

### Query mode

The new public retrieval mode is:

$$
\texttt{mode} = \texttt{hi\_mcts}
$$

This is defined in [src/hirag/base.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/base.py) and resolved in [src/hirag/hirag.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/hirag.py) as:

$$
\texttt{hi\_mcts} \Rightarrow \texttt{bridge\_strategy} = \texttt{mcts}
$$

So MCTS is not separate retrieval pipeline. It is bridge-search policy inside normal hierarchical retrieval path.

### Shared hierarchical context builder

`hi_mcts` goes through same shared helper as other hierarchical retrieval modes:

$$
\_build\_hierarchical\_query\_context\_common(q, \dots)
$$

That means final context still has same high-level pieces:

- local entities
- community reports
- bridge path edges
- source chunks

Only bridge-path construction changes.

---

## 2. Step-By-Step Algorithm

For required entity pair `(s, t)`:

### Step 1. Build query-weighted edge costs

Before MCTS starts, repo computes edge costs with same query-aware scoring used by `hi_weighted`.

For edge \(e = (u,v)\):

$$
\text{semantic\_score}(e \mid q) = \cos(\mathbf{q}, \mathbf{e})
$$

$$
\text{semantic\_cost}(e \mid q) = 1 - \text{semantic\_score}(e \mid q)
$$

$$
\text{hub\_penalty}(e) = \frac{\deg(u) + \deg(v)}{Z_{\text{hub}}}
$$

where:

$$
Z_{\text{hub}} = \max_{(x,y)\in E_g} \left(\deg(x)+\deg(y)\right)
$$

Edge weight penalty:

$$
\text{inverse\_weight\_penalty}(e)=\frac{1}{1+w_e}
$$

Final edge cost:

$$
c(e \mid q)=
\alpha \cdot \text{semantic\_cost}(e \mid q)
 + \beta \cdot \text{hub\_penalty}(e)
 + \gamma \cdot \text{inverse\_weight\_penalty}(e)
$$

Current defaults from [src/hirag/base.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/base.py):

- \(\alpha = 1.0\)
- \(\beta = 0.15\)
- \(\gamma = 0.05\)

MCTS uses edge utility instead of edge cost in some later steps:

$$
u(e \mid q)=\max(0, 1-c(e \mid q))
$$

### Step 2. Build candidate subgraph

MCTS does not search full graph.

It first collects nodes within bounded hop radius around source and target:

$$
V_{\text{cand}} =
N_r(s) \cup N_r(t) \cup \{s,t\}
$$

Initial default:

$$
r_0 = 2
$$

However, the runtime now uses an adaptive radius schedule.

It starts from:

$$
r = r_0
$$

and if the source/target candidate graph is still disconnected, it retries with larger radii:

$$
r \leftarrow r + 1
$$


until either:

- source and target become connected inside the candidate graph
- or a configured maximum radius is reached

Current default maximum radius:

$$
r_{\max} = 4
$$

So candidate extraction is:

$$
V_{\text{cand}}(r)=
N_r(s) \cup N_r(t) \cup \{s,t\},
\qquad
r \in \{r_0, r_0+1, \dots, r_{\max}\}
$$

This keeps the default search local, but allows MCTS to recover when the initial two-hop neighborhoods are too small to contain any bridge.

After repo picks a connected candidate radius if possible, it prunes outgoing neighbors per node by keeping top query-weighted neighbors:

$$
\mathcal{N}_{\text{keep}}(v)=
\operatorname{TopB}_{x \in \mathcal{N}(v)}
\left(-c(v,x \mid q), -\deg(x)\right)
$$

Current default:

$$
B = 12
$$

Repo also forcibly preserves source and target neighbors if present. If pruned graph disconnects source/target, code falls back to unpruned candidate graph.

If source and target are still disconnected after trying all radii up to \(r_{\max}\), repo returns `mcts_no_path`.

### Step 3. Initialize root state

Root search state is:

$$
s_0 = (v_0, P_0, V_0, \tau_0) = (s, [s], \{s\}, 0)
$$

where:

- \(v_t\): current node
- \(P_t\): path so far
- \(V_t\): visited nodes
- \(\tau_t\): accumulated token estimate

Token estimate is cheap approximation from edge description text:

$$
\operatorname{tok}(e) \approx \max(1, \text{wordcount}(\text{description}(e)))
$$

### Step 4. Order available actions

For current node \(v\), neighbor ordering is heuristic:

$$
\text{score}_{\text{nbr}}(x)=
-u(v,x \mid q)
-0.4\cdot \frac{1}{1+d(x,t)}
+0.2\cdot \mathbf{1}[x \in V]
$$

where \(d(x,t)\) is shortest-path distance to target inside candidate graph.

Neighbors are sorted ascending by this score, then by:

- neighbor degree
- edge cost
- node id

So earlier expansion favors:

- higher edge utility
- closer-to-target nodes
- fewer revisits

### Step 5. Apply progressive widening

MCTS does not expand all neighbors immediately.

Allowed number of expanded children at tree node \(s\):

$$
K(s)=\max\left(1,\left\lfloor c_{pw}\cdot N(s)^{\alpha_{pw}}\right\rfloor\right)
$$

Current defaults:

- \(c_{pw}=2.0\)
- \(\alpha_{pw}=0.5\)

So repo expands only first unexpanded action while:

$$
|\text{children}(s)| < \min(|\text{ordered\_actions}(s)|, K(s))
$$

### Step 6. Selection with UCT

If node cannot expand more children, code selects child by UCT:

$$
\operatorname{UCT}(s,a)=
\bar{Q}(s,a)
 + c \sqrt{\frac{\ln N(s)}{N(s,a)}}
$$

where:

- \(\bar{Q}(s,a)=\frac{W(s,a)}{N(s,a)}\)
- \(W(s,a)\): cumulative reward
- \(N(s)\): parent visits
- \(N(s,a)\): child visits

Current exploration constant:

$$
c = 1.2
$$

Unvisited child gets:

$$
\operatorname{UCT}(s,a)=+\infty
$$

### Step 7. Terminal checks

Search node is terminal if any condition holds:

1. target reached
2. path depth limit reached
3. token budget reached
4. no ordered actions left

Formally:

$$
\text{terminal}(s)=
\begin{cases}
\text{target\_reached} & v_t = t \\
\text{depth\_limit} & |P_t|-1 \ge D_{\max} \\
\text{token\_budget} & \tau_t \ge T_{\max} \\
\text{dead\_end} & |\mathcal{A}(s)| = 0
\end{cases}
$$

Current defaults:

- \(D_{\max} = 120\)
- \(T_{\max} = 12500\)

In practice `T_max` is bound to `QueryParam.max_token_for_bridge_knowledge`.

### Step 8. Heuristic rollout

Rollout from expanded node is not random.

At rollout step, repo filters out visited nodes:

$$
\mathcal{N}_{\text{rollout}}(v)=\{x \in \mathcal{N}(v): x \notin V\}
$$

Then keeps top:

$$
k = 3
$$

neighbors under same heuristic ordering as expansion.

Action choice is epsilon-greedy:

$$
a_t=
\begin{cases}
\text{best candidate} & \text{with prob. } 1-\epsilon \\
\text{random top-k candidate} & \text{with prob. } \epsilon
\end{cases}
$$

Current default:

$$
\epsilon = 0.1
$$

Rollout depth cap:

$$
R_{\max}=4
$$

### Step 9. Score rollout path

For path \(P=(v_0,\dots,v_L)\), edge-utility average is:

$$
\operatorname{mean\_edge\_utility}(P)=
\frac{1}{L}\sum_{i=0}^{L-1} u(v_i,v_{i+1}\mid q)
$$

Estimated token ratio:

$$
\operatorname{token\_ratio}(P)=
\min\left(1, \frac{\sum_{e\in P}\operatorname{tok}(e)}{T_{\max}}\right)
$$

Final path reward in current code:

$$
R(P)=
\operatorname{mean\_edge\_utility}(P)
+ \lambda_{\text{target}}\cdot \mathbf{1}[v_L=t]
- \lambda_{\text{len}}\cdot L
- \lambda_{\text{tok}}\cdot \operatorname{token\_ratio}(P)
$$

Current defaults:

- \(\lambda_{\text{target}}=1.0\)
- \(\lambda_{\text{len}}=0.02\)
- \(\lambda_{\text{tok}}=0.15\)

Important: current implementation does **not** yet include explicit facet coverage, novelty, learned value model, or hub term directly inside final reward. Hub information only enters through earlier edge-cost construction.

### Step 10. Backpropagate reward

For every node on selected tree path:

$$
N(s) \leftarrow N(s) + 1
$$

$$
W(s) \leftarrow W(s) + R(P)
$$

So average value later used by UCT is:

$$
\bar{Q}(s)=\frac{W(s)}{N(s)}
$$

### Step 11. Keep best path

Repo tracks:

- `best_path`
- `best_reward`
- `successful_rollouts`

Rollout counted successful if:

$$
v_L = t
$$

### Step 12. Compare against weighted fallback

Before loop, repo computes weighted Dijkstra path inside candidate graph:

$$
P_{\text{wd}}=
\operatorname{Dijkstra}(G_{\text{cand}}, s, t; c(e \mid q))
$$

Its reward is:

$$
R_{\text{wd}} = R(P_{\text{wd}})
$$

After MCTS:

If best MCTS path reaches target and:

$$
R(P_{\text{mcts}}) \ge R_{\text{wd}}
$$

repo returns:

$$
P_{\text{mcts}}
$$

Else repo returns:

$$
P_{\text{wd}}
$$

This is why implementation is safe experimental policy, not hard replacement.

### Step 13. Early stop rule

Loop can stop early when rollout reaches target and is already at least as good as weighted fallback:

$$
v_L = t
\quad \land \quad
R(P) \ge R_{\text{wd}}
$$

Then extra exploration is skipped.

---

## 3. Exact Defaults

Current MCTS defaults from [src/hirag/base.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/base.py):

- `mcts_max_iterations = 96`
- `mcts_exploration_constant = 1.2`
- `mcts_candidate_hops = 2`
- `mcts_candidate_max_hops = 4`
- `mcts_candidate_top_neighbors = 12`
- `mcts_progressive_widening_coefficient = 2.0`
- `mcts_progressive_widening_exponent = 0.5`
- `mcts_rollout_top_k = 3`
- `mcts_rollout_depth = 4`
- `mcts_rollout_epsilon = 0.1`
- `mcts_token_penalty = 0.15`
- `mcts_target_reward = 1.0`

Shared bridge defaults also used by MCTS:

- `bridge_max_path_edges = 120`
- `bridge_max_total_edges = 220`
- `bridge_length_penalty = 0.02`
- `max_token_for_bridge_knowledge = 12500`

---

## 4. Mapping Between Code And Algorithm

### Config

- `QueryParam` MCTS fields:
  - [src/hirag/base.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/base.py)
- frozen runtime config object:
  - `MCTSBridgeConfig` in [src/hirag/mcts.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/mcts.py)
- bridge-param -> MCTS-config adapter:
  - `_mcts_config_from_query(...)` in [src/hirag/_op.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/_op.py)

### Search helpers

- candidate subgraph:
  - `_build_candidate_subgraph(...)`
- neighbor ordering:
  - `_neighbor_order(...)`
- widening:
  - `_allowed_expansions(...)`
- terminal condition:
  - `_is_terminal(...)`
- expansion:
  - `_expand(...)`
- UCT selection:
  - `_select_child(...)`
- rollout:
  - `_rollout(...)`
- reward:
  - `_path_reward(...)`
- top-level path search:
  - `find_mcts_bridge_path(...)`

### Retrieval integration

- mode resolution:
  - `HiRAG._resolve_query_param(...)` in [src/hirag/hirag.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/hirag.py)
- bridge-path dispatch:
  - `_path_between(...)` in [src/hirag/_op.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/src/hirag/_op.py)
- bridge-path stitching over required key entities:
  - `_find_path_with_required_nodes(...)`
- common hierarchical context assembly:
  - `_build_hierarchical_query_context_common(...)`

---

## 5. Multi-Segment Bridge Construction

Final bridge is not one single source-target search over all key entities.

If required key entities are:

$$
[k_1, k_2, \dots, k_m]
$$

repo solves bridge segments:

$$
(k_1 \to k_2), (k_2 \to k_3), \dots, (k_{m-1} \to k_m)
$$

For each segment, `_path_between(...)` may choose:

- `mcts`
- `mcts_weighted_fallback`
- `mcts_no_path`

Then `_find_path_with_required_nodes(...)` concatenates subpaths:

$$
P_{\text{final}} = P_1 \oplus P_2 \oplus \dots \oplus P_{m-1}
$$

subject to total bridge-edge budget:

$$
\sum_i (|P_i|-1) \le E_{\max}
$$

with current default:

$$
E_{\max}=220
$$

If this total budget would be exceeded, retrieval records `budget_stop`.

---

## 6. Debug Instrumentation

MCTS writes per-segment metadata into `QueryParam.debug_info`.

Important fields:

- `bridge_strategy`
- `bridge_path_decisions`
- `bridge_path_edges`
- `bridge_edges`
- `mean_edge_score`
- `max_edge_cost`

For MCTS segments, each decision may include:

- `decision`
- `iterations`
- `successful_rollouts`
- `best_reward`
- `fallback_reward`
- `candidate_hops`
- `candidate_radius_expanded`
- `candidate_max_hops`
- `candidate_nodes`
- `candidate_edges`

Generation-side eval compaction in [eval/answer_generation_benchmark.py](/Users/bsh2022/Study/master_ue/rital/RITAL-IR-project/eval/answer_generation_benchmark.py) further summarizes:

- `mcts_segments`
- `mcts_iterations`
- `mcts_successful_rollouts`
- `mcts_candidate_nodes_max`
- `mcts_candidate_edges_max`
- `mcts_used_weighted_fallback`

---

## 7. What This Implementation Does Not Yet Do

Current code intentionally does **not** implement:

- learned policy prior
- learned value network
- PUCT
- explicit `STOP` action
- explicit facet coverage reward
- explicit novelty reward
- explicit cycle-count penalty beyond visited-node filtering
- full-graph MCTS
- parallel tree search
- dedicated `hi_rerank_mcts`

So current implementation should be understood as:

> query-weighted candidate-subgraph MCTS with UCT, heuristic rollout, reward shaping, and weighted fallback

not:

> full final-form MCTS retrieval system

---

## 8. Minimal Pseudocode

```text
input: query q, source s, target t

1. build query-weighted edge costs c(e | q)
2. build candidate subgraph G_cand around s and t
3. compute weighted fallback path P_wd on G_cand
4. initialize root state at s
5. repeat up to max_iterations:
   a. descend tree with:
      - progressive widening if child budget not filled
      - else UCT child selection
   b. expand one new child if possible
   c. rollout with heuristic top-k epsilon-greedy policy
   d. score rollout path with:
      mean_edge_utility + target_bonus - length_penalty - token_penalty
   e. backpropagate reward
   f. keep best rollout path
   g. early stop if reached target and reward >= weighted fallback reward
6. if best MCTS path reaches target and beats weighted fallback:
   return best MCTS path
7. else:
   return weighted fallback path
```

---

## 9. Bottom Line

Exact implementation in this repo is:

$$
\text{MCTS over } G_{\text{cand}}(s,t,q)
$$

with:

- query-aware edge costs from existing weighted bridge scoring
- UCT for child selection
- progressive widening for branching control
- heuristic rollout toward target
- reward shaped by edge utility, target hit, length, and token budget
- weighted Dijkstra as guardrail fallback

That is why this implementation fits current repo well:

1. it reuses existing weighted bridge machinery
2. it stays inside current hierarchical retrieval path
3. it remains benchmarkable against `hi_weighted` and `hi_minmax_budgeted`
4. it fails safe by returning weighted path when MCTS is not better
