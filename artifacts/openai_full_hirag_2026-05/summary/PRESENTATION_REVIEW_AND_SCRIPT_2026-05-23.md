# HiRAG Presentation Review And 15-Minute Story

Date: 2026-05-23

Reviewed draft: `Présentation HiRAG_V1.pdf`.

## Executive Assessment

The current deck has the right ingredients, but the story is not yet sharp
enough for a 15-minute presentation.

Main issue: the deck spends too much time on algorithmic implementation details
before establishing the experimental surprise. The strongest story is not only
"we implemented weighted Dijkstra, Minimax, and MCTS"; it is:

> We reproduced HiRAG with a stronger OpenAI-based pipeline, improved its graph
> traversal, and found that direct chunk retrieval (`naive`) still often wins
> because QA tasks reward exact source evidence more than abstract graph
> reasoning.

That is a good, defensible research story. It is more interesting than a simple
"our variants did/did not improve" narrative.

## Recommended Core Story

### 1. Problem

Standard RAG retrieves chunks, but it has weak structure:

- no explicit entities;
- no relation reasoning;
- weak global view over a corpus.

GraphRAG-style systems add structure, but flat graphs struggle with multi-scale
questions: some questions need local facts, others need global summaries, and
some need bridges between them.

HiRAG tries to solve this by combining:

- local entity context;
- global community reports;
- bridge paths between local/global knowledge.

### 2. Our Hypothesis

HiRAG's bridge mechanism is useful, but too static:

- local entity retrieval uses dense retrieval only;
- bridge path selection is not query-aware enough;
- final prompts can become dominated by entity/path abstractions;
- raw source evidence may be underrepresented.

So we implemented query-aware retrieval variants:

- weighted Dijkstra;
- Minimax;
- Minimax budgeted;
- reranking;
- MCTS;
- query-aware source snippets;
- full OpenAI graph reconstruction and evaluation.

### 3. Evaluation

We rebuilt Agriculture and Mix with an OpenAI-homogeneous pipeline:

- answer and judge model: `gpt-5.4-mini`;
- embeddings: `text-embedding-3-small`;
- datasets: Agriculture and Mix from UltraDomain Benchmark;
- evaluation subset: 30 questions per dataset;
- generated answers for 7 variants;
- LLM-as-judge with swapped answer order;
- human annotation on 60 sampled pairwise comparisons.

### 4. Main Result

Our graph variants did not clearly beat classic HiRAG, and `naive` often beat
all graph modes.

This is not just a failure; it reveals something important:

> On these QA-style questions, direct source evidence is often more valuable
> than graph abstraction.

HiRAG answers are often acceptable, but `naive` answers are often preferred
because they contain more exact, concrete source text.

### 5. Diagnosis

We found several reasons:

- `naive` sends mostly raw source chunks;
- graph modes spend many tokens on entities, paths, and community reports;
- community reports initially disappeared because `max_token_for_community_report=1000`;
- after fixing this with `community2200`, graph modes included communities but
  still did not consistently beat `naive`;
- graph context can explode or become abstract, while QA evaluation rewards
  exact evidence.

### 6. Conclusion

The next improvement should not only improve graph traversal. It should improve
evidence composition:

> Hybrid graph+source retrieval: use graph traversal to find relevant entities
> and paths, but reserve a guaranteed budget for raw source evidence.

This is the most defensible next step.

## Current Deck: What Works

- The opening diagrams comparing standard RAG, GraphRAG, and HiRAG are useful.
- The problem list is good: chunks, dense local retrieval, non-query-aware bridge.
- The algorithm slides for weighted Dijkstra, Minimax, and MCTS show real
  technical work.
- The evaluation dataset table is strong and should stay.
- The human annotation table is useful and slide-ready.

## Current Deck: Main Problems

### 1. Too Much Algorithm Detail For 15 Minutes

There are many slides on MCTS internals:

- UCT;
- expansion;
- rollout;
- backpropagation;
- subgraph construction.

This is valuable, but too much for the main talk. Keep only one high-level MCTS
slide, and move detailed equations to backup.

### 2. Evaluation Slides Are Repetitive

Several LLMaJ slides repeat similar Mix/Agriculture tables with slightly
different formatting.

Recommendation:

- keep one compact LLM judge slide;
- keep one human annotation slide;
- keep one diagnostic slide explaining why `naive` wins.

### 3. "LLMaJ strict" Is Currently Ambiguous

Some slides say "LLMaJ strict" but show 60 valid comparisons per variant.
That table is closer to raw swapped-order judge calls:

- 30 questions;
- 2 answer orders;
- 60 judge rows.

Strict swapped-order evaluation should usually mean:

- judge both orders;
- keep a query only if both orders pick the same underlying variant;
- otherwise mark it as `disagree`.

So either:

- rename those tables to "LLM pairwise judge with swapped order"; or
- show strict wins/losses/disagree explicitly.

For the human-vs-LLM slide, use the strict interpretation because it makes the
judge instability visible.

### 4. The Naive Result Needs To Become The Narrative Center

Right now `naive` appears in results tables, but the deck does not sufficiently
explain why it matters.

The surprising result should be a key slide:

> Naive RAG often beats graph modes because it gives the answer model more raw
> evidence, while graph modes spend context on structured abstractions.

This is the central finding.

### 5. Conclusion Is Empty

The deck currently ends with "Conclusion" but no real conclusion text. This
needs a strong final message:

- HiRAG is promising for structured/global reasoning.
- But graph retrieval is not automatically better than dense chunk retrieval.
- Context composition matters as much as graph traversal.
- Future work: hybrid graph+source retrieval, dynamic community selection, and
  better evidence-budget control.

## Recommended 15-Minute Slide Structure

Aim for 13-15 slides, not 33.

### Slide 1: Title

Time: 20s.

Say:

> We studied HiRAG, a hierarchical graph-based RAG method, reproduced it with an
> OpenAI pipeline, implemented query-aware graph traversal variants, and compared
> them against classic HiRAG and naive RAG.

### Slide 2: Why RAG Needs Structure

Time: 1 min.

Use the standard RAG vs GraphRAG vs HiRAG diagram.

Message:

- chunk retrieval is simple and strong;
- graph retrieval adds structure;
- HiRAG adds hierarchy and bridge reasoning.

### Slide 3: What HiRAG Does

Time: 1 min.

Show:

- local knowledge;
- global communities;
- bridge context;
- final LLM answer.

Say:

> HiRAG does not only retrieve chunks. It retrieves entities, community reports,
> bridge relations, and then also source text units associated with graph nodes.

### Slide 4: Our Diagnosis Of HiRAG

Time: 1 min.

Keep the three problems:

- full chunks can saturate context;
- local retrieval needs reranking;
- bridge path is not query-aware enough.

Add:

- graph context can become abstract and crowd out source evidence.

### Slide 5: Our Architecture

Time: 1 min.

One pipeline diagram:

```text
Documents -> chunks -> entity/relation extraction -> graph -> communities
          -> OpenAI embeddings/vector stores
          -> retrieval variants -> answers -> LLM judge + humans
```

Mention:

- Batch API;
- `gpt-5.4-mini`;
- `text-embedding-3-small`;
- Agriculture + Mix.

### Slide 6: Implemented Variants

Time: 1 min.

One table:

| Variant | Idea |
|---|---|
| HiRAG | original baseline |
| Weighted Dijkstra | prefer query-relevant edges |
| Minimax | avoid one bad bridge edge |
| Minimax budgeted | same, with path budget |
| Rerank weighted | rerank local entities |
| MCTS | explore bridge paths |
| Naive | direct chunk retrieval baseline |

Do not explain all formulas here.

### Slide 7: One Technical Detail: Query-Aware Bridge

Time: 1 min.

Show edge cost idea only:

- edge embedding;
- similarity to query;
- penalties for hubs/weak relations;
- choose better path.

Move MCTS UCT/rollout equations to backup.

### Slide 7b: Why MCTS Is Interesting

Time: 45s-1 min.

Keep one MCTS slide in the main deck, but frame it as exploration rather than
as a guaranteed winner.

Message:

- Weighted Dijkstra and Minimax choose one path by an explicit graph objective.
- MCTS treats bridge construction as a sequential decision problem.
- It explores candidate next hops, estimates long-term reward, and can trade
  off local edge quality, path length, and token budget.
- This is conceptually appealing for retrieval because retrieval is not only
  "find the nearest edge"; it is "assemble a useful reasoning route under a
  context budget".

Important result framing:

> MCTS did not clearly beat HiRAG in human preference, but it was robust: in the
> human subset, both HiRAG and MCTS answered the question on all `MCTS vs HiRAG`
> examples, and humans rated MCTS almost tied with HiRAG (`52.5%` half-credit
> win rate).

So MCTS is a promising mechanism, but our current context assembly does not yet
turn its path exploration into a clearly preferred answer.

### Slide 8: Evaluation Setup

Time: 1 min.

Use dataset table:

| Dataset | Docs | Questions available | Chunks | Nodes | Edges | Reports |
|---|---:|---:|---:|---:|---:|---:|
| Agriculture | 12 | 100 | 1756 | 23224 | 48428 | 4639 |
| Mix | 61 | 130 | 579 | 16356 | 25960 | 2695 |

Say:

> We evaluated on 30 questions per dataset, generated answers for 7 variants,
> and judged all pairwise comparisons against HiRAG with swapped order.

### Slide 9: LLM Judge Results

Time: 1.5 min.

Do not show huge raw tables. Show the punchline:

- `naive` wins strongly vs HiRAG;
- graph variants are mixed / close;
- MCTS and Minimax do not reliably dominate.

Use compact chart/table from Core7 or Community2200. Be clear whether it is raw
judge calls or strict swapped-order.

### Slide 10: Human Annotation

Time: 1.5 min.

Use the slide table:

| Variant | Human win rate | Span density | Human gain | LLM strict |
|---|---:|---:|---:|---:|
| Naive | 65.0% | 32.0% | +0.30 | 78.6% |
| Minmax budgeted | 45.0% | 32.1% | -0.10 | 61.5% |
| MCTS | 52.5% | 30.4% | +0.05 | 38.5% |

Say:

> Humans confirm the strong `naive` signal. For closer graph variants, human and
> LLM judgments diverge more.

For MCTS specifically:

> MCTS is not a failure case. Humans judged MCTS answers as answering the
> question very reliably, but often not more useful than classic HiRAG. The
> likely issue is that better path exploration does not automatically produce
> better final evidence in the prompt.

### Slide 11: Human vs LLM Judge

Time: 1 min.

Use `human_vs_llm_decision_breakdown.png`.

Explain:

- LLM agrees on strong cases;
- LLM has swapped-order disagreement on close cases;
- LLM-as-judge is useful but not an oracle.

### Slide 12: Why Does Naive Win?

Time: 1.5 min.

This is crucial.

Show:

- `naive` sends mostly raw source chunks;
- graph modes send entity descriptions + bridge paths + communities + fewer
  direct source snippets;
- many QA questions need exact textual evidence.

Suggested phrase:

> HiRAG often answers the question, but `naive` often gives the answer model the
> exact passage it needs. So humans prefer `naive` even when HiRAG is acceptable.

This also helps explain MCTS:

> MCTS may choose a better bridge path, but if the final prompt still contains
> similar entities/source snippets to HiRAG, or if the extra path evidence is too
> abstract, humans will not strongly prefer it.

### Slide 13: What We Fixed And What We Learned

Time: 1 min.

Mention:

- we found `max_token_for_community_report=1000` suppressed community reports;
- reran with `2200`;
- graph modes then included communities, but this alone did not solve the issue.

Lesson:

> More graph context is not automatically better. The context must be useful,
> concise, and evidence-rich.

### Slide 14: Limitations

Time: 1 min.

Say:

- only Agriculture + Mix;
- q30 subset;
- LLM judge has order sensitivity;
- human annotation is small;
- our baseline HiRAG is our fork baseline with query-aware snippets, not a
  perfectly untouched upstream implementation;
- graph construction depends on extraction quality.

### Slide 15: Conclusion And Next Steps

Time: 1 min.

Final message:

> The next improvement is not just smarter graph traversal. It is hybrid
> evidence composition: use the graph to decide where to look, but preserve a
> guaranteed budget for raw source passages.

Next steps:

- hybrid graph+source retrieval;
- dynamic community traversal;
- better source evidence budgeting;
- larger evaluation and human annotation;
- evaluate question types where graph reasoning should actually help.

## Slides To Cut Or Move To Backup

Move to backup:

- detailed UCT equation slide;
- MCTS expansion formula;
- rollout details;
- backpropagation equation;
- repeated LLMaJ raw tables;
- duplicated human annotation table.

Keep in main:

- one MCTS conceptual slide;
- one edge-cost slide;
- one evaluation setup slide;
- one LLM judge result slide;
- one human annotation slide;
- one diagnostic/naive slide.

## Specific Corrections To Current Slides

### Terminology

- Replace "LLMaJ strict: 60 valid comparisons" with either:
  - "LLM pairwise judge with swapped order: 60 judge calls"; or
  - "Strict swapped-order results: wins/losses/disagree over 30 questions".

### Language Cleanup

Fix typos:

- "Parmís" -> "Parmi";
- "meilleur pire arête" -> "meilleure pire arête";
- "noeud source et cible devient déconnectées" -> "les noeuds source et cible deviennent déconnectés";
- "on arête l’exploration" -> "on arrête l’exploration";
- avoid mixing "punctuation and \\n" with French phrasing.

### Result Framing

Avoid saying "Naive baseline" as if it is weak. In our experiments it is a
strong baseline. Call it:

> Naive dense chunk RAG baseline.

### Human Annotation

Add the explanation:

> `Answers the question` measures minimal adequacy, while preference measures
> comparative usefulness/completeness. This explains why HiRAG can score high on
> adequacy while `naive` is preferred head-to-head.

### MCTS Framing

Do not oversell MCTS as "best". Instead say:

- MCTS is the most exploratory and conceptually novel retrieval variant.
- It is competitive with classic HiRAG in human preference.
- It reaches 100% answer-question adequacy on the annotated `MCTS vs HiRAG`
  subset, same as HiRAG.
- It does not yet yield clear preference gains, probably because answer quality
  is bottlenecked by evidence composition, not only bridge path selection.

Good slide title:

> MCTS improves the search formulation, but context composition remains the
> bottleneck.

## Generation Prompt Fairness

For answer generation, the system/user prompt is the same across variants in
the Batch evaluation.

The code path is:

- `eval/export_openai_batch_requests.py` retrieves a context for each variant;
- `eval/eval_utils.py::build_answer_messages()` builds the same message format;
- the only variant-dependent input is the retrieved `context`;
- the model is the same: `gpt-5.4-mini`.

So the comparison is fair at generation time: differences in answers come from
differences in retrieved context, not from different generation instructions.

This is worth saying in the evaluation section:

> We fixed the answer model and prompt. Only the retrieval context changes
> across variants.

## MCTS Human Annotation Diagnosis

On the human annotation subset:

- `MCTS vs HiRAG`: 20 rows.
- Human preference: 7 MCTS wins, 6 HiRAG wins, 7 ties.
- Half-credit win rate: 52.5%.
- Answer-question score: MCTS 100%, HiRAG 100%.

Interpretation:

- MCTS is reliable: annotators did not mark it as failing to answer.
- MCTS is not clearly preferred: many answers are very similar to HiRAG or only
  differ in wording/organization.
- In several sampled rows, retrieved contexts were identical or nearly
  identical between HiRAG and MCTS, so the answer model had little reason to
  produce meaningfully different answers.
- When MCTS loses, it is often not because it is wrong, but because HiRAG is
  more concise, more direct, or includes the exact phrasing humans prefer.
- This supports the broader diagnosis: improving graph path search is useful,
  but insufficient unless the final evidence budget is also improved.

## Suggested 15-Minute Timing

| Section | Time |
|---|---:|
| Motivation + HiRAG concept | 3 min |
| Our diagnosis + variants | 3 min |
| Evaluation setup | 2 min |
| LLM judge + human annotation results | 3 min |
| Why naive wins / context diagnosis | 2 min |
| Limitations + next steps | 2 min |

## One-Sentence Thesis

> HiRAG adds useful structure to RAG, but our experiments show that graph
> traversal alone is not enough: for QA-style tasks, the decisive factor is often
> whether the final context preserves enough exact source evidence.
