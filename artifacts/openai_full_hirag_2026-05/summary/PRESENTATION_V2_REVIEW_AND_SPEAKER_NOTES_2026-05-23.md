# HiRAG Presentation V2 Review And Speaker Notes

Date: 2026-05-23

Reviewed historical draft: `Présentation HiRAG_V2.pdf` (available in Git history).
Current presentation: [Présentation HiRAG_final.pdf](../../../Présentation%20HiRAG_final.pdf).

## Overall Verdict

V2 is much better than V1 in three ways:

- it includes the stronger evaluation story;
- it adds human annotation results;
- it starts to show the surprising `naive` result.

But it still feels too much like an algorithm walkthrough. For a 15-minute
presentation, the deck should be sharper:

> We studied HiRAG, improved its graph traversal, and discovered that retrieval
> quality is not only about finding better graph paths. The final context must
> preserve enough direct source evidence, otherwise naive chunk retrieval can
> win.

That should be the central story.

## Main Recommendations

### Keep

- The RAG / GraphRAG / HiRAG conceptual diagrams.
- The three identified HiRAG problems.
- The edge-cost slide.
- One high-level MCTS slide.
- The dataset/statistics slide.
- One LLM-as-judge result slide.
- The human annotation table.
- The human-vs-LLM comparison.

### Cut Or Move To Backup

Move detailed algorithm slides to backup:

- detailed MCTS UCT formula;
- MCTS expansion/rollout/backpropagation details;
- duplicate Minimax diagrams;
- duplicate LLM-as-judge tables;
- duplicate human annotation tables.

You can still mention MCTS proudly, but the main deck should not spend 5-6
slides deriving it unless the presentation is specifically about MCTS.

### Add

The current V2 still needs three missing story slides:

1. **Architecture / Pipeline**
   Show how the project was actually executed:

   ```text
   documents -> chunks -> OpenAI entity/relation extraction -> graph
             -> communities/reports -> OpenAI embeddings
             -> retrieval variants -> answer generation -> LLM judge + humans
   ```

2. **Why Naive Wins**
   Explain the diagnostic:

   - `naive` gives mostly direct source chunks;
   - graph modes spend context on entities, paths, communities;
   - QA questions reward exact evidence;
   - therefore graph traversal can be good but final answer evidence can still
     be worse.

3. **Context Composition Bottleneck**
   Define it explicitly:

   > Retrieval decides where to look. Context composition decides what the LLM
   > actually sees.

   This is the key lesson.

## Important Correction: "LLMaJ Strict"

Several V2 slides call the LLM judge table "strict", but the shown tables are
raw swapped-order judge rows:

- 30 questions;
- 2 answer orders;
- 60 judge calls per variant.

Strict swapped-order means:

- judge the same pair twice with A/B reversed;
- map answers back to variant names;
- keep only if both orders choose the same underlying variant;
- otherwise mark as `disagree`.

Recommendation:

- Rename current tables to **LLM pairwise judge, swapped order**.
- Use **strict** only on the human-vs-LLM slide where disagreements are shown.

## Suggested Final 15-Minute Deck

Aim for 15 slides.

### Slide 1: Title

Keep.

Speaker notes:

> We worked on HiRAG, a hierarchical graph-based retrieval-augmented generation
> system. Our project had two goals: reproduce and evaluate it on full OpenAI
> runs, and test whether query-aware graph traversal improves retrieval.

Time: 20s.

### Slide 2: Plan

Keep, but make the plan more story-driven:

1. What problem HiRAG solves.
2. What we changed.
3. How we evaluated.
4. What we found.
5. What this teaches us.

Speaker notes:

> The important part is not only the implementation, but what the evaluation
> revealed about graph retrieval.

Time: 20s.

### Slide 3: RAG, GraphRAG, HiRAG

Keep one of the two concept slides, not both.

Speaker notes:

> Standard RAG retrieves chunks. It is simple and strong, but has no explicit
> structure. GraphRAG adds entities and relations, but often works at one level
> of granularity. HiRAG adds hierarchy: local entities, global community
> summaries, and bridge paths between them.

Time: 1 min.

### Slide 4: What HiRAG Actually Gives The LLM

Add or modify concept slide.

Content:

- local entity descriptions;
- community reports;
- bridge relations/path;
- source text units/snippets.

Speaker notes:

> A key point for our later results: HiRAG does not simply replace chunks with a
> graph. It builds graph context and still attaches source text units. The final
> answer depends heavily on how these pieces are composed into the prompt.

Time: 1 min.

### Slide 5: Problems We Identified

Keep current "Problèmes de HiRAG" slide.

Speaker notes:

> We identified three practical weaknesses. First, source chunks can saturate
> the context. Second, local entity retrieval is dense retrieval only, so it can
> miss more query-specific entities. Third, the bridge path is mostly structural:
> it is not sufficiently conditioned by the query.

Time: 1 min.

### Slide 6: Query-Aware Source Snippets

Keep, but simplify language.

Change:

- replace "punctuation and \\n" with French wording;
- emphasize that this is context compression, not a different answer model.

Speaker notes:

> Instead of dumping full chunks, we split a source chunk into windows and keep
> the window with the strongest overlap with the query. This reduces context
> explosion while keeping direct evidence.

Time: 45s.

### Slide 7: Reranking Local Entities

Keep.

Speaker notes:

> We also tested reranking the local entities. Dense retrieval gives us a larger
> candidate set, then a late-interaction ColBERT-style reranker reorders those
> candidates to keep more query-relevant entities.

Time: 45s.

### Slide 8: Query-Aware Bridge

Keep one bridge overview slide plus one edge-cost slide.

Speaker notes:

> The original HiRAG bridge is close to a shortest path between selected
> entities. We asked: if the query is about a specific relationship, should all
> edges be treated equally? Our answer is no. We embed edge descriptions and
> compute a query-dependent cost.

Time: 1 min.

### Slide 9: Weighted Dijkstra, Minimax, MCTS

Compress several current slides into one.

Content:

| Method | Intuition |
|---|---|
| Weighted Dijkstra | minimize total query-aware edge cost |
| Minimax | avoid a single bad edge |
| Budgeted Minimax | avoid bad edges under path budget |
| MCTS | explore bridge construction as sequential decision |

Speaker notes:

> These variants represent different assumptions about what makes a good bridge.
> Weighted Dijkstra optimizes total cost. Minimax focuses on the weakest link.
> MCTS is the most exploratory: it treats bridge construction as a sequence of
> choices under a token and path budget.

Time: 1 min.

### Slide 10: MCTS: Why It Is Interesting

Keep one high-level MCTS slide. Move UCT/rollout/backpropagation to backup.

Speaker notes:

> MCTS is interesting because retrieval is not just nearest-neighbor search. It
> is a planning problem: each hop changes what evidence we can include later.
> MCTS lets us explore several possible bridge paths and reward paths that are
> relevant, short, and token-efficient.

Important caveat:

> Experimentally, MCTS is robust but not a clear winner. Humans judged MCTS
> answers as answering the question reliably, but they did not strongly prefer
> them over classic HiRAG.

Time: 1 min.

### Slide 11: Evaluation Pipeline

Add this slide.

Content:

```text
OpenAI Batch graph construction:
entity extraction -> relation extraction -> clustering -> community reports
-> OpenAI embeddings -> retrieval variants -> answers -> LLM judge -> humans
```

Speaker notes:

> To avoid local vLLM limitations, we rebuilt Agriculture and Mix with OpenAI:
> GPT-5.4-mini for extraction, reports, answers and judge, and
> text-embedding-3-small for embeddings. This gives us a homogeneous and
> reproducible evaluation pipeline.

Time: 1 min.

### Slide 12: Dataset And Metrics

Keep current benchmark table.

Clarify:

- Agriculture has 100 available questions, Mix has 130;
- we evaluated q30 per dataset;
- generated 7 variants;
- LLM judge compares every variant against HiRAG with swapped order.

Speaker notes:

> We evaluated 30 questions per dataset. For Core7, that means 60
> dataset-question pairs, 420 generated answers, and 720 LLM judge calls.

Time: 1 min.

### Slide 13: LLM-as-Judge Results

Use one compact slide.

Recommendation:

- show Agriculture and Mix together;
- do not show two duplicate Mix tables;
- label as "LLM pairwise judge, swapped order".

Speaker notes:

> The surprising result is that naive RAG is very strong. It beats classic HiRAG
> by the LLM judge on both datasets. Our graph variants are mixed: MCTS and
> weighted variants are sometimes competitive, but no graph variant dominates.

Time: 1.5 min.

### Slide 14: Human Annotation And Human vs LLM

Use generated charts:

- `human_vs_llm_win_rates.png`;
- or `human_vs_llm_decision_breakdown.png`;
- and the table with human win rate/span density/gain.

Speaker notes:

> Human annotation confirms the strongest signal: naive is preferred over
> HiRAG. MCTS is close to HiRAG, with many ties. The LLM judge is useful but
> order-sensitive on close comparisons, so human annotation is essential for
> calibration.

Time: 1.5 min.

### Slide 15: Why Does Naive Win?

Add this slide. This is essential.

Content:

| Naive | Graph modes |
|---|---|
| mostly raw source chunks | entity descriptions |
| direct evidence | bridge paths |
| more exact wording | community summaries |
| less structure | more abstraction |

Speaker notes:

> This was our main finding. Graph retrieval is not automatically better. On
> QA-style questions, the model often needs the exact passage. Naive retrieval
> gives more raw evidence. HiRAG gives more structure, but that structure can
> consume the context budget.

Time: 1.5 min.

### Slide 16: Community2200 And Context Composition

Optional but useful if time allows.

Speaker notes:

> We found a concrete issue: community reports were initially dropped because
> the community report budget was too low. We reran with a higher budget. This
> restored community context, but it did not fully solve the problem. That
> taught us that the issue is not just adding more graph context; it is choosing
> the right mix of graph structure and source evidence.

Time: 1 min.

### Slide 17: Conclusion

Current V2 conclusion is empty. Replace it with three bullets:

- HiRAG is useful for structured and multi-scale retrieval.
- Our query-aware bridge variants, especially MCTS, are promising and robust but
  not enough alone.
- The key bottleneck is context composition: the final prompt must preserve
  enough exact source evidence.

Speaker notes:

> Our next step would be hybrid graph+source retrieval: use the graph to decide
> where to look, but reserve a guaranteed budget for raw source passages.

Time: 1 min.

## What V2 Represents Well

- It clearly shows that we understood HiRAG conceptually.
- It shows real engineering work: OpenAI embeddings, graph construction,
  reranking, weighted paths, minimax, MCTS.
- It includes the new human annotation results.
- It starts to surface the important `naive` result.

## What V2 Still Underrepresents

### 1. The Full OpenAI Architecture

The audience may not understand how much work went into rebuilding the graph.
Add one architecture slide.

### 2. The Evaluation Coverage

Clarify:

- q30 per dataset;
- 7 variants;
- answer prompt and model fixed;
- only retrieval context changes;
- LLM judge with swapped order;
- human annotation sampled from Core7.

### 3. The Central Diagnosis

Add a clear "Why Naive Wins" slide. Without it, the audience may think:

> So graph retrieval just failed?

But the real message is more subtle:

> Graph retrieval found useful structure, but answer quality was bottlenecked by
> what evidence reached the LLM.

### 4. MCTS Interpretation

V2 contains many MCTS mechanics, but not enough result interpretation.

Say:

> MCTS is conceptually interesting and performs robustly on answer adequacy, but
> humans do not strongly prefer it because the final context often remains
> similar to HiRAG or lacks more direct source evidence.

## Suggested Backup Slides

Move these to backup:

- detailed UCT formula;
- MCTS expansion;
- MCTS rollout;
- MCTS backpropagation;
- duplicate Minimax diagram;
- duplicate evaluation tables;
- community2200 detailed table.

If asked, these backup slides show technical depth.

## Short Oral Script

Use this as the 15-minute version:

> HiRAG tries to improve RAG by moving from raw chunk retrieval to hierarchical
> graph retrieval. It combines local entity knowledge, global community
> knowledge, and bridge paths between them.

> We identified three limitations: full chunks can saturate context, local
> entity retrieval is dense-only, and the bridge path is not query-aware enough.
> We implemented query-aware source snippets, ColBERT-style reranking, weighted
> Dijkstra, Minimax, budgeted Minimax, and MCTS.

> We rebuilt Agriculture and Mix with an OpenAI-based pipeline. We generated
> answers for seven variants, judged them with an LLM using swapped answer
> order, and then collected human annotations on a subset.

> The surprising result is that naive RAG is very strong. It often beats graph
> modes, including our variants. Human annotation confirms this for the strongest
> comparison: naive is preferred over classic HiRAG.

> This does not mean HiRAG is useless. HiRAG often answers the question, and MCTS
> is robust. But the graph context often spends tokens on entities, paths, and
> summaries, while QA questions need exact source evidence. In other words,
> retrieval path quality is not enough; context composition matters.

> Our conclusion is that the next step should be hybrid graph+source retrieval:
> use the graph to decide where to search, but guarantee enough budget for raw
> evidence in the final prompt.

## Final Thesis

> HiRAG improves the structure of retrieval, but our experiments show that the
> final context must balance graph reasoning with direct source evidence. Better
> graph traversal helps, but context composition is the bottleneck.
