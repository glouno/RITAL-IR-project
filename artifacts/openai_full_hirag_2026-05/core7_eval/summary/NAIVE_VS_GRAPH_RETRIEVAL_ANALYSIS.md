# Naive vs Graph Retrieval Analysis

Date: 2026-05-23

This note analyzes why `naive` beats the graph modes so clearly in the Core 7
Agriculture + Mix evaluation.

## Executive Takeaway

`naive` is winning mostly because this benchmark is answer-evidence heavy:
questions often ask for specific facts or short explanatory passages that are
present verbatim in source chunks. The graph modes usually retrieve the right
entities, but they spend much of the final prompt budget on entity descriptions
and reasoning-path tables rather than raw source evidence.

In short:

- `naive` gives the answer model more direct source text.
- `hi` gives the answer model a structured graph view, but less textual proof.
- community reports are effectively absent because the per-report token cap is
  too low for the first selected community report.
- some graph-selected entities point to missing text chunk ids, reducing the
  usable source-doc evidence available to graph modes.

## Key Numbers

### Mean Context Tokens

| Dataset | naive | hi | hi_minmax_budgeted | hi_rerank_weighted | hi_mcts |
|---|---:|---:|---:|---:|---:|
| Agriculture | 6016 | 5040 | 5141 | 5801 | 4990 |
| Mix | 6614 | 3174 | 3206 | 3597 | 3159 |

For Mix, `naive` gives about twice as many context tokens as `hi`.

### Raw Source Evidence Share

For graph modes, the final prompt is split into:

- `Backgrounds`: community reports;
- `Reasoning Path`: graph relation/path table;
- `Detail Entity Information`: entity descriptions;
- `Source Documents`: selected original text snippets.

Mean share of graph prompt spent on `Source Documents`:

| Dataset | hi | hi_minmax_budgeted | hi_rerank_weighted | hi_mcts | naive |
|---|---:|---:|---:|---:|---:|
| Agriculture | 17% | 17% | 14% | 17% | ~100% |
| Mix | 29% | 28% | 25% | 29% | ~100% |

This is the biggest explanation. The graph prompts are not empty, but most of
their tokens are graph abstractions rather than original evidence.

### Community Reports Are Not Helping

Mean retrieved communities:

| Dataset | hi | graph variants |
|---|---:|---:|
| Agriculture | 0.0 | ~0.0 |
| Mix | 0.0 | ~0.0 |

Why: `max_token_for_community_report` in the answer export defaults to `1000`.
The first selected related community report is usually larger than this. The
current truncation helper returns `[]` when the first item alone exceeds the
budget, so no community report is included.

Measured first selected community report length:

| Dataset | Queries where first selected report > 1000 tokens | Mean first-report tokens |
|---|---:|---:|
| Agriculture | 29/30 | 1320 |
| Mix | 30/30 | 1337 |

So our “global” graph context is practically disabled in this run.

### Missing Text Chunk References

For selected entities, some `source_id` chunk references do not exist in
`kv_store_text_chunks.json`, triggering repeated warnings:

```text
Text chunks are missing, maybe the storage is damaged
```

Measured missing rates among selected entity source ids:

| Dataset | hi missing source ids | hi_rerank_weighted missing source ids |
|---|---:|---:|
| Agriculture | 22.5% | 20.9% |
| Mix | 50.3% | 49.0% |

This does not break retrieval, but it reduces how much original source evidence
the graph modes can attach to the final prompt.

## Qualitative Findings

### Example: Mix Q0, Mary And *The Witch of Atlas*

Question:

```text
What is the main objection Mary has to the poem "The Witch of Atlas"?
```

`naive` retrieves the opening source passage, including the explicit evidence:

```text
ON HER OBJECTING TO THE FOLLOWING POEM, UPON THE SCORE OF ITS CONTAINING NO HUMAN INTEREST
```

`hi` retrieves relevant entities (`THE WITCH OF ATLAS`, `PERCY BYSSHE SHELLEY`,
`MARY`, etc.) but answers that the exact objection is not spelled out. The graph
context missed or diluted the exact source sentence.

This is a canonical failure mode: the graph retrieval is semantically close, but
the QA task rewards exact textual evidence.

### Example: Mix Q3, Abd al-Muttalib And The Kaaba

Question asks for a significant event involving Abd al-Muttalib and the Kaaba
and its regional religious impact.

`naive` answers with the Year of the Elephant and explains the attempted
destruction of the Kaaba.

`hi` focuses on Abd al-Muttalib's vow to sacrifice a son at the Kaaba. That is
related, but not the best answer to the question. The entity graph found
strongly related nodes, but selected the wrong local episode.

This suggests graph traversal needs better query-aware event/path selection, not
just better path algorithms.

### Example: Agriculture Q5, Two Dogmas Of Environmental Philosophy

`naive` identifies:

- dogma of pristine nature;
- dogma of stable balance / equilibrium.

`hi` and graph variants talk generally about environmental ethics, intrinsic
value, functional integrity, and pragmatism. These are related concepts, but
they miss the exact named answer. Again, direct source chunks preserve the
wording needed by the answer model.

## Why The Custom Graph Variants Help Only Modestly

The custom variants mainly change bridge path selection:

- weighted Dijkstra changes edge costs;
- minmax changes path objective;
- budgeted minmax controls path budget;
- MCTS explores candidate bridge paths;
- rerank changes local entity ordering.

These variants improve which graph path appears in the prompt, but they do not
fix the main bottleneck:

```text
The final prompt still contains too little direct source evidence and too much
entity/path abstraction for QA-style questions.
```

This explains why `hi_minmax_budgeted` is the best custom variant, but still
does not beat `naive` overall.

## Interpretation Of The LLM Judge Results

The LLM judge seems to prefer answers that are:

- directly grounded in source wording;
- complete for enumerative/explanatory questions;
- less hedged;
- more likely to include exact named facts.

That favors `naive` in this dataset because `naive` often passes larger,
verbatim chunks to the answer model.

This does not prove graph retrieval is useless. It means our current graph
prompt assembly is not yet exploiting the graph in a way that beats direct
chunk retrieval for these QA pairs.

## What To Do Next

### 1. Run Human Annotation

Use the 60-row blinded packet:

```text
artifacts/openai_full_hirag_2026-05/core7_eval/human_annotation_packet/human_pairwise_annotation_packet.csv
```

Priority: verify whether humans agree that `naive` is better, especially for
`hi` vs `naive`.

### 2. Fix Community Report Inclusion

The current community cap of `1000` is too low. Options:

- raise `max_token_for_community_report` to at least `1800` or `2200`;
- truncate each community report individually instead of dropping it if it is
  too large;
- use a query-focused compressed community report section.

This is the highest-priority graph-side fix because global context is currently
near-disabled.

### 3. Increase Source Evidence Budget For Graph Modes

Graph modes should include more original text evidence. Possible changes:

- allocate a minimum source-doc budget, for example 50% of final context tokens;
- include full source chunks for top graph-selected text units;
- include naive top chunks as a fallback/evidence supplement;
- reduce entity description verbosity when source evidence is scarce.

### 4. Repair Missing Source References

Investigate why selected graph entities reference chunk ids not present in
`kv_store_text_chunks.json`, especially Mix where about half of selected entity
source ids are missing.

This may come from graph materialization, entity merging, stale source ids, or
document/chunk id transformations.

### 5. Add A Hybrid Variant

Most promising next variant:

```text
hi_minmax_budgeted + query-aware source expansion + naive evidence fallback
```

Concrete behavior:

- use graph retrieval to select entities/paths;
- collect source chunks from selected entities and path entities;
- add naive top-k chunks if graph evidence is below a source-token threshold;
- allocate prompt budget explicitly across graph and source sections.

Suggested budget:

- 20% entity summaries;
- 20% reasoning path;
- 10-20% community report;
- 40-50% source snippets/chunks.

### 6. Evaluate With The Same Core 7 Harness

After implementing the above, rerun only Agriculture + Mix first:

- `hi`;
- `naive`;
- current `hi_minmax_budgeted`;
- new hybrid graph+source variant.

Then judge + human spot-check.
