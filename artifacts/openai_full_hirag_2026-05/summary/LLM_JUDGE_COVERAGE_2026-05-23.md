# LLM Judge Coverage

Date: 2026-05-23

This note clarifies how many questions were evaluated with LLM-as-judge and how
that differs from the human annotation packet.

## Important Distinction

The main OpenAI evaluation runs used a `q30` subset:

- Agriculture query file has 100 questions available.
- Mix query file has 130 questions available.
- The evaluated OpenAI runs used `--query-limit 30`.

So the LLM judge covered all questions in the chosen evaluation subset, but not
all questions in the full dataset files.

## Current OpenAI Full HiRAG, 5 Variants

Location:

```text
artifacts/openai_full_hirag_2026-05/agriculture/
artifacts/openai_full_hirag_2026-05/mix/
```

Variants answered:

```text
hi, naive, hi_nobridge, hi_rerank_weighted, hi_minmax_budgeted
```

Coverage per dataset:

- 30 unique questions.
- 150 generated answers: 30 questions x 5 variants.
- 240 LLM judge rows: 30 questions x 4 variants-vs-`hi` x 2 swapped orders.

Across Agriculture + Mix:

- 60 unique dataset-question pairs.
- 300 generated answers.
- 480 LLM judge rows.

## Core 7 Evaluation

Location:

```text
artifacts/openai_full_hirag_2026-05/core7_eval/
```

Variants answered:

```text
naive, hi, hi_weighted, hi_minmax, hi_minmax_budgeted, hi_rerank_weighted, hi_mcts
```

Coverage per dataset:

- 30 unique questions.
- 210 generated answers: 30 questions x 7 variants.
- 360 LLM judge rows: 30 questions x 6 variants-vs-`hi` x 2 swapped orders.

Across Agriculture + Mix:

- 60 unique dataset-question pairs.
- 420 generated answers.
- 720 LLM judge rows.

This is not just the human annotation subset. Every generated Core 7 answer pair
against `hi` was judged with swapped order.

## Core 7 Community2200 Rerun

Location:

```text
artifacts/openai_full_hirag_2026-05/core7_eval_community2200/
```

Same question and variant coverage as Core 7:

- 60 unique dataset-question pairs across Agriculture + Mix.
- 420 generated answers.
- 720 LLM judge rows.

Difference from Core 7:

- `max_token_for_community_report=2200`.
- `cluster-*` provenance ids are ignored during text-chunk lookup.

## Human Annotation Packet

Location:

```text
artifacts/openai_full_hirag_2026-05/core7_eval/human_annotation_packet/
```

Coverage:

- 60 human annotation rows.
- 42 unique dataset-question ids.
- 20 rows: `hi` vs `naive`.
- 20 rows: `hi` vs `hi_mcts`.
- 20 rows: `hi` vs `hi_minmax_budgeted`.

The human packet is a sampled subset for manual annotation. It is not the same
as the LLM judge coverage. Some questions appear in more than one human pair,
which is why 60 rows correspond to 42 unique dataset-question ids.

## Legacy/Older Shared Runs

Several older shared artifacts also use `q30`:

- `artifacts/agriculture_graphs_2026-04-23/final_evaluation/`
- `artifacts/legacy_answer_eval_2026-05-21/`
- `artifacts/mix_batch_requests_2026-05-10/final_*`

Typical coverage in those older runs:

- 30 unique questions.
- 90 answers: 30 questions x 3 variants.
- 120 LLM judge rows: 30 questions x 2 variants-vs-`hi` x 2 swapped orders.

These are useful historically, but the current comparable runs are the OpenAI
Full HiRAG 5-variant run, Core 7, and Core 7 Community2200.
