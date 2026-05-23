# Human Annotation App Spec

Date: 2026-05-23

This document describes the Streamlit annotation app we want for the Core 7
HiRAG evaluation.

## Goal

Build a small Streamlit app that lets human annotators compare two blinded
answers for the same question.

The app should make annotation fast, safe, and consistent. Annotators should
never see model or variant names. The hidden A/B-to-variant mapping must stay
separate from the public annotation interface.

## Input Files

Use the public packet:

```text
artifacts/openai_full_hirag_2026-05/core7_eval/human_annotation_packet/human_pairwise_annotation_packet.csv
```

Keep this hidden metadata file available only to project maintainers:

```text
artifacts/openai_full_hirag_2026-05/core7_eval/human_annotation_packet/human_pairwise_hidden_metadata.jsonl
```

The CSV contains 60 rows:

- 20 rows: `hi` vs `naive`
- 20 rows: `hi` vs `hi_mcts`
- 20 rows: `hi` vs `hi_minmax_budgeted`

The visible CSV fields are:

- `packet_id`
- `query_id`
- `comparison_pair`
- `query`
- `answer_a`
- `answer_b`
- `does_answer_a_answer_question`
- `does_answer_b_answer_question`
- `preferred_answer`
- `useful_span_a`
- `useful_span_b`
- `notes`

Important: `comparison_pair` is useful for maintainers, but the annotation UI
should hide it by default because it leaks the compared variants.

## Annotation Questions

For each row, annotators answer:

1. Does Answer A answer the question?
   Allowed values: `yes`, `partially`, `no`.
2. Does Answer B answer the question?
   Allowed values: `yes`, `partially`, `no`.
3. Which answer do you prefer?
   Allowed values: `A`, `B`, `tie`, `neither`.
4. Useful span from Answer A.
   Free text. Ask annotators to copy the smallest useful text span that directly
   helps answer the question. Leave blank if no useful span exists.
5. Useful span from Answer B.
   Same rule as Answer A.
6. Optional notes.
   Free text for uncertainty, factual concerns, missing evidence, or why neither
   answer is satisfactory.

## UI Requirements

Use Streamlit.

Recommended layout:

- Sidebar:
  - annotator id input;
  - load CSV button or fixed path selector;
  - progress indicator: annotated rows / assigned rows;
  - filter: all / incomplete / completed;
  - optional packet id jump.
- Main page:
  - show `packet_id` and `query_id`;
  - show the question prominently;
  - two side-by-side panels: Answer A and Answer B;
  - radio controls for answerability and preference;
  - text areas for useful spans and notes;
  - save button;
  - next incomplete button.

Do not show variant names, hidden metadata, or `comparison_pair` to annotators.

## Assignment Strategy

For a quick first version, one annotator can label all 60 rows.

If multiple annotators are used:

- support an `annotator_id`;
- save one output file per annotator;
- allow all annotators to annotate all rows if we want inter-annotator
  agreement;
- or assign packet ids by modulo/hash if we want to split work.

Recommended default for reliability: at least two annotators label the same 60
rows, then resolve disagreements.

## Output Files

Save annotations to a new folder, for example:

```text
artifacts/openai_full_hirag_2026-05/core7_eval/human_annotation_results/
```

Suggested file naming:

```text
annotations_<annotator_id>.csv
annotations_<annotator_id>.jsonl
```

Every output row should include:

- `packet_id`
- `query_id`
- `annotator_id`
- `does_answer_a_answer_question`
- `does_answer_b_answer_question`
- `preferred_answer`
- `useful_span_a`
- `useful_span_b`
- `notes`
- `created_at`
- `updated_at`

The output should not include hidden variant mapping. A later analysis script
can join with `human_pairwise_hidden_metadata.jsonl`.

## Validation Rules

Before marking a row complete:

- `does_answer_a_answer_question` must be one of `yes`, `partially`, `no`;
- `does_answer_b_answer_question` must be one of `yes`, `partially`, `no`;
- `preferred_answer` must be one of `A`, `B`, `tie`, `neither`;
- useful spans may be blank;
- notes may be blank.

Warn, but do not block, if:

- the preferred answer is `A` while Answer A is marked `no`;
- the preferred answer is `B` while Answer B is marked `no`;
- preference is `tie` while answerability labels are very different.

## Persistence

The app should persist after every saved row, not only at the end.

Implementation options:

- simplest: update a CSV/JSONL file on each save;
- safer: keep a CSV plus periodic backup files;
- optional: use SQLite if concurrent annotators will write to the same server.

For the first version, per-annotator CSV files are enough and avoid concurrent
write conflicts.

## Analysis After Annotation

After annotations are collected, create a join script that:

- joins annotation rows with `human_pairwise_hidden_metadata.jsonl`;
- maps `preferred_answer` from A/B back to the true variant;
- computes human preference rates per pair;
- computes answerability rates per variant;
- compares human preference with LLM judge preference;
- reports disagreements and useful-span coverage.

Useful aggregate outputs:

- `human_preference_summary.csv`;
- `human_answerability_summary.csv`;
- `human_vs_llm_agreement.csv`;
- `human_annotation_disagreements.csv`;
- `HUMAN_ANNOTATION_SUMMARY.md`.

## Important Context For The Agent

The LLM judge found that `naive` strongly beats `hi` in this Core 7 run, while
`hi_minmax_budgeted` was the selected best custom graph variant. Human
annotation is meant to sanity-check these LLM-judge conclusions and inspect
whether preferred answers actually contain useful evidence.

The app should optimize for careful comparison, not model debugging. Keep the
UI neutral and blinded.
