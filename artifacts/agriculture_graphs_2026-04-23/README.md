# Agriculture HiRAG Curated Export (2026-04-23)

This folder contains a commit-friendly snapshot of the completed agriculture graph builds for three prompt strategies:

- `baseline_no_glean`
- `lean`
- `ultra_lean`

The source run outputs were copied from `.runs/` and renamed so files are strategy-explicit.

## Folder Layout

- `graphs/`: Graph structure (`.graphml`) for visual inspection and network tooling
- `metrics/`: Community-report exports and benchmark summary JSON
- `metadata/`: Resume logs that confirm recovery/completion flow

## Files Included

### GraphML (core graph structure)

- `graphs/agriculture_baseline_no_glean.graphml`
- `graphs/agriculture_lean.graphml`
- `graphs/agriculture_ultra_lean.graphml`

### JSON metrics and reports

- `metrics/agriculture_baseline_no_glean_community_reports.json`
- `metrics/agriculture_lean_community_reports.json`
- `metrics/agriculture_ultra_lean_community_reports.json`
- `metrics/agriculture_ultra_lean_benchmark_summary.json`

### Metadata logs

- `metadata/screen_resume.log`
- `metadata/screen_resume2.log`

## What The JSON Files Contain

- `*_community_reports.json`:
  - Community-level synthesized summaries generated after graph clustering.
  - Typical content includes report text and per-community structure used by retrieval.
  - Useful for qualitative analysis and comparing how strategies summarize communities.

- `agriculture_ultra_lean_benchmark_summary.json`:
  - Benchmark run metadata and runtime stats for `ultra_lean`.
  - Includes dataset info, preflight checks, token usage, latency, and counts (chunks/entities/relations/communities).

## Visualization

You can visualize each GraphML with Gephi/Cytoscape directly, or with the in-repo helper script:

- `case/graphml_visualize.py`

If you use the helper script, point it to one of the files in `graphs/`.

## Notes

- This export intentionally keeps only the most useful shareable outputs.
- Raw vector stores and internal caches were not included to avoid unnecessary repository bloat.
