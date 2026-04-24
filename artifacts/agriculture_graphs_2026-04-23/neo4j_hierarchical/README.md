# Neo4j Hierarchical Graph Artifacts

This folder contains Neo4j/Bloom-friendly graph artifacts generated from the curated agriculture exports.

## Files

- `agriculture_baseline_no_glean_hierarchical.graphml`
- `agriculture_lean_hierarchical.graphml`
- `agriculture_ultra_lean_hierarchical.graphml`
- `hierarchical_manifest.json`
- `neo4j_import_queries.cypher`

## What Was Added

Compared to the original graph files, these hierarchical variants include:

- Original entity graph edges as `RELATES_TO`
- Community nodes (`kind=Community`) from `*_community_reports.json`
- Entity to community links (`IN_COMMUNITY`)
- Parent/child community links (`HAS_SUBCOMMUNITY`)
- Readability helper fields:
  - `description_clean` on entities/edges
  - JSON list fields for split segments (`*_segments_json`)

## Import (Neo4j + APOC)

1. Copy one `*_hierarchical.graphml` file into Neo4j import dir.
2. Run the statements in `neo4j_import_queries.cypher`.
3. In Bloom, build a Perspective using labels `Entity` and `Community` and relationships `RELATES_TO`, `IN_COMMUNITY`, `HAS_SUBCOMMUNITY`.

## Why This Helps Hierarchical Visualization

The original GraphML stores hierarchy hints as text (`clusters`), but not as explicit graph topology.
These artifacts materialize hierarchy as explicit nodes and relationships, enabling:

- Level-based filtering (`Community.level`)
- Drilldown/rollup navigation along `HAS_SUBCOMMUNITY`
- Entity-centric exploration within community scopes via `IN_COMMUNITY`

