# Neo4j Graph Visualization Guide (DGX + Mac)

This guide explains how to visualize the exported HiRAG graphs with Neo4j from your own laptop.

It covers:

- Why `podman ps` may not show your colleague's container
- How to connect to an existing Neo4j instance (recommended if available)
- How to run your own Neo4j Podman container (recommended for isolation)
- How to import GraphML and query large graphs without freezing the browser

## 1) Why You Cannot See Your Colleague's Rootless Podman Container

If your colleague started Neo4j with **rootless Podman**, containers are user-scoped.

- `podman ps` only shows containers owned by your Linux user.
- You cannot list/manage another user's rootless containers directly.
- You can still use their service if ports are published and reachable (for example `7474`, `7687`).

## 2) Prerequisites

From this repo, the curated GraphML files are:

- `artifacts/agriculture_graphs_2026-04-23/graphs/agriculture_baseline_no_glean.graphml`
- `artifacts/agriculture_graphs_2026-04-23/graphs/agriculture_lean.graphml`
- `artifacts/agriculture_graphs_2026-04-23/graphs/agriculture_ultra_lean.graphml`

Neo4j ports:

- `7474`: Browser HTTP
- `7687`: Bolt protocol

## 3) Option A: Connect to Existing Neo4j (Fastest)

Use this when someone already has a Neo4j service running on DGX.

### 3.1 On DGX, verify service is reachable

```bash
curl -sS http://127.0.0.1:7474 | head
```

If it returns JSON metadata (`neo4j_version`, `bolt_direct`, etc.), service is up.

### 3.2 On your Mac, create SSH tunnel

```bash
ssh -N \
  -L 7474:127.0.0.1:7474 \
  -L 7687:127.0.0.1:7687 \
  <your_user>@<dgx_host>
```

Then open:

- `http://localhost:7474/browser/`

Use credentials shared by the container owner.

### 3.3 Community Edition caveat

Neo4j Community is single-database in practice for this workflow.
If that instance is already used by another project (for example PMIND), you should avoid overwriting data.

Use Option B (own container) for clean separation.

## 4) Option B: Run Your Own Neo4j Podman Container (Recommended)

This is the safest setup for your own project data.

### 4.1 Start from repo root on DGX

```bash
cd /home/paulbeglin/projects/RITAL-IR-project
mkdir -p .neo4j-rital/{data,logs,import}

cp artifacts/agriculture_graphs_2026-04-23/graphs/agriculture_baseline_no_glean.graphml .neo4j-rital/import/
cp artifacts/agriculture_graphs_2026-04-23/graphs/agriculture_lean.graphml .neo4j-rital/import/
cp artifacts/agriculture_graphs_2026-04-23/graphs/agriculture_ultra_lean.graphml .neo4j-rital/import/
```

### 4.2 Run Neo4j on alternate ports (avoid conflicts)

```bash
podman run -d --name neo4j-rital \
  -p 17474:7474 -p 17687:7687 \
  -v "$PWD/.neo4j-rital/data:/data" \
  -v "$PWD/.neo4j-rital/logs:/logs" \
  -v "$PWD/.neo4j-rital/import:/var/lib/neo4j/import" \
  -e NEO4J_AUTH=neo4j/rital-graph-2026 \
  -e NEO4J_PLUGINS='["apoc"]' \
  -e NEO4J_apoc_import_file_enabled=true \
  -e NEO4J_dbms_security_procedures_unrestricted=apoc.* \
  docker.io/library/neo4j:5
```

### 4.3 Verify locally on DGX

```bash
curl -sS http://127.0.0.1:17474 | head
```

### 4.4 Tunnel from Mac to your own Neo4j

```bash
ssh -N \
  -L 17474:127.0.0.1:17474 \
  -L 17687:127.0.0.1:17687 \
  <your_user>@<dgx_host>
```

Open:

- `http://localhost:17474/browser/`

Login:

- user: `neo4j`
- password: `rital-graph-2026`

## 5) Import GraphML

In Neo4j Browser, run one import at a time.

### 5.1 Import ultra_lean

```cypher
CALL apoc.import.graphml(
  "file:///agriculture_ultra_lean.graphml",
  {readLabels:true, storeNodeIds:true}
);
```

### 5.2 Switch to another variant

If you want to replace current graph content:

```cypher
MATCH (n) DETACH DELETE n;
```

Then import another file (baseline or lean).

## 6) Query and Visualize Without Freezing

Do not render the entire graph at once.

Use bounded queries:

```cypher
MATCH (n)-[r]->(m)
RETURN n,r,m
LIMIT 300;
```

```cypher
MATCH (n)
WITH n, size((n)--()) AS deg
ORDER BY deg DESC
LIMIT 20
MATCH (n)-[r]-(m)
RETURN n,r,m
LIMIT 800;
```

```cypher
MATCH (n {id:"TERESA"})-[r*1..2]-(m)
RETURN n,r,m
LIMIT 500;
```

Tip: start with `LIMIT 100` then increase gradually.

## 7) Useful Operational Commands

### Container status

```bash
podman ps --filter name=neo4j-rital
```

### Container logs

```bash
podman logs -f neo4j-rital
```

### Stop/start container

```bash
podman stop neo4j-rital
podman start neo4j-rital
```

### Remove container (keeps mounted data on disk)

```bash
podman rm -f neo4j-rital
```

## 8) Troubleshooting

### APOC procedure not found

- Confirm container was started with APOC env vars shown above.
- Check logs: `podman logs neo4j-rital`.

### Browser opens but queries fail

- Confirm tunnel is active.
- Confirm container is up.
- Try a simple query: `MATCH (n) RETURN count(n);`

### Slow visualization in Browser

- This is normal for 10k+ node full renders.
- Always use filtered subgraph queries with limits.

## 9) Recommended Team Workflow

- Keep PMIND and RITAL in separate Neo4j containers.
- Use separate ports and credentials.
- Import one graph variant at a time when exploring.
- Share reusable Cypher snippets in repo docs for consistent analysis.
