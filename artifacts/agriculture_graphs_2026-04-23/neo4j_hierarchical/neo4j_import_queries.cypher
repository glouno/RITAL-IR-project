// Neo4j import helpers for hierarchical GraphML artifacts
// Usage example (Neo4j Browser):
//   :param file => 'agriculture_ultra_lean_hierarchical.graphml'
//   CALL apoc.import.graphml($file, {readLabels: true, storeNodeIds: true, defaultRelationshipType: 'RELATES_TO'});

// 0) Optional: clean current DB first (DANGER: removes all nodes)
// MATCH (n) DETACH DELETE n;

// 1) Import one hierarchical graphml file from Neo4j import dir
// :param file => 'agriculture_ultra_lean_hierarchical.graphml'
CALL apoc.import.graphml(
  $file,
  {readLabels: true, storeNodeIds: true, defaultRelationshipType: 'RELATES_TO'}
)
YIELD file, source, format, nodes, relationships, properties, time, done
RETURN file, source, format, nodes, relationships, properties, time, done;

// 2) Add native labels from exported 'kind' field for easier Bloom perspectives
MATCH (n)
WHERE n.kind = 'Entity'
SET n:Entity;

MATCH (n)
WHERE n.kind = 'Community'
SET n:Community;

// 3) Normalize relationship types from exported 'kind' field
MATCH ()-[r]-()
WHERE r.kind = 'IN_COMMUNITY'
CALL apoc.refactor.setType(r, 'IN_COMMUNITY') YIELD input, output
RETURN count(*) AS in_community_retyped;

MATCH ()-[r]-()
WHERE r.kind = 'HAS_SUBCOMMUNITY'
CALL apoc.refactor.setType(r, 'HAS_SUBCOMMUNITY') YIELD input, output
RETURN count(*) AS has_subcommunity_retyped;

MATCH ()-[r]-()
WHERE r.kind = 'RELATES_TO' AND type(r) <> 'RELATES_TO'
CALL apoc.refactor.setType(r, 'RELATES_TO') YIELD input, output
RETURN count(*) AS relates_to_retyped;

// 4) Helpful indexes for interactive exploration
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type);
CREATE INDEX community_level IF NOT EXISTS FOR (c:Community) ON (c.level);
CREATE INDEX community_id IF NOT EXISTS FOR (c:Community) ON (c.community_id);

// 5) Quick sanity checks
MATCH (e:Entity) RETURN count(e) AS entity_nodes;
MATCH (c:Community) RETURN count(c) AS community_nodes;
MATCH ()-[r:RELATES_TO]-() RETURN count(r) AS relates_to_edges;
MATCH ()-[r:IN_COMMUNITY]->() RETURN count(r) AS in_community_edges;
MATCH ()-[r:HAS_SUBCOMMUNITY]->() RETURN count(r) AS has_subcommunity_edges;

// 6) Example hierarchy queries
// Communities by level
MATCH (c:Community)
RETURN c.level AS level, count(*) AS communities
ORDER BY level;

// Drilldown from one community id
// :param cid => 2280
MATCH (c:Community {community_id: $cid})<-[:IN_COMMUNITY]-(e:Entity)
RETURN c, e
LIMIT 500;

// Community hierarchy neighborhood
// :param cid => 2280
MATCH (c:Community {community_id: $cid})-[:HAS_SUBCOMMUNITY*0..2]->(sub:Community)
RETURN c, sub
LIMIT 500;
