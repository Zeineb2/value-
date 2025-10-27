# ingest_graph.py — safer + idempotent graph load (improved)
import os, json
from neo4j import GraphDatabase

NEO4J_URI = "neo4j+s://49921c84.databases.neo4j.io"
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "iJKV1cRrVTOan16QYmD1rXc76jix61YqtAgrpgVyNew")

TRIPLES_FILE = os.getenv(
    "TRIPLES_FILE",
    os.path.join("..", "scraping", "output", "graph_triples.json")
)

BOOTSTRAP = """
CREATE CONSTRAINT concept_name IF NOT EXISTS
FOR (c:Concept) REQUIRE c.name IS UNIQUE;
"""

CREATE_REL = """
MERGE (h:Concept {name: $head})
MERGE (t:Concept {name: $tail})
MERGE (h)-[r:RELATION]->(t)
ON CREATE SET
    r.type = $relation,
    r.year = $year,
    r.source = $source
ON MATCH SET
    r.type = coalesce(r.type, $relation),
    r.year = coalesce(r.year, $year),
    r.source = coalesce(r.source, $source)

"""

def load_triples(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ingest_triples(session, triples, batch_size=100):
    total = len(triples)
    for i in range(0, total, batch_size):
        batch = triples[i:i+batch_size]
        for tr in batch:
            head = tr.get("head")
            relation = tr.get("relation")
            tail = tr.get("tail")
            if not (head and relation and tail):
                continue
            params = {
                "head": head,
                "relation": relation,
                "tail": tail,
                "year": tr.get("year"),
                "source": tr.get("source", "unknown"),
            }
            session.run(CREATE_REL, params)
        print(f"✅ Ingested {min(i+batch_size, total)}/{total} triples...")

def main():
    try:
        triples = load_triples(TRIPLES_FILE)
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as s:
            s.run(BOOTSTRAP)
            ingest_triples(s, triples)
        driver.close()
        print(f"🎉 Finished ingesting {len(triples)} triples from {TRIPLES_FILE}")
    except Exception as e:
        print(f"❌ Error ingesting triples: {e}")

if __name__ == "__main__":
    main()
