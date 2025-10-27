# query_graph.py — Improved Cypher generator + safe property checks
import os
import re
from typing import List, Dict, Any

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

NEO4J_URI = "neo4j+s://49921c84.databases.neo4j.io"
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "iJKV1cRrVTOan16QYmD1rXc76jix61YqtAgrpgVyNew")

# Simple year detector like "2024"
YEAR_RE = re.compile(r"(19|20)\d{2}")

class GraphQuerier:
    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def _guess_cypher(self, question: str) -> Dict[str, Any]:
        """
        Very light NL → Cypher mapping for a tiny Concept graph.

        We expect triples of the form:
          (Concept {name: '<Indicator>'})-[:RELATION {type, year, source}]->(Concept {name: '<value or label>'})
        """
        q = question.lower()

        # Try to extract a year if present
        year = None
        m = YEAR_RE.search(question)
        if m:
            year = int(m.group(0))

        # Known indicator name mappings
        indicators = {
            "inflation": "Inflation",
            "unemployment": "Unemployment Rate",
        }

        num_regex = r"^[0-9]+(?:\.[0-9]+)?%?$"  # 12, 12.3, 12%, 12.3%

        for key, concept_name in indicators.items():
            if key in q:
                cypher = """
                    MATCH (i:Concept {name: $concept})-[r:RELATION]->(v:Concept)
                    // Keep only relations that look like facts: either a typed edge or a numeric-ish tail node
                    WHERE (r.type IS NOT NULL OR v.name =~ $num_regex)
                    {year_filter}
                    RETURN
                        r.year   AS year,
                        coalesce(r.source, 'unknown') AS source,
                        v.name   AS value
                    ORDER BY coalesce(r.year, 0) DESC
                    LIMIT 10
                """.replace("{year_filter}", "AND r.year = $year" if year else "")
                params = {"concept": concept_name, "num_regex": num_regex}
                if year:
                    params["year"] = year
                return {"cypher": cypher, "params": params}

        # Fallback: user asked only by year (e.g., "show facts for 2023")
        if year:
            return {
                "cypher": """
                    MATCH ()-[r:RELATION]->(v:Concept)
                    WHERE r.year = $year
                    RETURN
                        r.year AS year,
                        coalesce(r.source, 'unknown') AS source,
                        v.name AS value
                    LIMIT 10
                """,
                "params": {"year": year},
            }

        # Nothing we can confidently translate
        return {"cypher": None, "params": {}}

    def query(self, question: str) -> List[Dict[str, Any]]:
        plan = self._guess_cypher(question)
        if not plan["cypher"]:
            return []

        try:
            with self.driver.session() as s:
                res = s.run(plan["cypher"], plan["params"])
                return res.data()
        except Neo4jError as e:
            print(f"⚠️ Neo4j error: {e}")
            return []

if __name__ == "__main__":
    g = GraphQuerier()
    try:
        print("💬 Ask me something, e.g., 'What was inflation in 2022?'")
        q = input("You: ")
        rows = g.query(q)
        if not rows:
            print("📭 No matching facts in the graph.")
        else:
            for r in rows:
                y = r.get("year")
                val = r.get("value")
                src = r.get("source") or "unknown"
                print(f"• {y}: {val}  [source: {src}]")
    finally:
        g.close()
