# vectorization/query_vectorstore.py — alias-aware + growth vs level disambiguation
import re, json, pathlib
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

FAISS_DIR = "faiss_index"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

# ---------- Load indicator aliases ----------
def _load_indicator_aliases() -> Dict[str, list]:
    candidates = [
        pathlib.Path("scraping/economic_indicators.json"),
        pathlib.Path("scraping/economic_indicator.json"),
        pathlib.Path("economic_indicators.json"),
        pathlib.Path("economic_indicator.json"),
    ]
    path = next((p for p in candidates if p.exists()), None)

    aliases_map: Dict[str, list] = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        for it in items:
            canon = (it.get("Canonical Name") or it.get("CanonicalName") or "").strip()
            if not canon:
                continue
            aliases = it.get("Aliases") or []
            terms = {canon.lower(), *(a.lower() for a in aliases if isinstance(a, str))}
            aliases_map[canon] = sorted(terms)

    # ---- Manual extras (quick fix) ----
    # Add growth / rate variants that are commonly asked but may be missing in the file
    extras = {
        "GDP growth (annual %)": [
            "gdp growth", "real gdp growth", "growth of gdp", "croissance du pib",
            "taux de croissance du pib", "croissance économique"
        ],
        "Inflation rate": [
            "inflation rate", "taux d'inflation"
        ],
        "Current Account Balance (% of GDP)": [
            "current account deficit", "current account balance as % of gdp",
            "balance du compte courant", "déficit courant"
        ],
        "Unemployment Rate": [
            "unemployment", "unemployment rate", "taux de chômage", "chômage"
        ],
    }
    for canon, terms in extras.items():
        base = set(aliases_map.get(canon, []))
        aliases_map[canon] = sorted(base.union({canon.lower(), *[t.lower() for t in terms]}))

    return aliases_map

INDICATOR_ALIASES = _load_indicator_aliases()

# ---------- Embedder + Vector store ----------
embedder = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True}
)
vs = FAISS.load_local(FAISS_DIR, embedder, allow_dangerous_deserialization=True)

# ---------- Parsers ----------
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")  # years 1900-2099

def detect_year(q: str) -> Optional[int]:
    m = YEAR_RE.search(q)
    return int(m.group(0)) if m else None

def detect_indicator(q: str) -> Optional[str]:
    ql = q.lower()
    for canon, aliases in INDICATOR_ALIASES.items():
        if any(a in ql for a in aliases):
            return canon
    return None

def wants_growth(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in ["growth", "croissance", "annual %", "annual percent", "yoy", "year-on-year"])

def wants_rate(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in ["rate", "taux", "%"])

# ---------- Query ----------
def ask(q: str, k: int = 5) -> List[Dict[str, Any]]:
    q2 = f"query: {q.strip()}"
    year = detect_year(q)
    indicator = detect_indicator(q)
    need_growth = wants_growth(q)

    # Try filters: year+indicator → year → indicator → global
    tried_filters = []
    results = []

    def _search(flt: Optional[dict]):
        if flt:
            try:
                return vs.similarity_search_with_score(q2, k=k, filter=flt)
            except Exception:
                return []
        return vs.similarity_search_with_score(q2, k=k)

    for flt in (
        ({"year": year, "indicator": indicator} if year and indicator else None),
        ({"year": year} if year else None),
        ({"indicator": indicator} if indicator else None),
        None,
    ):
        tried_filters.append(flt)
        results = _search(flt)
        if results:
            break

    # Re-rank:
    # - Small bonus if indicator matches the detected canonical
    # - If the user asked for "growth", penalize non-growth indicators
    rescored = []
    for doc, score in results:
        meta = doc.metadata or {}
        meta_ind = (meta.get("indicator") or "").lower()

        bonus = 0.0
        if indicator and indicator.lower() in meta_ind:
            bonus += 0.05  # nudge correct indicator up

        if need_growth:
            # if user wants growth but the indicator doesn't mention 'growth', penalize
            if ("growth" not in meta_ind) and ("croissance" not in meta_ind):
                bonus -= 0.10  # lower is better, so negative bonus worsens the score

        rescored.append((doc, max(0.0, score - bonus)))

    rescored.sort(key=lambda x: x[1])

    out = []
    for doc, score in rescored[:k]:
        m = doc.metadata or {}
        out.append({
            "score": float(score),
            "snippet": doc.page_content[:200],
            "indicator": m.get("indicator"),
            "year": m.get("year"),
            "value": m.get("value"),
            "unit": m.get("unit"),
            "source": m.get("source"),
            "category": m.get("category"),
            "_debug_filter": tried_filters[0],
        })
    return out

if __name__ == "__main__":
    q = input("💬 Ask a question: ")
    for i, r in enumerate(ask(q), 1):
        print(f"{i}. [{r['score']:.4f}] {r['indicator']} {r['year']} → {r['value']} {r['unit']} | {r['source']}")
        print("   ", r["snippet"])
