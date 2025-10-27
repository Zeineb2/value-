# agentic/tools/vector_tools.py
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

# Prefer langchain.tools, fall back to langchain_core if needed
try:
    from langchain.tools import tool
except Exception:  # pragma: no cover
    from langchain_core.tools import tool  # type: ignore

from agentic.config import (
    FAISS_DIR as _FAISS_DIR,
    EMBED_MODEL as _EMBED_MODEL,
    EMBED_DEVICE as _EMBED_DEVICE,
    TOP_K as _CFG_TOP_K,
)

from langchain_community.vectorstores import FAISS
try:
    # Prefer the new package (avoids deprecation warnings)
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ───────────────────────────────────────────────────────────────────────────────
# Globals
# ───────────────────────────────────────────────────────────────────────────────
FAISS_DIR = os.path.abspath(_FAISS_DIR or "vectorization/faiss_index")
DEFAULT_TOP_K = int(_CFG_TOP_K) if str(_CFG_TOP_K).isdigit() else 6
MIN_K_FLOOR = 6  # never allow k < 6 to avoid starving recall

_embeddings = HuggingFaceEmbeddings(
    model_name=_EMBED_MODEL or "BAAI/bge-base-en-v1.5",
    model_kwargs={"device": _EMBED_DEVICE or "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

_vector: Optional[FAISS] = None

# Trusted/official sources (used only when recency is explicitly requested)
_TRUSTED = (
    "ins.tn",
    "bct.gov.tn",
    "imf.org",
    "data.imf.org",
    "worldbank.org",
    "documents.worldbank.org",
    "databank.worldbank.org",
    "oecd.org",
    "afdb.org",
)


def _is_trusted_url(u: Optional[str]) -> bool:
    if not u:
        return False
    lu = u.lower()
    return any(dom in lu for dom in _TRUSTED)


def _load_vector_if_needed() -> None:
    global _vector
    if _vector is None:
        if not os.path.exists(FAISS_DIR):
            raise RuntimeError(f"FAISS directory not found: {FAISS_DIR}")
        _vector = FAISS.load_local(
            FAISS_DIR, _embeddings, allow_dangerous_deserialization=True
        )


def reload_vector() -> bool:
    global _vector
    try:
        _vector = FAISS.load_local(
            FAISS_DIR, _embeddings, allow_dangerous_deserialization=True
        )
        return True
    except Exception:
        return False


# ───────────────────────────────────────────────────────────────────────────────
# Intent / recency gates
# ───────────────────────────────────────────────────────────────────────────────
_CORE_TERMS = [r"\bcore\b", r"\bunderlying\b", r"\bsous[-\s]?jacente\b"]
_POLICY_RATE_TERMS = [
    r"\bpolicy\s*(interest\s*)?rate\b",
    r"\btaux\s+directeur\b",
    r"\bmonetary\s+policy\s+rate\b",
]
_PPI_TERMS = [r"\bppi\b", r"\bproducer\s+price\b"]
_RESERVES_TERMS = [
    r"\b(fx|foreign)\s+reserves\b",
    r"\br[ée]serves\s+(de\s+devises|[aà]r)?\b",
    r"\breserves\b.*\b(fx|devises|foreign)\b",
]

_TERM_GROUPS = [
    ("core", _CORE_TERMS),
    ("policy_rate", _POLICY_RATE_TERMS),
    ("ppi", _PPI_TERMS),
    ("reserves", _RESERVES_TERMS),
]


def _compile_group(terms: List[str]) -> List[re.Pattern]:
    return [re.compile(t, flags=re.I) for t in terms]


_COMPILED = {name: _compile_group(ts) for name, ts in _TERM_GROUPS}


def _required_groups_for_query(q: str) -> List[str]:
    ql = (q or "").lower()
    required: List[str] = []
    if any(p.search(ql) for p in _COMPILED["core"]):
        required.append("core")
    if any(p.search(ql) for p in _COMPILED["policy_rate"]):
        required.append("policy_rate")
    if any(p.search(ql) for p in _COMPILED["ppi"]):
        required.append("ppi")
    if any(p.search(ql) for p in _COMPILED["reserves"]):
        required.append("reserves")
    return required


def _text_matches_group(text: str, group: str) -> bool:
    pats = _COMPILED.get(group, [])
    return any(p.search(text) for p in pats)


def _doc_text(doc: Document) -> str:
    parts = [doc.page_content or ""]
    md = doc.metadata or {}
    for key in ("indicator", "Indicator", "CanonicalIndicator", "Canonical Name", "raw_text"):
        v = md.get(key)
        if isinstance(v, str):
            parts.append(v)
    return " ".join(parts)


def _filter_docs_by_query_intent(query: str, docs: List[Document]) -> List[Document]:
    required = _required_groups_for_query(query)
    if not required:
        return docs
    filtered: List[Document] = []
    for d in docs:
        txt = _doc_text(d)
        if all(_text_matches_group(txt, g) for g in required):
            filtered.append(d)
    return filtered


# Recency: only when user explicitly asks for freshness AND no explicit year is given
_YEAR_PAT = re.compile(r"\b(19|20)\d{2}\b")
_FRESH_PAT = re.compile(
    r"\b(latest|current|most\s+recent|as\s+of|date\s+of\s+last\s+change|yoy|year[-\s]?on[-\s]?year)\b",
    re.I,
)


def _explicit_year(q: str) -> Optional[int]:
    m = _YEAR_PAT.search(q or "")
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _requires_recency(q: str) -> bool:
    """
    Return True only when the user asks for freshness AND the query does not
    specify a concrete year. This protects Scenario-1 (year-specific lookups).
    """
    if _explicit_year(q):
        return False
    return bool(_FRESH_PAT.search(q or ""))


def _is_recent(doc: Document, years: int = 2) -> bool:
    md = doc.metadata or {}
    y = md.get("year") or md.get("Year")
    try:
        y = int(y)
    except Exception:
        y = None
    if y is None:
        return True  # no year info → include
    return y >= (datetime.utcnow().year - years)


def _prefer_official_when_recent(query: str, docs: List[Document]) -> List[Document]:
    """When recency matters, prefer docs from official/trusted domains; fall back to all if none."""
    if not _requires_recency(query):
        return docs
    official = []
    for d in docs:
        md = d.metadata or {}
        url = md.get("url") or md.get("URL") or md.get("SourceURL")
        if _is_trusted_url(url):
            official.append(d)
    return official or docs


def _serialize_docs(docs: List[Document]) -> Dict[str, Any]:
    hits: List[Dict[str, Any]] = []
    for d in docs:
        md = d.metadata or {}
        meta = {
            "indicator": md.get("indicator") or md.get("CanonicalIndicator") or md.get("Indicator"),
            "year": md.get("year") or md.get("Year"),
            "value": md.get("value") or md.get("Value"),
            "unit": md.get("unit") or md.get("Unit"),
            "source": md.get("source") or md.get("Source"),
            "confidence": md.get("confidence") or md.get("Confidence"),
            "raw_text": md.get("raw_text") or md.get("RawText"),
            "url": md.get("url") or md.get("URL") or md.get("SourceURL"),
            "DateISO": md.get("DateISO") or md.get("dateISO") or md.get("dateIso"),
        }
        hits.append({"page_content": d.page_content, "metadata": meta})
    return {"hits": hits}


# ───────────────────────────────────────────────────────────────────────────────
# Utilities: coercion
# ───────────────────────────────────────────────────────────────────────────────
def _coerce_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _coerce_str(x: Any) -> str:
    if isinstance(x, str):
        return x
    if x is None:
        return ""
    return str(x)


# ───────────────────────────────────────────────────────────────────────────────
# Tool: vector_search
# ───────────────────────────────────────────────────────────────────────────────
@tool("vector_search", return_direct=False)
def vector_search(query: Any = "", k: Any = None) -> str:
    """
    Search local FAISS with semantic matching.

    Args (both optional to avoid Pydantic validation issues):
      - query: str (if empty → {"hits": [], "error": "query_missing"})
      - k: int top-k (minimum floor enforced to avoid k=1 starving recall)

    Behavior:
      - Apply flavor filters (core / policy rate / ppi / reserves).
      - Apply recency filter ONLY when user asks for freshness AND no explicit year is present.
      - When recency is active, prefer hits from official/trusted domains.
    """
    q = _coerce_str(query).strip()
    req_k = _coerce_int(k, DEFAULT_TOP_K)
    eff_k = max(req_k, MIN_K_FLOOR)

    if not q:
        return json.dumps({"hits": [], "error": "query_missing"}, ensure_ascii=False)

    try:
        _load_vector_if_needed()
    except Exception as e:
        # Surface a soft error so the agent can fall back to web ingest if needed
        return json.dumps({"hits": [], "error": f"faiss_load_failed: {e}"}, ensure_ascii=False)

    retriever = _vector.as_retriever(search_kwargs={"k": eff_k})
    docs: List[Document] = retriever.invoke(q)

    # Intent filter (e.g., require "core" if asked)
    filtered = _filter_docs_by_query_intent(q, docs)

    # Recency only when explicitly requested AND no explicit year in the query
    if _requires_recency(q):
        filtered_recent = [d for d in filtered if _is_recent(d, years=2)]
        if not filtered_recent:
            # Return empty to trigger Scenario-2 (fresh ingest) upstream
            return json.dumps({"hits": []}, ensure_ascii=False)
        # Prefer official sources when recency matters
        filtered = _prefer_official_when_recent(q, filtered_recent)

    final_docs = filtered if filtered else docs
    return json.dumps(_serialize_docs(final_docs), ensure_ascii=False)
