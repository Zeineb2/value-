# vectorization/upsert_embeddings.py
from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from langchain_community.vectorstores import FAISS

try:
    # Preferred new package (avoids deprecation)
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:  # pragma: no cover
    from langchain_community.embeddings import HuggingFaceEmbeddings


# ───────────────────────────────────────────────────────────────────────────────
# Paths & Config
# ───────────────────────────────────────────────────────────────────────────────

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

IMPROVED_JSON = PROJECT_ROOT / "scraping" / "output" / "improved_structured_indicators.json"
FAISS_DIR = Path(os.getenv("FAISS_DIR", str(PROJECT_ROOT / "vectorization" / "faiss_index")))
STATE_PATH = FAISS_DIR / "upsert_state.json"

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")  # "cuda" if available


# ───────────────────────────────────────────────────────────────────────────────
# IO helpers
# ───────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"seen": []}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"seen": []}


def _save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ───────────────────────────────────────────────────────────────────────────────
# Normalization & Keys
# ───────────────────────────────────────────────────────────────────────────────

def _pick_source_and_url(row: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """
    Prefer SourceURL if present; also keep a human-readable source string.
    Returns (source_for_display, url_for_metadata)
    """
    url = row.get("SourceURL") or row.get("URL") or row.get("url")
    source_display = row.get("Source") or row.get("SourceName") or (url or "unknown")
    return str(source_display), (str(url) if url else None)


def _stable_key(row: Dict[str, Any]) -> str:
    """
    Create a robust key so we don't re-upsert duplicates.
    Use CanonicalIndicator/Indicator + DateISO/Year + Value + Unit + SourceURL/Source
    """
    indicator = (row.get("CanonicalIndicator") or row.get("Indicator") or "").strip().lower()
    date = (row.get("DateISO") or row.get("Year") or "").__str__().strip().lower()
    value = (row.get("Value")).__str__() if row.get("Value") is not None else ""
    unit = (row.get("Unit") or "").strip().lower()
    src_url = (row.get("SourceURL") or row.get("URL") or row.get("Source") or "").strip().lower()
    payload = f"{indicator}|{date}|{value}|{unit}|{src_url}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _row_to_text_meta(row: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Build the text that goes into FAISS + metadata dict used by downstream filters.
    Prefer DateISO, include URL, and keep raw text for context.
    """
    indicator = row.get("CanonicalIndicator") or row.get("Indicator") or ""
    date_iso = row.get("DateISO")
    year = row.get("Year")
    when = date_iso or year

    value = row.get("Value")
    unit = row.get("Unit")
    source_display, url = _pick_source_and_url(row)
    confidence = row.get("Confidence") or row.get("confidence")
    raw = row.get("RawText") or row.get("raw_text") or ""

    # human-readable page content for retrieval
    parts = [f"Indicator: {indicator}"]
    if when is not None:
        parts.append(f"Date: {when}")
    if value is not None:
        parts.append(f"Value: {value}{(' ' + unit) if unit else ''}")
    if source_display:
        parts.append(f"Source: {source_display}")
    if url:
        parts.append(f"URL: {url}")
    if raw:
        parts.append(f"Raw: {raw}")

    text = " | ".join(parts)

    metadata = {
        "indicator": indicator,
        "year": year,
        "value": value,
        "unit": unit,
        "source": source_display,
        "url": url,                 # <— important for downstream domain filters
        "confidence": confidence,
        "raw_text": raw,
        "DateISO": date_iso,        # keep exact date if present
    }
    return text, metadata


def _load_or_create_faiss(embeddings: HuggingFaceEmbeddings) -> FAISS:
    idx_file = FAISS_DIR / "index.faiss"
    pkl_file = FAISS_DIR / "index.pkl"
    if idx_file.exists() and pkl_file.exists():
        return FAISS.load_local(str(FAISS_DIR), embeddings=embeddings, allow_dangerous_deserialization=True)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    return FAISS.from_texts(texts=[], embedding=embeddings, metadatas=[])


# ───────────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────────

def upsert_latest(max_new: int | None = None) -> int:
    """
    Incrementally upsert newly extracted rows from
    scraping/output/improved_structured_indicators.json into FAISS.

    Returns the number of new rows added.
    """
    rows = _load_json(IMPROVED_JSON)
    if not rows:
        return 0

    state = _load_state(STATE_PATH)
    seen = set(state.get("seen", []))

    # Prepare new pairs
    new_pairs: List[Tuple[str, Dict[str, Any]]] = []
    newly_seen: List[str] = []

    for r in rows:
        key = _stable_key(r)
        if key in seen:
            continue

        # Optional guardrails: skip incomplete rows
        if not (r.get("CanonicalIndicator") or r.get("Indicator")):
            continue
        # Require at least value or a year/date to be useful
        if r.get("Value") is None and r.get("Year") is None and r.get("DateISO") is None:
            continue

        text, meta = _row_to_text_meta(r)
        new_pairs.append((text, meta))
        newly_seen.append(key)

    if not new_pairs:
        return 0

    if max_new is not None and max_new > 0:
        new_pairs = new_pairs[:max_new]
        newly_seen = newly_seen[:max_new]

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    vs = _load_or_create_faiss(embeddings)
    vs.add_texts(texts=[t for t, _ in new_pairs], metadatas=[m for _, m in new_pairs])
    vs.save_local(str(FAISS_DIR))

    state["seen"] = list(seen.union(newly_seen))
    _save_state(STATE_PATH, state)

    return len(new_pairs)


if __name__ == "__main__":
    added = upsert_latest()
    print(f"✅ Upserted {added} new row(s) into FAISS at {FAISS_DIR}")
