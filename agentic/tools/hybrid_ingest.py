# agentic/tools/hybrid_ingest.py
from __future__ import annotations

import asyncio
import json
import os
import re
import hashlib
from typing import Any, Dict, List, Tuple

# LangChain tool decorator
try:
    from langchain.tools import tool
except Exception:  # pragma: no cover
    from langchain_core.tools import tool  # type: ignore

# 1) URL picking (question-targeted search)
from agentic.tools.url_pick import pick_verified_urls

# 2) Scrape ONLY those URLs
from scraping.scrapers.scrape_and_download import scrape_and_download

# 3) Parse HTML → text (HTML only; PDFs are handled later by the extractor)
from scraping.core.parse_html import extract_text_from_html

# 4) Extraction (existing extractor pipeline)
from scraping.core import extractor as _extractor
from scraping.core.extractor import extract_structured_indicators as _run_extraction

# 5) Vector hot-reload
from agentic.tools import vector_tools


# ───────────────────────────────────────────────────────────────────────────────
# Constants / Paths
# ───────────────────────────────────────────────────────────────────────────────

HTML_DIR = "data/html"
FILES_DIR = "data/files"
OUTPUT_DIR = os.path.join("scraping", "output")
MANIFEST_JSON = os.path.join(OUTPUT_DIR, "download_manifest.json")

# Caching controls (env-tunable)
FRESH_DEFAULT_HOURS = int(os.getenv("SCRAPE_FRESH_HOURS", "72"))  # 3 days
SCRAPE_FORCE_DEFAULT = os.getenv("SCRAPE_FORCE", "0").strip().lower() in {"1", "true", "yes"}

# If a query implies “latest/current/YoY”, we tighten freshness
_RECENT_PAT = re.compile(
    r"\b(latest|current|most\s+recent|as\s+of|date\s+of\s+last|yoy|year[-\s]?on[-\s]?year)\b",
    re.I,
)


# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────

def _taxonomy_path() -> str:
    """Single source-of-truth taxonomy file (singular)."""
    return os.path.abspath("economic_indicator.json")


def _ensure_indicator_taxonomy() -> Tuple[str, bool]:
    """
    Ensure economic_indicator.json exists and point extractor to it.
    """
    path = _taxonomy_path()
    existed = os.path.exists(path)
    if not existed:
        minimal = [
            {"Canonical Name": "Inflation (headline, % YoY)", "Aliases": ["inflation", "cpi", "ipc"]},
            {"Canonical Name": "Policy interest rate (%)", "Aliases": ["policy rate", "taux directeur", "monetary policy rate"]},
        ]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(minimal, f, ensure_ascii=False, indent=2)
    # Tell the extractor where to read the taxonomy from
    _extractor.INDICATOR_JSON = path
    return path, existed


def _maybe_extend_from_question(question: str) -> int:
    """
    VERY conservative taxonomy extension: if question clearly mentions an
    indicator phrase absent from the file, append it as its own entry.
    """
    path = _taxonomy_path()
    try:
        from scraping.utils.indicator_matcher import load_indicators as _load_inds
        items = _load_inds(path)
    except Exception:
        try:
            items = json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            items = []

    q = (question or "").strip().lower()
    if not q:
        return 0

    seen = set()
    for it in items:
        cn = (it.get("Canonical Name") or "").strip().lower()
        if cn:
            seen.add(cn)
        for a in it.get("Aliases") or []:
            if isinstance(a, str):
                seen.add(a.strip().lower())

    # candidate phrases (short list + question fallback)
    CANDS = [
        "retail sales index", "indice du commerce de détail", "policy interest rate",
        "producer price index", "ppi", "industrial production index", "unemployment rate",
        "current account balance", "money supply m2", "fx reserves", "core inflation",
    ]
    cand = None
    for t in CANDS:
        if t in q:
            cand = t
            break
    if cand is None and 5 <= len(q) <= 140:
        cand = q

    if not cand or cand in seen:
        return 0

    items.append({"Canonical Name": cand, "Aliases": [cand]})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return 1


def _upsert_latest_py() -> int:
    """
    Incrementally add new rows into FAISS from scraping/output/improved_structured_indicators.json
    """
    try:
        from vectorization.upsert_embeddings import upsert_latest
        return int(upsert_latest())
    except Exception:
        return -1


def _try_reload_agent_vectorstore() -> bool:
    """Hot-reload the in-process FAISS retriever used by vector_search."""
    try:
        return bool(vector_tools.reload_vector())
    except Exception:
        return False


def _safe_name(url: str) -> str:
    """
    Use the same filename strategy as the scraper to deterministically
    map a URL to its saved file base name. This lets us build a manifest
    even though downloads are concurrent.
    """
    u = re.sub(r"[^a-zA-Z0-9._/-]+", "_", url)
    u = u.strip("/").replace("://", "_").replace("/", "_")
    # Keep short and unique – use the same 10-char SHA1 suffix
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    if len(u) > 80:
        u = u[:80]
    return f"{u}_{h}"


def _build_and_save_manifest(urls: List[str]) -> Dict[str, str]:
    """
    Build { saved_filename: original_url } for outputs produced by the scraper.
    We support html and common doc types so the extractor can attach SourceURL.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest: Dict[str, str] = {}

    exts = (".html", ".pdf", ".xlsx", ".xls", ".csv", ".json", ".txt")
    for url in urls:
        base = _safe_name(url)
        for ext in exts:
            name = f"{base}{ext}"
            path = os.path.join(HTML_DIR if ext == ".html" else FILES_DIR, name)
            if os.path.exists(path):
                manifest[name] = url

    # Persist manifest for the extractor (it reads scraping/output/download_manifest.json)
    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def _coerce_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _choose_fresh_hours(question: str) -> int:
    """Tighten freshness for 'latest/current/YoY' questions."""
    if _RECENT_PAT.search(question or ""):
        return min(FRESH_DEFAULT_HOURS, 24)  # 24h for time-sensitive asks
    return FRESH_DEFAULT_HOURS


# ───────────────────────────────────────────────────────────────────────────────
# Tool
# ───────────────────────────────────────────────────────────────────────────────

@tool("hybrid_ingest", return_direct=False)
def hybrid_ingest(question: str = "", allow_discovery: Any = False) -> str:
    """
    Question-scoped refresh & ingest.

    Parameters (both optional to avoid Pydantic validation errors):
      - question: str        → the user question (indicator to look for)
      - allow_discovery: any → truthy/falsey; will be coerced to bool

    Steps:
      1) Search & pick URLs for THIS question (pick_verified_urls)
      2) Scrape ONLY those URLs (cache-aware)
      3) Parse HTML → text
      4) Ensure taxonomy + (optional) extend from question
      5) Run extractor → improved_structured_indicators.{json,csv}
      6) Upsert only new rows into FAISS
      7) Hot-reload the in-process retriever

    Returns: short report with counts and URLs (never raises).
    """
    lines: List[str] = []

    try:
        # Resolve inputs safely
        allow_flag = _coerce_bool(allow_discovery, default=False)
        if not isinstance(question, str):
            question = str(question or "")

        if not question.strip():
            lines.append("INPUT_ERR=question_missing")
            return "\n".join(lines)

        # 1) Targeted URLs
        urls: List[str] = pick_verified_urls(
            question=question,
            top_k=3,
            allow_discovery=allow_flag,
            write_links=True,
        )
        # Normalize to https to minimize http/https duplicates (belt-and-suspenders;
        # url_pick already normalizes but keep this for robustness)
        urls = [re.sub(r"^http://", "https://", u.strip()) for u in urls if u.strip()]

        lines.append(f"PICKED_URLS={len(urls)}")
        lines.append("URLS:")
        lines.extend(urls)

        # 2) Scrape ONLY those URLs, cache-aware (no forced re-downloads unless env says so)
        fresh_hours = _choose_fresh_hours(question)
        force_fetch = SCRAPE_FORCE_DEFAULT  # can override via env SCRAPE_FORCE=1

        processed: List[str] = []
        try:
            processed = asyncio.run(
                scrape_and_download(urls=urls, fresh_hours=fresh_hours, force=force_fetch)
            )
        except RuntimeError:
            # Fallback if an event loop is already running
            loop = asyncio.get_event_loop()
            processed = loop.run_until_complete(
                scrape_and_download(urls=urls, fresh_hours=fresh_hours, force=force_fetch)
            )
        lines.append(f"SCRAPED={len(processed)}")
        lines.append(f"FRESH_HOURS={fresh_hours}")
        lines.append(f"FORCE={'1' if force_fetch else '0'}")

        # 3) Parse HTML → text (sync HTML to text folder; PDFs handled by extractor)
        try:
            stats = extract_text_from_html(html_dir=HTML_DIR, out_dir="data/text")
            lines.append(f"SYNCHRONIZED_HTML={stats.get('processed', 0)}")
            lines.append(f"SYNCHRONIZED_FILES={stats.get('written', 0)}")
            lines.append("PARSE_OK=1")
        except Exception as e:
            lines.append(f"PARSE_ERR={e}")

        # Build and save manifest so extractor can attach SourceURL to rows
        try:
            manifest = _build_and_save_manifest(urls)
            lines.append(f"MANIFEST_ENTRIES={len(manifest)}")
        except Exception as e:
            lines.append(f"MANIFEST_ERR={e}")

        # 4) Taxonomy
        tax_path, existed = _ensure_indicator_taxonomy()
        lines.append(f"TAXONOMY={'existing' if existed else 'created'}:{os.path.relpath(tax_path)}")
        try:
            added = _maybe_extend_from_question(question)
            if added:
                lines.append(f"TAXONOMY_UPDATED_FROM_QUESTION={added}")
        except Exception as e:
            lines.append(f"TAXONOMY_UPDATE_ERR={e}")

        # 5) Extraction
        try:
            _run_extraction()
            lines.append("EXTRACT_OK=1")
        except FileNotFoundError as e:
            lines.append(f"EXTRACT_ERR=FileNotFound:{e}")
        except Exception as e:
            lines.append(f"EXTRACT_ERR={e}")

        # 6) Upsert new rows to FAISS
        upserted = _upsert_latest_py()
        lines.append(f"UPSERTED={max(upserted, 0)}")

        # 7) Hot reload the agent’s in-memory retriever
        reloaded = _try_reload_agent_vectorstore()
        lines.append(f"RELOADED={'yes' if reloaded else 'no'}")

        return "\n".join(lines)

    except Exception as e:
        lines.append(f"HYBRID_INGEST_FATAL={e}")
        return "\n".join(lines)
