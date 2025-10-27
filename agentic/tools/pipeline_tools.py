# agentic/tools/pipeline_tools.py
from __future__ import annotations

import json
from typing import Any, Dict

# Prefer langchain.tools, fall back to langchain_core if needed
try:
    from langchain.tools import tool
except Exception:  # pragma: no cover
    from langchain_core.tools import tool  # type: ignore

# Local utilities
from scraping.core.parse_html import extract_text_from_html

# Optional (keep wrapped so missing deps don’t crash import)
def _try(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return {"ok": False, "error": str(e)}


@tool("extract_text_from_html", return_direct=False)
def parse_html_to_text() -> str:
    """
    Convert files under data/html/*.html → data/text/*.txt.
    Preserves lists and simple tables. Returns a JSON summary string.
    """
    stats = extract_text_from_html(html_dir="data/html", out_dir="data/text")
    out = {"stage": "extract_text_from_html", "stats": stats}
    return json.dumps(out, ensure_ascii=False)


@tool("fetch_links_from_serper", return_direct=False)
def fetch_links_from_serper() -> str:
    """
    (Compatibility shim) Previously used Serper to discover links.
    Current pipeline prefers question-scoped discovery in hybrid_ingest.
    Returns a small JSON note so the agent doesn't crash if it calls this.
    """
    return json.dumps({
        "stage": "fetch_links_from_serper",
        "note": "Discovery is handled by hybrid_ingest(question, allow_discovery=True)."
    })


@tool("scrape_and_download_xlsx", return_direct=False)
def scrape_and_download_xlsx() -> str:
    """
    (Compatibility shim) If you had an XLSX downloader, wrap it here.
    For now, this indicates hybrid_ingest does the targeted scraping.
    """
    return json.dumps({
        "stage": "scrape_and_download_xlsx",
        "note": "Use hybrid_ingest for scoped scraping/downloading."
    })


@tool("extract_indicators", return_direct=False)
def extract_indicators() -> str:
    """
    (Compatibility shim) Run your extractor to append to
    scraping/output/improved_structured_indicators.json
    if you need to call it directly (hybrid_ingest already handles this).
    """
    try:
        from scraping.core.extractor import extract_structured_indicators  # your existing extractor
        extract_structured_indicators()
        return json.dumps({"stage": "extract_indicators", "ok": True})
    except Exception as e:
        return json.dumps({"stage": "extract_indicators", "ok": False, "error": str(e)})


@tool("run_web_crawler", return_direct=False)
def run_web_crawler(seed_url: str = "", max_pages: int = 20) -> str:
    """
    (Optional) Placeholder for a deeper crawler (e.g., crawler4ai).
    Keep a harmless default so agent calls don't crash.
    """
    result: Dict[str, Any] = {
        "stage": "run_web_crawler",
        "seed_url": seed_url,
        "max_pages": max_pages,
        "note": "Crawler is not configured; hybrid_ingest does targeted scraping."
    }
    return json.dumps(result, ensure_ascii=False)
