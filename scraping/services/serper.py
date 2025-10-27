# scraping/services/serper.py
from __future__ import annotations
import os
import re
import time
import json
import requests
import pandas as pd
from typing import List, Dict, Iterable

# --- Config / constants ------------------------------------------------------

SERPER_API_KEY = os.getenv("SERPER_API_KEY") or "afa7eef885b486212e58ef2eaf29d08efbefc683"
SERPER_URL = "https://google.serper.dev/search"

# Add ONTT (tourism office) since you often need arrivals/overnights
TRUSTED_DOMAINS = [
    "ins.tn", "bct.gov.tn", "ontt.tn",
    "worldbank.org", "imf.org", "afdb.org", "oecd.org",
    "data.imf.org", "databank.worldbank.org", "tradingeconomics.com",
]

# Durable sweep seed queries (kept as-is)
QUERIES = [
    "Tunisia inflation site:ins.tn",
    "Tunisia GDP site:bct.gov.tn",
    "Tunisia economic indicators site:worldbank.org",
    "Tunisia macroeconomic site:afdb.org",
    "Tunisia report site:oecd.org",
    "Tunisia economic outlook site:imf.org",
    "Tunisia trade statistics site:tradingeconomics.com",
    "Tunisia site:databank.worldbank.org",
    "Tunisia quarterly bulletin site:bct.gov.tn",
]

RELEVANT_KEYWORDS = [
    "gdp", "gross domestic product", "inflation", "cpi", "consumer price index",
    "unemployment", "employment", "labor force", "growth rate", "fiscal deficit",
    "public debt", "government revenue", "budget", "subsidies",
    "agriculture", "industry", "manufacturing", "services sector",
    "construction", "tourism", "energy", "mining", "transport",
    "export", "import", "trade balance", "current account",
    "foreign reserves", "exchange rate", "fdi", "remittances",
    "price index", "producer price index", "interest rate",
    "monetary policy", "credit", "banking", "money supply",
    "financial indicators", "stock market", "capital market",
    "poverty", "inequality", "household consumption",
    "education spending", "health expenditure", "demographics",
    "economic outlook", "quarterly bulletin", "statistical report",
    "dataset", "dashboard", "tableau", "time series",
    "macro framework", "indicator"
]

IRRELEVANT_KEYWORDS = [
    "mexico", "turkey", "romania", "nigeria", "canada", "brazil", "latvia"
]

# Where we persist links (unchanged)
JSON_PATH = "serper_links.json"
EXCEL_PATH = "serper_links.xlsx"


# --- Small utils -------------------------------------------------------------

def _headers() -> Dict[str, str]:
    return {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

def _norm_url(u: str) -> str:
    return (u or "").strip()

def _dedupe(seq: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        x = _norm_url(x)
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def domain_trust_score(url: str) -> int:
    """Rough weight by domain trust (used by some scoring flows)."""
    u = (url or "").lower()
    for d in TRUSTED_DOMAINS:
        if d in u:
            return 2
    return 0


# --- Public: Serper search (question-aware) ----------------------------------

def search(query: str, num: int = 10, gl: str = "tn") -> List[Dict]:
    """
    Minimal Serper wrapper returning [{'link','title','snippet'}, ...].
    Safe to import from url_pick.py as serper_search.
    """
    payload = {"q": query, "gl": gl, "num": max(1, min(num, 20)), "type": "search"}
    try:
        r = requests.post(SERPER_URL, headers=_headers(), json=payload, timeout=30)
        r.raise_for_status()
        organic = r.json().get("organic", []) or []
        out = []
        for item in organic:
            link = item.get("link") or item.get("url") or ""
            if not link:
                continue
            out.append({
                "link": link,
                "title": item.get("title") or "",
                "snippet": item.get("snippet") or item.get("description") or "",
            })
        return out
    except Exception:
        return []


def _extract_year_hints(text: str) -> List[str]:
    """
    Pull common year/period hints out of a free-form question:
      - 4-digit years
      - H1/H2 / S1/S2 / 'first half' / 'second half' / 'June' etc. (simple tags)
    """
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    tags = []
    if re.search(r"\b(H1|S1|first half)\b", text, re.I): tags.append("H1")
    if re.search(r"\b(H2|S2|second half)\b", text, re.I): tags.append("H2")
    if re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|June|July|August|September|October|November|December)\b", text, re.I):
        tags.append("monthly")
    return list(dict.fromkeys(years + tags))  # dedupe, keep order


def build_queries_from_question(
    question: str,
    include_domains: bool = True,
    add_time_hints: bool = True,
) -> List[str]:
    """
    Turn a user question into 3–8 focused queries for Serper.
    - Adds Tunisia anchor
    - Adds time hints (H1 2025, 2019, June…)
    - Optionally pins to trusted domains via site: filters
    """
    base = question.strip()
    if "tunisia" not in base.lower():
        base = f"Tunisia {base}"

    hints = _extract_year_hints(base) if add_time_hints else []
    hint_str = " ".join(hints) if hints else ""

    q0 = base
    q1 = f"{base} {hint_str}".strip() if hint_str else base

    queries = [q0, q1]

    if include_domains:
        for d in TRUSTED_DOMAINS:
            queries.append(f"{q1} site:{d}")

    # keep it short & unique
    uniq = []
    for q in queries:
        if q not in uniq:
            uniq.append(q)
    return uniq[:8]


def search_new_links_for_question(
    question: str,
    per_query: int = 5,
    max_total: int = 12,
    allow_discovery: bool = False,
) -> List[str]:
    """
    Question-aware discovery.
    - Builds queries from the user question
    - Searches Serper
    - Returns up to `max_total` fresh URLs (optionally filtered to trusted)
    """
    queries = build_queries_from_question(
        question,
        include_domains=True,   # always bias to trusted first
        add_time_hints=True
    )

    urls: List[str] = []
    for q in queries:
        for r in search(q, num=per_query):
            u = r.get("link")
            if u:
                urls.append(u)
        time.sleep(0.6)  # be polite

    urls = _dedupe(urls)

    if not allow_discovery:
        # keep only trusted
        urls = [u for u in urls if any(d in u.lower() for d in TRUSTED_DOMAINS)]

    # light re-rank by trust + doc-type hints
    def _score(u: str) -> int:
        s = domain_trust_score(u)
        if any(u.lower().endswith(ext) for ext in (".pdf", ".xlsx", ".csv", ".json")):
            s += 2
        if "bulletin" in u.lower() or "press" in u.lower() or "stat" in u.lower():
            s += 1
        return s

    urls = sorted(urls, key=_score, reverse=True)[:max_total]
    return urls


# --- Persistence helpers (unchanged behavior outwardly) ----------------------

def score_link(link: str) -> int:
    link = link.lower()
    score = 0
    if "tunisia" not in link:
        return 0
    if any(bad in link for bad in IRRELEVANT_KEYWORDS):
        return 0
    if any(domain in link for domain in TRUSTED_DOMAINS):
        score += 2
    if any(link.endswith(ext) for ext in (".pdf", ".xlsx", ".csv", ".json")):
        score += 2
    if any(kw in link for kw in RELEVANT_KEYWORDS):
        score += 1
    return score

def load_existing_links() -> list:
    if not os.path.exists(JSON_PATH):
        return []
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return [str(u).strip() for u in data if str(u).strip()]
    except Exception:
        pass
    return []

def merge_links(existing: list, new_links: list) -> list:
    seen = set(existing)
    merged = existing[:]
    for url in new_links:
        if url not in seen:
            merged.append(url)
            seen.add(url)
    return merged

def save_links(urls: List[str]) -> int:
    """
    Merge URLs into serper_links.json and serper_links.xlsx.
    Returns the total number of URLs saved after merge.
    """
    urls = _dedupe(urls)
    # filter obvious junk
    scored = [(u, score_link(u)) for u in urls]
    filtered_sorted = [u for (u, s) in sorted(scored, key=lambda x: x[1], reverse=True) if s > 0]

    existing = load_existing_links()
    merged = merge_links(existing, filtered_sorted)

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    df = pd.DataFrame(merged, columns=["URL"])
    df.index += 1
    df.to_excel(EXCEL_PATH, index_label="Link No")

    return len(merged)


# --- Durable sweep (your original function) ----------------------------------

def fetch_links():
    """
    DURABLE SWEEP:
    Runs your static QUERIES against Serper, merges results,
    writes serper_links.json/.xlsx (keeps existing + adds new unique).
    """
    print("🔍 Searching Serper for economic indicator links...")

    all_links = set()

    for query in QUERIES:
        print(f"\n📤 Querying: {query!r}")
        try:
            results = search(query, num=10)  # uses SERPER_API_KEY / Serper API
            query_links = [r.get("link", "") for r in results if r.get("link")]
            print(f"✅ {len(query_links)} links retrieved.")
            all_links.update(query_links)
        except Exception as e:
            print(f"❌ Failed query: {query} → {e}")
        time.sleep(1.0)

    # Save raw unique links before filtering (optional, helpful for debugging)
    with open("serper_links_before.json", "w", encoding="utf-8") as f:
        json.dump(sorted(all_links), f, indent=2)
    print(f"\n📝 Saved {len(all_links)} unfiltered unique links to serper_links_before.json")

    total_after_merge = save_links(sorted(all_links))
    print(f"📁 Saved {total_after_merge} total links to {JSON_PATH} and {EXCEL_PATH}")


# --- CLI ---------------------------------------------------------------------

if __name__ == "__main__":
    # Default behavior: run the durable sweep
    fetch_links()
