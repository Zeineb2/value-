# agentic/tools/url_pick.py
from __future__ import annotations

import os
import re
import json
import time
import socket
import urllib.parse
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse

import requests

# ========= Config =========
try:
    from agentic.config import SERPER_API_KEY as CFG_SERPER_API_KEY, SERPER_COUNTRY as CFG_COUNTRY
except Exception:
    CFG_SERPER_API_KEY, CFG_COUNTRY = None, None

# Default key if env/config missing (override via env in prod)
_DEFAULT_SERPER_KEY = "afa7eef885b486212e58ef2eaf29d08efbefc683"

SERPER_API_KEY = (CFG_SERPER_API_KEY or os.getenv("SERPER_API_KEY") or _DEFAULT_SERPER_KEY).strip()
SERPER_COUNTRY = (CFG_COUNTRY or os.getenv("SERPER_COUNTRY") or "tn").strip().lower()
SERPER_URL = "https://google.serper.dev/search"

# ========= Domain policy =========
TRUSTED_DOMAINS = [
    # Tunisia official + IFIs
    "ins.tn", "bct.gov.tn",
    "imf.org", "data.imf.org",
    "worldbank.org", "documents.worldbank.org", "databank.worldbank.org",
    "oecd.org", "afdb.org",
]

# Aggregators (we prefer to avoid; especially when discovery=False)
AGGREGATOR_DOMAINS = [
    "tradingeconomics.com", "ceicdata.com", "macrotrends.net", "statista.com",
    "countryeconomy.com", "cbrates.com", "knoema.com", "focus-economics.com",
]

# Known dead/irrelevant endpoints that frequently 404 or are not indicator pages
BAD_PATH_FRAGMENTS = [
    "/actualites", "/bct/siteprod/actualites.jsp", "/regular.aspx?key=61545865",
    "/news", "/press-release", "/press-releases",
]

# Strong hints that a URL is a publication or statistics page
GOOD_PATH_HINTS = (
    "stat", "statistique", "statistics", "publication", "bulletin",
    "communique", "communiqué", "press", "rapport", "monthly", "mensuel", "pdf",
    "xlsx", "xls", "csv", "indice", "index", "cpi", "ppi", "inflation",
    "rate", "interest", "policy", "m2", "reserve", "production",
    "unemployment", "commerce-de-detail", "commerce-de-détail",
    "commerce de detail", "commerce de détail",
    "chiffre d'affaires", "chiffre d’affaires", "dataset", "table", "figure",
)

# Negative content to exclude for ICA queries
NEGATIVE_KWS = [
    "commerce extérieur", "commerce exterieur",
    "retail banking", "banque de détail", "banking sector",
]

# ========= Persistence for auditing =========
LINKS_JSON = Path("scraping/services/serper_links.json")
LINKS_XLSX = Path("scraping/services/serper_links.xlsx")


# ========= Utilities =========
def _ua() -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        )
    }


def _normalize_url(u: str) -> str:
    p = urlparse(u)
    scheme = "https"  # force https to avoid http/https duplicates
    netloc = (p.netloc or "").lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = re.sub(r"/+", "/", p.path).rstrip("/")
    return f"{scheme}://{netloc}{path}"



def _domain_of(u: str) -> str:
    try:
        p = urlparse(u if u.startswith("http") else f"https://{u}")
        host = (p.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _is_trusted(u: str) -> bool:
    d = _domain_of(u)
    return any(d.endswith(t) for t in TRUSTED_DOMAINS)


def _is_official(u: str) -> bool:
    """Tunisia official sites we want to lock to when discovery is OFF."""
    d = _domain_of(u)
    return d.endswith("ins.tn") or d.endswith("bct.gov.tn")


def _is_aggregator(u: str) -> bool:
    d = _domain_of(u)
    return any(d.endswith(t) for t in AGGREGATOR_DOMAINS)


def _is_probably_dead(u: str) -> bool:
    lu = u.lower()
    return any(b in lu for b in BAD_PATH_FRAGMENTS)


def _has_any(s: str, terms) -> bool:
    s = (s or "").lower()
    return any(t in s for t in terms)


def _fast_head_ok(u: str, timeout: float = 8.0) -> bool:
    """
    Light HEAD ping to weed out obvious non-2xx BEFORE scraping (keeps logs clean).
    - 403 is considered NOT OK (don’t fallback to GET for 403)
    - 405 (HEAD not allowed): fallback to a tiny streamed GET
    """
    try:
        r = requests.head(u, headers=_ua(), allow_redirects=True, timeout=timeout)
        if 200 <= r.status_code < 300:
            return True
        if r.status_code == 405:
            g = requests.get(u, headers=_ua(), stream=True, timeout=timeout)
            ok = 200 <= g.status_code < 300
            try:
                g.close()
            except Exception:
                pass
            return ok
        return False
    except (requests.Timeout, requests.ConnectionError, socket.timeout):
        return False
    except Exception:
        return False


# ========= ICA trigger =========
_RETAIL_TRIGGERS = [
    "ica", "retail", "commerce de detail", "commerce de détail",
    "chiffre d'affaires", "chiffre d’affaires"
]

def _looks_like_ica(question: str) -> bool:
    q = (question or "").lower()
    return any(t in q for t in _RETAIL_TRIGGERS)


# ========= Query expansion =========
def _expanded_queries(question: str, allow_discovery: bool) -> List[str]:
    """
    Build Google queries:
    - When discovery is OFF → bias hard to official via site: filters
    - Add structured filetypes for higher-precision hits (pdf/xlsx)
    - For ICA queries, prepend strong INS-only queries regardless of discovery flag
    """
    q = (question or "").strip()
    retail_lock = _looks_like_ica(q)

    try:
        from scraping.core.taxonomy_utils import build_search_queries as _build  # optional
        base = _build(q, prefer_official=(not allow_discovery or retail_lock))
    except Exception:
        base = _heuristic_queries(q, allow_discovery and not retail_lock)

    # Prepend strong INS-only queries for ICA
    if retail_lock:
        ins_only = [
            'site:ins.tn "commerce de détail" "indice du chiffre d’affaires"',
            'site:ins.tn "commerce de détail" filetype:pdf',
            'site:ins.tn "indice du chiffre d’affaires" filetype:pdf',
            'site:ins.tn ICA filetype:pdf',
            'site:ins.tn ICA filetype:xlsx',
            'site:ins.tn "indice du chiffre d’affaires" 2025',
            'site:ins.tn "indice du chiffre d’affaires" 2024',
        ]
        base = ins_only + base

    extra = []
    for s in base:
        extra.append(s + " filetype:pdf")
        extra.append(s + " filetype:xlsx")

    # Dedup while keeping order
    seen = set()
    uniq: List[str] = []
    for s in base + extra:
        if s not in seen:
            seen.add(s)
            uniq.append(s)

    return uniq[:14]


def _heuristic_queries(question: str, allow_broad: bool) -> List[str]:
    q = (question or "").lower()

    # pick likely indicator keywords from the question (EN + FR)
    bits: List[str] = []
    if re.search(r"\b(ica|retail|commerce\s+de\s+d[ée]tail|chiffre\s+d[’']?affaires)\b", q, re.I):
        bits += ["retail sales index", "indice du commerce de détail", "indice du chiffre d'affaires"]
    if ("policy" in q and "rate" in q) or ("taux directeur" in q) or ("monetary policy" in q):
        bits += ["policy interest rate", "taux directeur", "monetary policy rate"]
    if "core inflation" in q or "sous-jacente" in q:
        bits += ["core inflation", "inflation sous-jacente", "core cpi"]
    if "inflation" in q and not bits:
        bits += ["inflation", "cpi", "indice des prix"]
    if "ppi" in q or "producer price" in q:
        bits += ["producer price index", "ppi", "indice des prix à la production"]
    if "m2" in q or "money supply" in q:
        bits += ["M2 money supply", "agrégats monétaires M2"]
    if "reserves" in q or "devises" in q:
        bits += ["foreign exchange reserves", "réserves en devises"]
    if "industrial" in q or "ipi" in q:
        bits += ["industrial production index", "indice de production industrielle"]
    if "unemployment" in q or "chômage" in q:
        bits += ["unemployment rate", "taux de chômage"]

    if not bits:
        bits = [question]

    queries: List[str] = []
    for kb in bits:
        base = f"{kb} Tunisia"
        if not allow_broad:
            # strong official bias when discovery is off
            for d in ("ins.tn", "bct.gov.tn", "data.imf.org"):
                queries.append(f"{base} site:{d}")
        else:
            queries.append(base)
        # helpful recency & YoY cues
        queries.append(base + " YoY")
        queries.append(base + " 2024")
        queries.append(base + " 2025")
    return queries


# ========= Search engines =========
def _serper_search(q: str, num: int = 10) -> List[Tuple[str, str]]:
    if not SERPER_API_KEY:
        return []
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": q, "gl": SERPER_COUNTRY or "tn", "num": num, "type": "search"}
    try:
        r = requests.post(SERPER_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        out: List[Tuple[str, str]] = []
        for it in (data.get("organic") or []):
            u = (it.get("link") or "").strip()
            sn = (it.get("snippet") or it.get("title") or "").strip()
            if u.startswith("http"):
                out.append((u, sn))
        return out
    except Exception:
        return []


def _ddg_fallback(q: str, num: int = 10) -> List[Tuple[str, str]]:
    try:
        url = "https://duckduckgo.com/html/?q=" + urllib.parse.quote(q)
        html = requests.get(url, headers=_ua(), timeout=25).text
        links = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"', html, flags=re.I)
        out: List[Tuple[str, str]] = []
        for href in links[:num]:
            href = urllib.parse.unquote(href)
            if href.startswith("http"):
                out.append((href, ""))
        return out
    except Exception:
        return []


def _search_candidates(question: str, allow_discovery: bool, k: int = 10) -> List[Tuple[str, str]]:
    """
    Build the pool of (url, snippet) candidates.
    When allow_discovery=False we emphasize official site: queries.
    For ICA queries, we restrict to INS and drop negative keywords.
    """
    queries = _expanded_queries(question, allow_discovery)
    seen: set[str] = set()
    out: List[Tuple[str, str]] = []

    retail_lock = _looks_like_ica(question)

    for q in queries:
        rows = _serper_search(q, num=k) or _ddg_fallback(q, num=k)
        for u, s in rows:
            nu = _normalize_url(u)
            lu = nu.lower()

            # Skip clearly bad paths
            if _is_probably_dead(lu):
                continue

            # If discovery is OFF, skip known aggregators early
            if not allow_discovery and _is_aggregator(lu):
                continue

            # ICA strictness: INS only + no negative keywords
            if retail_lock:
                if _domain_of(nu) != "ins.tn":
                    continue
                if _has_any(nu, NEGATIVE_KWS) or _has_any(s, NEGATIVE_KWS):
                    continue

            if nu not in seen:
                seen.add(nu)
                out.append((nu, s))
        time.sleep(0.20)  # be polite and avoid rate limits
    return out


# ========= Scoring =========
_RECENCY_URL = re.compile(r"\b(202[3-5]|q[1-4]|2024|2025|monthly|mensuel|yoy|year[-\s]on[-\s]year)\b", re.I)

def _score(url: str, snippet: str, question: str, allow_discovery: bool) -> float:
    u = url.lower()
    sn = (snippet or "").lower()
    ql = question.lower()

    s = 0.0

    # Strong boost for official domains when discovery is OFF
    if not allow_discovery and _is_official(u):
        s += 4.0
    # General trust boost (even when discovery ON)
    if _is_trusted(u):
        s += 1.0

    # Prefer publication/statistics paths
    if any(h in u for h in GOOD_PATH_HINTS):
        s += 2.0

    # Snippet clues for recency / tables / YoY
    if _RECENCY_URL.search(u) or _RECENCY_URL.search(sn):
        s += 1.2
    if "%" in sn:
        s += 0.35
    if re.search(r"\b\d{3,}(?:[.,]\d{3})+\b", sn):
        s += 0.6

    # Recency/age from URL path if it contains a year
    ym = re.search(r"/(20\d{2}|19\d{2})\b", u)
    if ym:
        try:
            yr = int(ym.group(1))
            if yr >= 2023:
                s += 1.2
            elif yr <= 2018:
                s -= 1.0
        except Exception:
            pass

    # Query token overlap
    for w in set(re.findall(r"[a-zA-ZÀ-ÿ]{4,}", ql)):
        if w in sn or w in u:
            s += 0.03

    # Structured files are often the gold source
    if any(u.endswith(ext) for ext in (".pdf", ".xlsx", ".xls", ".csv", ".json")):
        s += 0.9

    # Aggregators get a penalty (light if discovery ON, stronger if OFF)
    if _is_aggregator(u):
        s -= (3.0 if not allow_discovery else 1.0)

    # Hard penalties
    if _is_probably_dead(u):
        s -= 5.0

    # ICA bonus on INS with key phrases
    if _looks_like_ica(question) and _domain_of(u) == "ins.tn":
        if _has_any(u, ["commerce-de-detail", "commerce-de-détail", "commerce de detail", "commerce de détail", "chiffre d'affaires", "chiffre d’affaires", "ica"]):
            s += 1.5

    return s


# ========= Persistence (optional) =========
def _append_link_bank(urls: List[str]) -> None:
    LINKS_JSON.parent.mkdir(parents=True, exist_ok=True)
    existing: List[str] = []
    if LINKS_JSON.exists():
        try:
            existing = [str(x).strip() for x in json.loads(LINKS_JSON.read_text("utf-8")) if str(x).strip()]
        except Exception:
            existing = []
    seen = set(existing)
    merged = existing[:]
    for u in urls:
        if u not in seen:
            merged.append(u)
            seen.add(u)
    LINKS_JSON.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    try:
        import pandas as pd
        pd.DataFrame(merged, columns=["URL"]).to_excel(LINKS_XLSX, index_label="Link No")
    except Exception:
        pass


# ========= Helpers for gating =========
def _official_allowed(u: str, allow_discovery: bool, ica_only: bool) -> bool:
    if allow_discovery and not ica_only:
        return True
    if ica_only:
        return _domain_of(u).endswith("ins.tn")
    return _is_official(u)


# ========= Public API =========
def pick_verified_urls(
    question: str,
    top_k: int = 3,
    allow_discovery: bool = False,
    write_links: bool = True,
    min_score: float = 4.8,
) -> List[str]:
    """
    Return high-signal URLs for THIS question.

    Behavior:
    - allow_discovery=False: STRICT official-only (INS/BCT); ICA questions → INS only.
    - allow_discovery=True: broader search; aggregators allowed but rank lower.
    - HEAD/GET ping to drop obvious non-2xx before scraping.
    """
    ica_only = _looks_like_ica(question)

    # 1) search
    cands = _search_candidates(question, allow_discovery=allow_discovery, k=max(10, top_k * 4))
    if not cands:
        # sensible official seeds if search failed
        if ica_only:
            seeds = [
                "https://www.ins.tn/statistiques",
                "https://www.ins.tn/publications?keys=commerce%20de%20d%C3%A9tail",
            ]
        else:
            seeds = [
                "https://www.ins.tn/statistiques",
                "https://www.bct.gov.tn/bct/siteprod/indicateurs.jsp",
            ]
        return seeds[:top_k]

    # 2) score + sort
    scored = sorted(((_score(u, sn, question, allow_discovery), u) for (u, sn) in cands), reverse=True)

    # 3) strict pass
    picks: List[str] = []
    for s, u in scored:
        if s < min_score:
            continue
        if not _official_allowed(u, allow_discovery, ica_only):
            continue
        picks.append(u)
        if len(picks) >= top_k:
            break

    # 4) relaxed pass if needed
    if len(picks) < top_k and scored:
        relaxed = max(min_score - 1.6, 3.0)
        for s, u in scored:
            if len(picks) >= top_k:
                break
            if s < relaxed:
                continue
            if not _official_allowed(u, allow_discovery, ica_only):
                continue
            if u not in picks:
                picks.append(u)

    # 5) enforce top_k unique + quick HEAD validation
    uniq: List[str] = []
    seen = set()
    for u in picks:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
        if len(uniq) >= top_k:
            break

    validated: List[str] = []
    for u in uniq:
        if _fast_head_ok(u):
            validated.append(u)
    if not validated and uniq:
        validated = uniq[:1]  # keep at least one if all HEADs fail

    # 6) ultimate fallback to safe official hubs
    if not validated:
        if ica_only:
            validated = [
                "https://www.ins.tn/statistiques",
                "https://www.ins.tn/publications?keys=commerce%20de%20d%C3%A9tail",
            ][:top_k]
        else:
            validated = [
                "https://www.ins.tn/statistiques",
                "https://www.bct.gov.tn/bct/siteprod/indicateurs.jsp",
            ][:top_k]

    if write_links and validated:
        _append_link_bank(validated)

    return validated[:top_k]


# ========= LangChain Tools =========
def _coerce_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(x)

try:
    from langchain.tools import tool
except Exception:
    from langchain_core.tools import tool  # fallback


@tool("pick_urls")
def pick_urls_tool_main(question: str, top_k: str | int = 3, allow_discovery: str | bool = True) -> str:
    """Pick URLs (newline-separated) for THIS question; uses SERP + scoring + HEAD validation."""
    try:
        k = int(top_k)
    except Exception:
        k = 3
    allow = _coerce_bool(allow_discovery)
    urls = pick_verified_urls(question=question, top_k=k, allow_discovery=allow, write_links=True)
    return "\n".join(urls)


@tool("pick_urls_tool")
def pick_urls_tool_alias(question: str, top_k: str | int = 3, allow_discovery: str | bool = True) -> str:
    """Alias for models that call 'pick_urls_tool'."""
    return pick_urls_tool_main(question=question, top_k=top_k, allow_discovery=allow_discovery)
