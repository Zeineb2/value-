# scraping/utils/indicator_matcher.py
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

# spaCy is OPTIONAL (we fall back gracefully if it's not installed)
try:
    import spacy
    try:
        _NLP = spacy.load("en_core_web_sm")
    except Exception:
        _NLP = spacy.blank("en")  # tokenization only
    _NLP.max_length = 4_000_000
except Exception:
    _NLP = None

# Light fallback patterns (only used when taxonomy yields nothing).
# We DO NOT hard-map to a canonical here. We return the phrase found.
FALLBACK_KEYWORDS = [
    r"\bpolicy\s*(interest\s*)?rate\b",
    r"\btaux\s+directeur\b",
    r"\bmonetary\s+policy\s+rate\b",
    r"\bcore\s+inflation\b",
    r"\binflation\s+sous[-\s]?jacente\b",
    r"\bcore\s*cpi\b",
    r"\b(cpi|ipc|inflation|indice\s+des\s+prix)\b",
    r"\b(ppi|producer\s+price)\b",
    r"\bindustrial\s+production\b",
    r"\b(\bipi\b)\b",
    r"\bm2\b",
    r"\bmoney\s*supply\b",
    r"\bunemployment\b",
    r"\btaux\s+de\s+ch[oô]mage\b",
    r"\bcurrent\s+account\b",
    r"\bcompte\s+courant\b",
    r"\b(fx|foreign)\s+reserves\b",
    r"\br[ée]serves\b.*\b(fx|devises|foreign)\b",
    r"\b(primary\s+balance|budget\s+deficit|fiscal\s+balance)\b",
    r"\b(gdp|gross\s+domestic\s+product|pib)\b",
    r"\b(retail\s+sales(\s+index)?)\b",
]


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKD", (text or "")).encode("ascii", "ignore").decode("utf-8").lower().strip()


def _iter_nouny_phrases(text: str) -> List[str]:
    """Return noun-like candidate phrases (spaCy if available; else a light fallback)."""
    if _NLP is not None:
        doc = _NLP(text)
        phrases = set()
        if hasattr(doc, "noun_chunks"):
            for chunk in doc.noun_chunks:
                s = chunk.text.strip()
                if 3 <= len(s) <= 80:
                    phrases.add(s)
        for ent in getattr(doc, "ents", []):
            s = ent.text.strip()
            if 3 <= len(s) <= 80:
                phrases.add(s)
        return list(phrases)

    # Very light fallback: split and keep spans likely referencing indicators
    out: List[str] = []
    for part in re.split(r"[.;:\n]\s*", text):
        if re.search(r"(index|rate|balance|inflation|account|reserves|production|supply|exports|imports|gdp|deficit|unemployment|sales)", part, flags=re.I):
            part = part.strip()
            if 3 <= len(part) <= 160:
                out.append(part)
    return out


def load_indicators(json_path: str = "economic_indicator.json") -> List[Dict[str, Any]]:
    """
    Load taxonomy entries. Accepts either:
      - [{"Canonical Name": "...", "Aliases": [...]}]
      - {"indicators": [ {...}, ... ]}
    Returns a flat list of dicts with "Canonical Name" and "Aliases".
    """
    try:
        data = json.load(open(json_path, "r", encoding="utf-8"))
    except Exception:
        return []

    items = data.get("indicators", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []

    norm: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        canonical = (it.get("Canonical Name") or it.get("name") or "").strip()
        aliases = it.get("Aliases") or it.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        aliases = [a for a in aliases if isinstance(a, str) and a.strip()]
        if canonical or aliases:
            norm.append({"Canonical Name": canonical, "Aliases": aliases})
    return norm


def _find_original_span(text: str, pattern: str) -> Optional[str]:
    m = re.search(pattern, text, flags=re.I)
    return m.group(0) if m else None


def regex_match_aliases(text: str, indicators: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Try to match any canonical OR alias in text.
    IMPORTANT: We return the PHRASE seen in text as "Indicator" (not the canonical name).
    Downstream (extractor) will map it or auto-add a canonical+alias as needed.
    """
    out: List[Dict[str, Any]] = []
    text_norm = normalize(text)

    for entry in indicators or []:
        canonical = (entry.get("Canonical Name") or "").strip()
        aliases = entry.get("Aliases") or []

        if canonical:
            c_norm = normalize(canonical)
            if re.search(r"\b" + re.escape(c_norm) + r"\b", text_norm):
                orig = _find_original_span(text, r"\b" + re.escape(canonical) + r"\b") or canonical
                out.append({
                    "Indicator": orig,
                    "RawText": orig,
                    "Matched": True,
                    "Confidence": 90,
                    "Method": "Regex",
                })

        for alias in aliases:
            a_norm = normalize(alias)
            if not a_norm:
                continue
            if re.search(r"\b" + re.escape(a_norm) + r"\b", text_norm):
                orig = _find_original_span(text, r"\b" + re.escape(alias) + r"\b") or alias
                out.append({
                    "Indicator": orig,
                    "RawText": orig,
                    "Matched": True,
                    "Confidence": 80,
                    "Method": "Regex",
                })

    # dedupe by Indicator text
    seen = set()
    deduped = []
    for m in out:
        key = normalize(m["Indicator"])
        if key not in seen:
            seen.add(key)
            deduped.append(m)
    return deduped


def nlp_match_indicators(text: str) -> List[Dict[str, Any]]:
    """
    Very mild heuristic when taxonomy finds nothing.
    Returns phrases like "retail sales index", "policy rate", etc.
    """
    matches: List[Dict[str, Any]] = []
    s_norm = normalize(text)

    # keyword hits
    for pat in FALLBACK_KEYWORDS:
        m = re.search(pat, s_norm, flags=re.I)
        if m:
            phrase = m.group(0)
            if _NLP is not None:
                for span in _iter_nouny_phrases(text):
                    if phrase.lower() in span.lower():
                        phrase = span.strip()
                        break
            if 3 <= len(phrase) <= 120:
                matches.append({
                    "Indicator": phrase,
                    "RawText": phrase,
                    "Matched": True,
                    "Confidence": 60,
                    "Method": "Heuristic",
                })

    # extra nouny phrases (very conservative)
    for span in _iter_nouny_phrases(text):
        if re.search(r"(index|rate|balance|inflation|account|reserves|production|supply|exports|imports|gdp|deficit|unemployment|sales)", span, flags=re.I):
            s = span.strip()
            matches.append({
                "Indicator": s,
                "RawText": s,
                "Matched": True,
                "Confidence": 50,
                "Method": "Heuristic",
            })

    # dedupe by (Indicator, Method)
    seen = set()
    deduped = []
    for m in matches:
        key = (normalize(m["Indicator"]), m["Method"])
        if key not in seen:
            seen.add(key)
            deduped.append(m)
    return deduped


def has_conflicting_term(indicator_phrase: str, sentence: str) -> bool:
    """
    Minimal guard to avoid classic confusions (e.g., GDP vs current account).
    """
    s = sentence.lower()
    ind = indicator_phrase.lower()
    if "gdp" in ind and ("current account" in s or "deficit" in s or "cad" in s):
        return True
    if ("deficit" in ind or "current account" in ind) and "gdp" in s:
        return True
    return False


def match_indicators(text: str, indicators: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    1) Try taxonomy-based regex (returns PHRASES).
    2) Else use light NLP/regex heuristics (returns PHRASES).
    """
    exact = regex_match_aliases(text, indicators)
    base = exact if exact else nlp_match_indicators(text)

    cleaned: List[Dict[str, Any]] = []
    for m in base:
        if not has_conflicting_term(m["Indicator"], text):
            cleaned.append(m)
    return cleaned


def extract_year(text: str) -> Optional[int]:
    """Return a plausible year (1970..2035)."""
    m = re.search(r"\b(19[7-9]\d|20[0-3]\d|203[0-5])\b", text)
    return int(m.group()) if m else None


def extract_value(text: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract numeric value + coarse unit.
    Returns (value, unit) where unit in {%, USD, EUR, TND, None}.
    Filters out years masquerading as values unless context indicates otherwise.
    """
    t = (text or "").lower().replace("\u202f", " ").replace("\xa0", " ")
    pattern = r"(\d{1,3}(?:[\s.,]\d{3})+|\d+(?:[.,]\d+)?)(\s?(%|percent|tnd|usd|eur|dinar[s]?|euros?|dollars?|million[s]?|billion[s]?|thousand[s]?|milliers|milliard[s]?))?"
    m = re.search(pattern, t)
    if not m:
        return None, None

    raw = m.group(1)
    unit = (m.group(3) or "").strip().lower() or None

    raw = raw.replace("\u202f", "").replace(" ", "").replace(",", ".")
    try:
        val = float(raw)
    except ValueError:
        return None, None

    # looks like a year?
    if 1900 <= val <= 2099 and unit is None and not re.search(r"\b(as of|since|in|year|from|to|during|forecast|projected|between|by)\b", t):
        return None, None

    if unit:
        if unit in {"percent", "%"}:
            unit = "%"
        elif "dinar" in unit or unit == "tnd":
            unit = "TND"
        elif unit == "usd" or "dollar" in unit:
            unit = "USD"
        elif unit == "eur" or "euro" in unit:
            unit = "EUR"
        elif "billion" in unit or "milliard" in unit or "milliards" in unit:
            val *= 1e9
            unit = "USD"
        elif "million" in unit or "millions" in unit:
            val *= 1e6
            unit = "USD"
        elif "thousand" in unit or "milliers" in unit:
            val *= 1e3
            unit = "USD"

    return val, unit
