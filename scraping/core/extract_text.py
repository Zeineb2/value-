# scraping/core/extract_text.py
from __future__ import annotations

import os
import time
import re
import json
import runpy
from typing import List, Tuple

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
    _NLP.max_length = 4_000_000
except Exception:
    _NLP = None

from ..utils.indicator_matcher import (
    match_indicators, extract_year, nlp_match_indicators
)
from .utils import (
    canonicalize, score_confidence, format_display,
    extract_domain_from_filename, is_economic_context
)

# Single, authoritative taxonomy file
TAXONOMY_PATH = os.path.abspath("economic_indicator.json")
# Optional script you use to rebuild alias maps, if present
CANON_SCRIPT = os.path.join("scraping", "canonical_indicators.py")


# ───────────────────────────────────────────────────────────────────────────────
# Small helpers
# ───────────────────────────────────────────────────────────────────────────────

def _normalize_matches(matches, sentence_or_line: str) -> List[dict]:
    """Normalize various matcher outputs to [{Indicator, RawText, Method}]"""
    if matches is None:
        return []
    if not isinstance(matches, list):
        matches = [matches]
    norm = []
    for m in matches:
        if isinstance(m, dict) and "Indicator" in m:
            m.setdefault("RawText", sentence_or_line)
            m.setdefault("Method", "Alias")
            norm.append(m)
        elif isinstance(m, str):
            norm.append({"Indicator": m, "RawText": sentence_or_line, "Method": "Alias"})
        elif isinstance(m, tuple) and len(m) >= 1:
            norm.append({"Indicator": m[0], "RawText": sentence_or_line, "Method": "Alias"})
        else:
            norm.append({"Indicator": str(m), "RawText": sentence_or_line, "Method": "Alias"})
    return norm


def _safe_canonicalize(indicator_name: str) -> dict:
    """Return {'canonical', 'category'} safely for any phrase."""
    try:
        c = canonicalize(indicator_name)
        if isinstance(c, dict):
            canonical = c.get("canonical") or c.get("Canonical Name") or c.get("name") or indicator_name
            category = c.get("category") or c.get("Category") or "Unknown"
        else:
            canonical = str(c)
            category = "Unknown"
    except Exception:
        canonical = indicator_name
        category = "Unknown"
    return {"canonical": canonical, "category": category}


def _nlp_doc(text: str):
    """Split into sentences; fallback to regex if spaCy unavailable."""
    if _NLP is None:
        class _Sent:
            def __init__(self, t): self.text = t
        return type("Doc", (), {"sents": [_Sent(s.strip()) for s in re.split(r"[.!?]\s+", text) if s.strip()]})()
    return _NLP(text)


def token_distance(sentence: str, term: str, number: float) -> int:
    try:
        sentence = sentence.lower()
        term = term.lower()
        number_str = str(int(float(number)))
        tokens = re.findall(r'\w+(?:\.\d+)?', sentence)
        term_positions = [i for i, tok in enumerate(tokens) if term in tok]
        number_positions = [i for i, tok in enumerate(tokens) if number_str in tok]
        if not term_positions or not number_positions:
            return 9999
        return min(abs(t - n) for t in term_positions for n in number_positions)
    except Exception:
        return 9999


def is_valid_value(val: float, indicator: str) -> bool:
    ind = indicator.lower()
    if any(k in ind for k in ["gdp", "budget", "deficit", "income", "exports", "imports"]):
        return -1e12 <= val <= 1e12
    if any(k in ind for k in ["unemployment", "inflation", "growth", "rate"]):
        return 0 <= val <= 100
    return -1e12 <= val <= 1e12


def is_comparison_reference(sentence: str, indicator: str) -> bool:
    pattern = r"\b\d+(\.\d+)?\s*(%|percent)\s+of\s+" + re.escape(indicator.lower())
    return re.search(pattern, sentence.lower()) is not None


def is_conflicting_context(sentence: str, indicator: str) -> bool:
    ind = indicator.lower()
    s = sentence.lower()
    if any(w in s for w in ["increase", "decrease", "rise", "fall", "change"]):
        if "%" in s and not is_comparison_reference(sentence, indicator):
            return True
    if "gdp" in ind and any(w in s for w in ["cad", "current account", "deficit", "surplus", "balance"]):
        return True
    if "deficit" in ind and "gdp" in s and "% of" in s:
        return True
    return False


def extract_all_values(text: str) -> List[Tuple[float, str | None]]:
    """Loose numeric extractor; tags simple units so we can keep %/currency."""
    unit_keywords = {
        "%": ["%", "percent", "percentage"],
        "USD": ["usd", "dollars", "us dollars", "million usd", "billion usd"],
        "TND": ["tnd", "dinars", "million tnd", "billion tnd"],
        "EUR": ["eur", "euros", "million eur"]
    }
    raw_matches = re.findall(
        r"([0-9]+(?:\.[0-9]+)?)\s*(%|usd|eur|tnd|million|billion|percent|dollars|dinars|euros)?",
        text.lower()
    )
    results: List[Tuple[float, str | None]] = []
    for val, unit in raw_matches:
        val = float(val)
        detected_unit = None
        for symbol, variants in unit_keywords.items():
            if unit and unit in variants:
                detected_unit = symbol
                break
        results.append((val, detected_unit))
    return results


# ───────────────────────────────────────────────────────────────────────────────
# Auto-update taxonomy with new aliases/canonicals
# ───────────────────────────────────────────────────────────────────────────────

def _load_taxonomy_list() -> list:
    if not os.path.exists(TAXONOMY_PATH):
        os.makedirs(os.path.dirname(TAXONOMY_PATH) or ".", exist_ok=True)
        json.dump([], open(TAXONOMY_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        return []
    try:
        data = json.load(open(TAXONOMY_PATH, "r", encoding="utf-8"))
        items = data.get("indicators", data) if isinstance(data, dict) else data
        return items if isinstance(items, list) else []
    except Exception:
        return []


def _atomic_write_json(path: str, obj: object) -> None:
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _looks_like_alias(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    if len(s) < 3 or len(s) > 120:
        return False
    if any(ch in s for ch in "\n\r\t"):
        return False
    return True


def _rebuild_alias_map_if_possible() -> bool:
    """Optional: refresh any derived alias maps your project keeps."""
    if not os.path.exists(CANON_SCRIPT):
        return False
    try:
        runpy.run_path(CANON_SCRIPT, run_name="__main__")
        return True
    except Exception:
        return False


def update_taxonomy_alias(canonical_name: str, alias: str) -> bool:
    """
    Ensure `economic_indicator.json` contains the canonical entry and alias.
    If the canonical is new → create a new entry.
    If alias is new → append.
    Returns True if taxonomy changed.
    """
    changed = False
    try:
        if not _looks_like_alias(alias):
            return False

        items = _load_taxonomy_list()

        # find canonical
        idx = None
        for i, it in enumerate(items):
            if isinstance(it, dict) and (it.get("Canonical Name") or "").lower() == canonical_name.lower():
                idx = i
                break

        if idx is None:
            # new canonical
            items.append({
                "Canonical Name": canonical_name,
                "Aliases": [] if alias.lower() == canonical_name.lower() else [alias],
                "Category": None,
                "Unit": None
            })
            changed = True
        else:
            entry = items[idx]
            aliases = entry.get("Aliases") or []
            if alias.lower() not in {a.lower() for a in aliases} and alias.lower() != canonical_name.lower():
                aliases.append(alias)
                entry["Aliases"] = aliases
                items[idx] = entry
                changed = True

        if changed:
            _atomic_write_json(TAXONOMY_PATH, items)
        return changed
    except Exception:
        # Never block extraction due to taxonomy updates
        return False


# ───────────────────────────────────────────────────────────────────────────────
# Extraction routines (text)
# ───────────────────────────────────────────────────────────────────────────────

def extract_tabular_lines(full_text: str, indicators: list, filename: str) -> List[dict]:
    """
    Very lightweight table-ish extractor: looks for a label line followed by
    separate 'years' and 'values' lines below it.
    """
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    results: List[dict] = []
    changed_any = False

    for i, line in enumerate(lines):
        matches = _normalize_matches(match_indicators(line, indicators), line) \
                + _normalize_matches(nlp_match_indicators(line), line)
        if not matches:
            continue

        year_row = None
        value_row = None
        for offset in range(1, 6):
            if i + offset < len(lines):
                candidate = lines[i + offset]
                if re.findall(r"\b(19\d{2}|20\d{2})\b", candidate):
                    year_row = candidate
                elif re.findall(r"\d+(?:\.\d+)?", candidate):
                    value_row = candidate
        if not year_row or not value_row:
            continue

        years = re.findall(r"\b(19\d{2}|20\d{2})\b", year_row)
        raw_values = re.findall(r"\d+(?:\.\d+)?", value_row)
        values = [float(v.replace(",", ".")) for v in raw_values]

        if len(years) != len(values):
            limit = min(len(years), len(values))
            years, values = years[:limit], values[:limit]

        for match in matches:
            raw_alias = match["Indicator"]
            canon = _safe_canonicalize(raw_alias)
            if update_taxonomy_alias(canon["canonical"], raw_alias):
                changed_any = True

            for idx in range(len(years)):
                try:
                    year = int(years[idx]); val = float(values[idx])
                except Exception:
                    continue
                confidence = score_confidence(True, year, val, None)
                results.append({
                    "Indicator": canon["canonical"],
                    "Indicator Name": canon["canonical"],
                    "Year": year,
                    "Value": val,
                    "Unit": None,
                    "Confidence": confidence,
                    "RawText": f"{line} | {year_row} | {value_row}",
                    "DisplayValue": format_display(val, None),
                    "Source": extract_domain_from_filename(filename),
                    "Method": match.get("Method", "Tabular"),
                    "CanonicalIndicator": canon["canonical"],
                    "Canonical Name": canon["canonical"],
                    "Category": canon["category"],
                    "FileRef": filename  # ← so extractor can attach SourceURL via manifest
                })

    if changed_any:
        _rebuild_alias_map_if_possible()
    return results


def process_table_block(table_lines: List[str], indicators: list, filename: str) -> List[dict]:
    """
    Alternate table layout: header row with years, each subsequent row labeled line with values.
    """
    results: List[dict] = []
    changed_any = False
    if len(table_lines) < 2:
        return results

    header_line = table_lines[0]
    header_years = re.findall(r"\b(19\d{2}|20\d{2})\b", header_line)

    for i in range(1, len(table_lines)):
        line = table_lines[i]
        matches = _normalize_matches(match_indicators(line, indicators), line) \
                + _normalize_matches(nlp_match_indicators(line), line)
        if not matches:
            continue

        raw_vals = [v.replace(",", "") for v in re.findall(r"\d[\d.,]*", line)]
        values = [float(v.replace(",", ".")) for v in raw_vals if re.match(r"^\d+(\.\d+)?$", v.replace(",", "."))]

        if len(header_years) != len(values):
            limit = min(len(header_years), len(values))
            years = header_years[:limit]
            values = values[:limit]
        else:
            years = header_years

        for match in matches:
            raw_alias = match["Indicator"]
            canon = _safe_canonicalize(raw_alias)
            if update_taxonomy_alias(canon["canonical"], raw_alias):
                changed_any = True

            for idx in range(len(years)):
                try:
                    year = int(years[idx]); val = float(values[idx])
                except Exception:
                    continue
                confidence = score_confidence(True, year, val, None)
                results.append({
                    "Indicator": canon["canonical"],
                    "Indicator Name": canon["canonical"],
                    "Year": year,
                    "Value": val,
                    "Unit": None,
                    "Confidence": confidence,
                    "RawText": f"{match['RawText']} | {line}",
                    "DisplayValue": format_display(val, None),
                    "Source": extract_domain_from_filename(filename),
                    "Method": match.get("Method", "Tabular"),
                    "CanonicalIndicator": canon["canonical"],
                    "Canonical Name": canon["canonical"],
                    "Category": canon["category"],
                    "FileRef": filename  # ← keep origin file reference
                })

    if changed_any:
        _rebuild_alias_map_if_possible()
    return results


def extract_sentences(full_text: str, indicators: list, filename: str) -> List[dict]:
    """
    Sentence-level extraction (works for paragraphs, tables converted to lines, press releases).
    """
    results: List[dict] = []
    changed_any = False
    doc = _nlp_doc(full_text)

    for sent in doc.sents:
        sentence = sent.text.strip()
        if not sentence or not is_economic_context(sentence):
            continue

        matches = _normalize_matches(match_indicators(sentence, indicators), sentence) \
                + _normalize_matches(nlp_match_indicators(sentence), sentence)
        if not matches:
            continue

        values = extract_all_values(sentence)
        year = extract_year(sentence)
        used_values: set = set()

        for match in matches:
            raw_alias = match["Indicator"]
            canon = _safe_canonicalize(raw_alias)
            if update_taxonomy_alias(canon["canonical"], raw_alias):
                changed_any = True

            valid_candidates = [
                v for v in values
                if is_valid_value(v[0], raw_alias)
                and not is_conflicting_context(sentence, raw_alias)
                and not is_comparison_reference(sentence, raw_alias)
                and (
                    token_distance(sentence, raw_alias, v[0]) < 75
                    or v[1] in {"%", "USD", "TND", "EUR"}
                    or (year is not None and abs(v[0] - year) <= 1)
                )
                and v not in used_values
                and not (1970 <= int(v[0]) <= 2035 and v[1] is None and (year is None or abs(v[0] - year) > 1))
            ]
            if not valid_candidates:
                continue

            valid_candidates.sort(key=lambda v: token_distance(sentence, raw_alias, v[0]))
            value, unit = valid_candidates[0]
            used_values.add((value, unit))
            confidence = score_confidence(True, year, value, unit)

            results.append({
                "Indicator": canon["canonical"],
                "Indicator Name": canon["canonical"],
                "Year": year,
                "Value": value,
                "Unit": unit,
                "Confidence": confidence,
                "RawText": sentence,
                "DisplayValue": format_display(value, unit),
                "Source": extract_domain_from_filename(filename),
                "Method": match.get("Method", "Alias"),
                "CanonicalIndicator": canon["canonical"],
                "Canonical Name": canon["canonical"],
                "Category": canon["category"],
                "FileRef": filename  # ← keep origin file reference
            })

    if changed_any:
        _rebuild_alias_map_if_possible()
    return results


def extract_from_text(indicators: list, folder: str) -> List[dict]:
    results: List[dict] = []
    os.makedirs(folder, exist_ok=True)

    for filename in os.listdir(folder):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(folder, filename)
        print(f"📄 Processing file: {filename}")
        start = time.time()
        try:
            full_text = open(path, "r", encoding="utf-8").read().replace("\r", " ").strip()
            results.extend(extract_tabular_lines(full_text, indicators, filename))
            results.extend(process_table_block(full_text.splitlines(), indicators, filename))
            results.extend(extract_sentences(full_text, indicators, filename))
        except Exception as e:
            print(f"⚠️ Skipped {filename}: {e}")
        finally:
            print(f"✅ Done {filename} in {time.time() - start:.2f}s")
    return results
