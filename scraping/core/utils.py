# scraping/core/utils.py
import unicodedata
from difflib import SequenceMatcher
from rapidfuzz import process
import json
from pathlib import Path

# Resolve canonical_indicators.json relative to this file:
# .../scraping/core/utils.py  ->  parents[1] == .../scraping
CANONICAL_PATH = Path(__file__).resolve().parents[1] / "utils" / "canonical_indicators.json"
with CANONICAL_PATH.open("r", encoding="utf-8") as f:
    CANONICAL_MAP = json.load(f)

def normalize(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8").lower()

def format_display(value, unit):
    if unit == "%" and value < 100:
        return f"{value:.1f} %"
    if value >= 1e9:
        return f"{value / 1e9:.2f} B {unit}"
    elif value >= 1e6:
        return f"{value / 1e6:.2f} M {unit}"
    return f"{value:,.0f} {unit}" if unit else f"{value:,.0f}"

def canonicalize(raw):
    key = normalize(raw)
    if key in CANONICAL_MAP:
        entry = CANONICAL_MAP[key]
        canonical = entry.get("canonical", raw).strip()
        category = entry.get("category")
        return {"canonical": canonical, "category": category}

    match = process.extractOne(key, CANONICAL_MAP.keys(), score_cutoff=85)
    if match:
        entry = CANONICAL_MAP[match[0]]
        canonical = entry.get("canonical", raw).strip()
        category = entry.get("category")
        return {"canonical": canonical, "category": category}

    return {"canonical": raw.title(), "category": "Uncategorized"}

def extract_domain_from_filename(filename):
    name = filename.lower().split("_")[0]
    if name.startswith("www."):
        name = name[4:]
    return name

def is_valid_entry(ind):
    if ind["Value"] is None or ind["Value"] == 0.0:
        return False
    if ind["Year"] is None or not (1970 <= ind["Year"] <= 2050):
        return False
    if not ind["Indicator"]:
        return False
    if any(x in ind["RawText"].lower() for x in ["access denied", "cookies required"]):
        return False
    if ind["Value"] == ind["Year"]:
        return False
    return True

def remove_duplicates(entries):
    seen = []
    unique = []
    for entry in entries:
        is_duplicate = False
        for existing in seen:
            if (
                existing["Indicator"].lower() == entry["Indicator"].lower() and
                existing["Year"] == entry["Year"] and
                abs(existing["Value"] - entry["Value"]) < 0.5 and
                SequenceMatcher(None, existing["RawText"], entry["RawText"]).ratio() > 0.85
            ):
                is_duplicate = True
                break
        if not is_duplicate:
            seen.append(entry)
            unique.append(entry)
    return unique

def score_confidence(matched, year, value, unit):
    score = 0
    if matched:
        score += 30
    if year and 1970 <= year <= 2050:
        score += 25
    if value is not None and year and value == year:
        score += -50 if unit is None else -10
    if value is not None and value != 0.0:
        score += 30
        if 0 < value < 1 and unit not in ["%", "TND", "USD", "EUR"]:
            score -= 10
        elif value < 10 and not unit:
            score -= 15
    if unit:
        score += 20 if unit in ["%", "USD", "TND", "EUR"] else 10
    return max(min(score, 100), 0)

def convert_to_triples(entries):
    triples = []
    for e in entries:
        subject = f"{e['Indicator']} in {e['Year']}"
        predicate = "has value"
        value_part = format_display(e['Value'], e['Unit'])
        obj = f"{value_part} ({e['Category']})"
        triples.append({
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "source": e.get("Source", ""),
            "confidence": e.get("Confidence", 0)
        })
    return triples

def is_economic_context(text):
    text = normalize(text)
    economic_verbs = [
        "increase", "decrease", "grow", "decline", "rise", "fall", "accelerate", "slow",
        "improve", "drop", "expand", "contract", "totaled", "amounted to", "stood at"
    ]
    economic_nouns = [
        "growth", "rate", "ratio", "value", "deficit", "surplus", "inflation", "deflation",
        "export", "import", "gdp", "income", "revenue", "expenditure", "investment", "consumption",
        "indicator", "contribution", "performance", "trend", "price", "wage", "debt", "balance", "productivity"
    ]
    matched_terms = [term for term in economic_verbs + economic_nouns if term in text]
    has_numeric_econ_format = any(tok in text for tok in ["%", "million", "billion", "usd", "tnd", "$", "€"])
    return len(matched_terms) >= 2 or (len(matched_terms) >= 1 and has_numeric_econ_format)
