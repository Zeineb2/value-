# scraping/core/extractor.py
from __future__ import annotations

import os
import re
import json
import math
import pandas as pd
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any

from .extract_text import extract_from_text
from .extract_pdf import extract_from_pdfs
from .utils import is_valid_entry, remove_duplicates, convert_to_triples

# Folders produced by your pipeline
TEXT_FOLDER = "data/text"
PDF_FOLDER = "data/files"

# SINGLE taxonomy file (singular, source of truth)
# This is intentionally a module-level variable so agentic.tools.hybrid_ingest
# can re-point it before running extraction.
INDICATOR_JSON = os.path.abspath("economic_indicator.json")

# Outputs
OUTPUT_DIR = os.path.join("scraping", "output")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "improved_structured_indicators.json")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "improved_structured_indicators.csv")
SUMMARY_STATS = os.path.join(OUTPUT_DIR, "summary_stats.json")
TRIPLES_JSON = os.path.join(OUTPUT_DIR, "graph_triples.json")
MANIFEST_JSON = os.path.join(OUTPUT_DIR, "download_manifest.json")  # optional: filename -> URL


# ───────────────────────────────────────────────────────────────────────────────
# Utilities
# ───────────────────────────────────────────────────────────────────────────────

def safe_load_json(path: str) -> Any:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _alias_keys(rec: dict) -> dict:
    """Unify common field names without breaking existing code."""
    if "Canonical Name" in rec and "CanonicalIndicator" not in rec:
        rec["CanonicalIndicator"] = rec["Canonical Name"]
    if "Indicator" not in rec:
        for k in ("Indicator Name", "indicator", "name"):
            if k in rec and isinstance(rec[k], str):
                rec["Indicator"] = rec[k]
                break
    if "SourceURL" not in rec and "URL" in rec and isinstance(rec["URL"], str):
        rec["SourceURL"] = rec["URL"]
    return rec


def _alias_list(items: List[dict]) -> List[dict]:
    return [_alias_keys(dict(it)) for it in items if isinstance(it, dict)]


def _normalize_phrase(s: str) -> str:
    """Casefold + collapse whitespace + strip most punctuation."""
    t = (s or "").casefold()
    t = re.sub(r"[\s\-_]+", " ", t).strip()
    t = re.sub(r"[^\w\s%()\[\]/\.]", "", t)  # keep %, (), [], /, .
    return t


def _to_float_or_none(x) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, str):
            y = x.replace("\u202f", " ").replace("\xa0", " ")
            y = y.replace(",", ".")
            y = re.sub(r"[^\d\.\-\+eE]", "", y)
            if y in ("", "-", "+", ".", "+.", "-."):
                return None
            return float(y)
        if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
            return float(x)
    except Exception:
        return None
    return None


def _coerce_year(rec: dict) -> Optional[int]:
    """Infer Year from DateISO or Year field if possible."""
    # Prefer DateISO → YYYY-MM-DD or YYYY-MM
    for key in ("DateISO", "date_iso", "date", "Date"):
        val = rec.get(key)
        if isinstance(val, str):
            m = re.search(r"\b(19[7-9]\d|20[0-4]\d|2025)\b", val)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    pass
    # Then plain Year field
    y = rec.get("Year")
    try:
        return int(y) if y is not None and str(y).isdigit() else None
    except Exception:
        return None


# ───────────────────────────────────────────────────────────────────────────────
# Taxonomy manager (open-world, auto-expanding)
# Format: [
#   { "Canonical Name": "Policy interest rate (%)", "Aliases": ["taux directeur", "monetary policy rate"] },
#   ...
# ]
# ───────────────────────────────────────────────────────────────────────────────

class Taxonomy:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

        raw = safe_load_json(self.path)
        # tolerate legacy {"indicators":[...]} shape
        self.items: List[Dict] = raw.get("indicators", raw) if isinstance(raw, dict) else raw
        if not isinstance(self.items, list):
            self.items = []

        self._rebuild_index()

    def _rebuild_index(self):
        self._canon_norm_to_idx: Dict[str, int] = {}
        self._alias_norm_to_canon: Dict[str, str] = {}

        for i, it in enumerate(self.items):
            canon = (it.get("Canonical Name") or it.get("name") or "").strip()
            aliases = it.get("Aliases") or it.get("aliases") or []
            canon_norm = _normalize_phrase(canon)
            if canon_norm:
                self._canon_norm_to_idx[canon_norm] = i
            for a in aliases:
                a_norm = _normalize_phrase(a)
                if a_norm:
                    self._alias_norm_to_canon[a_norm] = canon

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.items, f, ensure_ascii=False, indent=2)
        self._rebuild_index()

    def find_canonical_by_alias(self, alias: str) -> Optional[str]:
        if not alias:
            return None
        return self._alias_norm_to_canon.get(_normalize_phrase(alias))

    def find_item_by_canonical(self, canonical: str) -> Optional[Dict]:
        idx = self._canon_norm_to_idx.get(_normalize_phrase(canonical))
        return self.items[idx] if idx is not None else None

    def ensure_alias(self, canonical: str, alias: str) -> bool:
        """Ensure alias exists under canonical. Returns True if taxonomy mutated."""
        if not canonical or not alias:
            return False
        item = self.find_item_by_canonical(canonical)
        if item is None:
            self.items.append({"Canonical Name": canonical, "Aliases": [alias]})
            self._rebuild_index()
            return True
        aliases = item.setdefault("Aliases", [])
        if alias not in aliases:
            aliases.append(alias)
            self._rebuild_index()
            return True
        return False

    def add_canonical(self, canonical: str, alias: Optional[str] = None) -> bool:
        """Add a brand-new canonical (with optional first alias)."""
        if not canonical:
            return False
        if self.find_item_by_canonical(canonical) is not None:
            if alias:
                return self.ensure_alias(canonical, alias)
            return False
        entry = {"Canonical Name": canonical, "Aliases": [alias] if alias else []}
        self.items.append(entry)
        self._rebuild_index()
        return True


# ───────────────────────────────────────────────────────────────────────────────
# Mapping & normalization per record
# ───────────────────────────────────────────────────────────────────────────────

def _attach_source_url(rec: dict, manifest: Dict[str, str]) -> None:
    """
    Fill SourceURL using FileRef via manifest, if missing.
    Manifest is { filename: original_url } written by the downloader.
    """
    if rec.get("SourceURL"):
        return
    ref = rec.get("FileRef") or rec.get("file") or rec.get("filename")
    if ref and isinstance(ref, str):
        url = manifest.get(ref)
        if url:
            rec["SourceURL"] = url


def _infer_canonical_and_alias(rec: dict, tax: Taxonomy) -> Tuple[str, Optional[str], bool]:
    """
    Decide the CanonicalIndicator and alias (if any) for this record.
    Returns (canonical, alias_used, taxonomy_mutated).
    """
    mutated = False

    # Prefer an explicit canonical if the extractor already produced one
    explicit_canon = (rec.get("CanonicalIndicator") or rec.get("Canonical Name") or "").strip()
    raw_indicator = (rec.get("Indicator") or rec.get("Indicator Name") or "").strip()

    if explicit_canon:
        if tax.find_item_by_canonical(explicit_canon) is None:
            mutated |= tax.add_canonical(explicit_canon, alias=raw_indicator or None)
        elif raw_indicator:
            mutated |= tax.ensure_alias(explicit_canon, raw_indicator)
        return explicit_canon, raw_indicator or None, mutated

    # No explicit canonical – try map raw_indicator through aliases
    if raw_indicator:
        canon = tax.find_canonical_by_alias(raw_indicator)
        if canon:
            mutated |= tax.ensure_alias(canon, raw_indicator)
            return canon, raw_indicator, mutated
        else:
            # Unknown phrase → create a new canonical using the phrase itself
            mutated |= tax.add_canonical(raw_indicator, alias=raw_indicator)
            return raw_indicator, raw_indicator, mutated

    # Fallback: try to guess from any other fields (very defensive)
    for k in ("indicator", "name", "Label", "Title"):
        if k in rec and isinstance(rec[k], str) and rec[k].strip():
            phrase = rec[k].strip()
            canon = tax.find_canonical_by_alias(phrase)
            if canon:
                mutated |= tax.ensure_alias(canon, phrase)
                return canon, phrase, mutated
            mutated |= tax.add_canonical(phrase, alias=phrase)
            return phrase, phrase, mutated

    return "", None, mutated


def _build_page_content(rec: dict) -> str:
    """
    A compact, readable summary used for FAISS page_content:
      "Indicator: X | Date: 2023-11-01 | Value: 8.5 % | Source: INS"
    """
    parts = []
    if rec.get("CanonicalIndicator"):
        parts.append(f"Indicator: {rec['CanonicalIndicator']}")
    elif rec.get("Indicator"):
        parts.append(f"Indicator: {rec['Indicator']}")

    if rec.get("DateISO"):
        parts.append(f"Date: {rec['DateISO']}")
    else:
        y = rec.get("Year")
        if y:
            parts.append(f"Year: {y}")

    if rec.get("Value") is not None:
        unit = f" {rec.get('Unit')}" if rec.get("Unit") else ""
        parts.append(f"Value: {rec['Value']}{unit}")

    src = rec.get("Source") or rec.get("SourceName")
    if not src and rec.get("SourceURL"):
        # derive a short source hint from URL domain
        m = re.search(r"https?://(?:www\.)?([^/]+)/?", rec["SourceURL"])
        if m:
            src = m.group(1)
    if src:
        parts.append(f"Source: {src}")

    return " | ".join(parts)


def _normalize_record(rec: dict, tax: Taxonomy, manifest: Dict[str, str]) -> Tuple[dict, bool]:
    """
    Ensure record has CanonicalIndicator, attach SourceURL, coerce numeric fields,
    infer Year (if missing), and keep original fields. Returns (record, taxonomy_mutated).
    """
    rec = _alias_keys(dict(rec))  # unify key names

    # Attach URL if we have a manifest mapping (agentic downloads write it)
    _attach_source_url(rec, manifest)

    # Canonical ↔ alias mapping / auto-expansion
    canonical, alias_used, mutated = _infer_canonical_and_alias(rec, tax)
    if canonical:
        rec["CanonicalIndicator"] = canonical

    # Normalize numerics
    val_num = _to_float_or_none(rec.get("Value"))
    if val_num is not None:
        rec["Value"] = val_num
    # Normalize unit (leave as-is if present)
    if "Unit" in rec and isinstance(rec["Unit"], str):
        rec["Unit"] = rec["Unit"].strip()
    # Infer Year if absent
    if rec.get("Year") is None:
        y = _coerce_year(rec)
        if y is not None:
            rec["Year"] = y

    # Provide a clean page_content for FAISS
    if "page_content" not in rec or not isinstance(rec["page_content"], str) or not rec["page_content"].strip():
        rec["page_content"] = _build_page_content(rec)

    return rec, mutated


# ───────────────────────────────────────────────────────────────────────────────
# Merge helper (stable)
# ───────────────────────────────────────────────────────────────────────────────

def merge_across_runs(existing: List[dict], new_items: List[dict]) -> List[dict]:
    def make_key(r: dict):
        return (
            (r.get("CanonicalIndicator") or r.get("Indicator") or "").strip().lower(),
            r.get("DateISO") or r.get("Year"),
            r.get("Value"),
            (r.get("Unit") or "").strip().lower(),
            (r.get("SourceURL") or r.get("Source") or "").strip().lower(),
        )

    merged_map: Dict[Tuple, dict] = {}
    for item in existing:
        merged_map[make_key(item)] = item
    for item in new_items:
        merged_map[make_key(item)] = item
    return list(merged_map.values())


# ───────────────────────────────────────────────────────────────────────────────
# Main entrypoint
# ───────────────────────────────────────────────────────────────────────────────

def extract_structured_indicators() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load taxonomy (open-world). The agentic flow may update INDICATOR_JSON before this call.
    tax = Taxonomy(INDICATOR_JSON)

    # Optional manifest for SourceURL enrichment
    manifest_obj = safe_load_json(MANIFEST_JSON)
    manifest = manifest_obj if isinstance(manifest_obj, dict) else {}

    print("🔍 Running extraction from text and PDF...")
    # Extract text-first (HTML → text), then PDFs (data/files)
    text_results = extract_from_text(tax.items, TEXT_FOLDER)
    pdf_results = extract_from_pdfs(tax.items, PDF_FOLDER)
    raw_results = text_results + pdf_results
    print(f"🔎 Total raw extracted this run: {len(raw_results)}")

    # Normalize + auto-expand taxonomy as needed
    normalized: List[dict] = []
    taxonomy_mutations = 0
    for rec in raw_results:
        norm_rec, mutated = _normalize_record(rec, tax, manifest)
        normalized.append(norm_rec)
        taxonomy_mutations += 1 if mutated else 0

    # Persist taxonomy if we added canonicals/aliases
    if taxonomy_mutations > 0:
        tax.save()
        print(f"🧭 Taxonomy updated: +{taxonomy_mutations} additions (canonicals/aliases)")

    # Validate + de-dupe this run
    filtered = [r for r in normalized if is_valid_entry(r)]
    deduped_this_run = remove_duplicates(filtered)
    print(f"🧹 This run after validation+dedupe: {len(deduped_this_run)}")

    # Merge with previous runs
    existing = _alias_list(safe_load_json(OUTPUT_JSON))
    print(f"📦 Previously saved records: {len(existing)}")

    merged = merge_across_runs(existing, deduped_this_run)
    print(f"🧩 After merging with previous runs (pre-final-dedupe): {len(merged)}")

    # Final dedupe + alias normalization
    merged = remove_duplicates(merged)
    merged = _alias_list(merged)
    print(f"✅ Final merged & deduped total: {len(merged)}")

    # Write outputs
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    pd.DataFrame(merged).to_csv(OUTPUT_CSV, index=False)

    # Summary
    summary = {
        "Taxonomy Mutations (this run)": taxonomy_mutations,
        "Total Extracted (this run)": len(deduped_this_run),
        "Previously Saved": len(existing),
        "Total After Merge": len(merged),
        "Top Indicators": Counter(
            [r.get("CanonicalIndicator") or r.get("Indicator") for r in merged if (r.get("CanonicalIndicator") or r.get("Indicator"))]
        ).most_common(10),
        "Top Sources": Counter(
            [r.get("SourceURL") or r.get("Source") for r in merged if (r.get("SourceURL") or r.get("Source"))]
        ).most_common(10),
        "Top Years": Counter([r.get("Year") for r in merged if r.get("Year") is not None]).most_common(10),
    }
    with open(SUMMARY_STATS, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"📊 Summary stats written to {SUMMARY_STATS}")

    # Triples for graph/QA enrichment
    triples = convert_to_triples(merged)
    with open(TRIPLES_JSON, "w", encoding="utf-8") as f:
        json.dump(triples, f, indent=2, ensure_ascii=False)
    print(f"🧠 {len(triples)} graph triples saved to {TRIPLES_JSON}")


if __name__ == "__main__":
    extract_structured_indicators()
