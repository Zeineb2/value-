# scraping/core/taxonomy_utils.py
from __future__ import annotations

import json, os, runpy, threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

# Single source of truth for indicators (singular file)
TAXONOMY_PATH = Path("economic_indicator.json")
# Your existing builder that emits utils/canonical_indicators.json
CANON_SCRIPT  = Path("scraping") / "canonical_indicators.py"
ALIAS_MAP_PATH = Path("utils") / "canonical_indicators.json"

_LOCK = threading.Lock()


def _atomic_write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(str(tmp), str(path))


def _load_list_or_empty(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        data = json.load(path.open("r", encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_singular(items: List[dict]) -> None:
    _atomic_write(TAXONOMY_PATH, items)


def _load_alias_map() -> Dict[str, Dict]:
    if ALIAS_MAP_PATH.exists():
        try:
            return json.load(ALIAS_MAP_PATH.open("r", encoding="utf-8"))
        except Exception:
            return {}
    return {}


def rebuild_canonical_map_if_possible() -> bool:
    """Re-run your builder to refresh utils/canonical_indicators.json."""
    if not CANON_SCRIPT.exists():
        return False
    try:
        runpy.run_path(str(CANON_SCRIPT), run_name="__main__")
        return True
    except Exception:
        return False


def _norm(s: str) -> str:
    return "".join(c for c in (s or "").lower().strip() if c.isalnum() or c.isspace())


def ensure_indicator_and_alias(alias: str, context: str = "") -> Tuple[str, bool]:
    """
    Ensure 'alias' exists in TAXONOMY_PATH under some canonical.
    Returns (canonical_name, changed_taxonomy).
    Strategy:
      1) If alias already present (via alias map) → return mapped canonical.
      2) Fuzzy-join to an existing canonical name (score ≥ 88) → append alias there.
      3) Else create a brand-new canonical (Title Case of alias) and add alias.
    """
    raw = (alias or "").strip()
    if not raw:
        return "", False

    # 1) Try alias map first (fast path)
    alias_map = _load_alias_map()
    key = _norm(raw)
    if key in alias_map:
        return alias_map[key].get("canonical", raw), False

    # 2) Fuzzy to existing canonical names
    with _LOCK:
        singular = _load_list_or_empty(TAXONOMY_PATH)
        best_name, best_score = None, -1
        if fuzz and singular:
            for item in singular:
                cname = (item.get("Canonical Name") or "").strip()
                if not cname:
                    continue
                score = fuzz.ratio(_norm(cname), key)
                if score > best_score:
                    best_name, best_score = cname, score

        if best_name and best_score >= 88:
            # append alias to nearest canonical
            for it in singular:
                if (it.get("Canonical Name") or "").strip().lower() == best_name.lower():
                    aliases = set((it.get("Aliases") or []))
                    if raw not in aliases:
                        it["Aliases"] = list(aliases | {raw})
                        _save_singular(singular)
                        rebuild_canonical_map_if_possible()
                        return best_name, True
                    return best_name, False

        # 3) Create a new canonical (Title Case)
        new_canonical = raw.title()
        singular.append({
            "Canonical Name": new_canonical,
            "Aliases": [raw],
        })
        _save_singular(singular)
        rebuild_canonical_map_if_possible()
        return new_canonical, True
