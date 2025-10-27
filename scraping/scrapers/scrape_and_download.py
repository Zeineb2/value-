# scraping/scrapers/scrape_and_download.py
from __future__ import annotations

import os
import re
import json
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import httpx
from urllib.parse import urlparse

# ---- Paths -------------------------------------------------------------------
HTML_DIR = "data/html"
FILES_DIR = "data/files"
OUTPUT_DIR = os.path.join("scraping", "output")
MANIFEST_JSON = os.path.join(OUTPUT_DIR, "download_manifest.json")

os.makedirs(HTML_DIR, exist_ok=True)
os.makedirs(FILES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- HTTP config --------------------------------------------------------------
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

DEFAULT_TIMEOUT = httpx.Timeout(50.0, connect=20.0, read=30.0)
HEADERS = {
    "User-Agent": UA,
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

# Optional size guard (MB)
try:
    from agentic.config import MAX_DOWNLOAD_MB as _MAX_MB  # reuse if present
except Exception:
    _MAX_MB = 50
MAX_DOWNLOAD_BYTES = int(_MAX_MB) * 1024 * 1024

# Guard rails for false/blocked pages
MIN_HTML_BYTES = 1500  # drop tiny HTML (likely error/placeholder)
BLOCK_HTML_IF_URL_LOOKS_BINARY = True  # e.g., URL ends with .pdf but returns text/html


# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────

def _safe_name(url: str) -> str:
    """
    Turn a URL into a deterministic, filesystem-safe name with a short hash.
    Must stay consistent with agentic/tools/hybrid_ingest.py
    """
    base = re.sub(r"[^a-zA-Z0-9._/-]+", "_", url).strip("/")
    base = base.replace("://", "_").replace("/", "_")
    if len(base) > 80:
        base = base[:80]
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{base}_{h}"


def _exists_recent(path: str, fresh_hours: int) -> bool:
    if not os.path.exists(path):
        return False
    if fresh_hours <= 0:
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return (datetime.now() - mtime) < timedelta(hours=fresh_hours)


def _parse_content_disposition_filename(cd: str) -> Optional[str]:
    """
    Extract filename from Content-Disposition header if present.
    Handles: attachment; filename="file.pdf"  OR filename*=UTF-8''file.pdf
    """
    if not cd:
        return None
    # RFC 5987 (filename*)
    m = re.search(r'filename\*\s*=\s*[^\'"]+\'\'([^;]+)', cd, flags=re.I)
    if m:
        return m.group(1).strip().strip('"')
    # Classic filename=
    m = re.search(r'filename\s*=\s*"([^"]+)"', cd, flags=re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r'filename\s*=\s*([^;]+)', cd, flags=re.I)
    if m:
        return m.group(1).strip().strip('"')
    return None


def _ext_from_filename(name: str) -> Optional[str]:
    name = name or ""
    lower = name.lower()
    for ext in (".pdf", ".xlsx", ".xls", ".csv", ".json", ".html", ".htm", ".xhtml", ".xml", ".txt"):
        if lower.endswith(ext):
            return ".html" if ext == ".htm" else ext
    return None


def _guess_extension(url: str, content_type: str, content_disp: str | None = None) -> str:
    """
    Decide file extension using (priority):
      1) Content-Disposition filename (if any)
      2) Content-Type
      3) URL path extension
      4) Default .html
    """
    # 1) Content-Disposition filename wins
    if content_disp:
        fn = _parse_content_disposition_filename(content_disp)
        ext = _ext_from_filename(fn) if fn else None
        if ext:
            return ext

    u = url.lower()
    ct = (content_type or "").lower()

    # 2) Content-Type
    if "application/pdf" in ct:
        return ".pdf"
    if "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in ct:
        return ".xlsx"
    if "application/vnd.ms-excel" in ct:
        return ".xls"
    if "text/csv" in ct or "application/csv" in ct:
        return ".csv"
    if "application/json" in ct:
        return ".json"
    if any(t in ct for t in ("text/html", "application/xhtml", "application/xml", "text/plain")):
        return ".html"

    # 3) URL path extension
    ext = _ext_from_filename(u)
    if ext:
        return ext

    # 4) Fallback to HTML
    return ".html"


def _load_manifest() -> Dict[str, str]:
    if not os.path.exists(MANIFEST_JSON):
        return {}
    try:
        with open(MANIFEST_JSON, "r", encoding="utf-8") as f:
            obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _save_manifest(manifest: Dict[str, str]) -> None:
    try:
        with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    except Exception:
        # non-fatal
        pass


def _target_paths(base_name: str, ext: str) -> Tuple[str, str]:
    """
    Returns (final_path, dir_label) where dir_label is 'html' or 'files'.
    """
    if ext == ".html":
        return os.path.join(HTML_DIR, f"{base_name}{ext}"), "html"
    else:
        return os.path.join(FILES_DIR, f"{base_name}{ext}"), "files"


def _origin_referer(url: str) -> str:
    """
    Best-effort origin referer (helps some hosts).
    """
    try:
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}/"
    except Exception:
        return ""


def _looks_like_block_page(text: str) -> bool:
    """
    Detect common block pages to avoid polluting corpus with useless HTML.
    """
    t = (text or "").lower()
    bad_snips = (
        "403", "forbidden", "access denied", "not authorized", "captcha",
        "you don't have permission", "you do not have permission",
        "robot check", "attention required",
    )
    return any(sn in t for sn in bad_snips)


def _url_looks_binary(url: str) -> bool:
    u = url.lower()
    return any(u.endswith(ext) for ext in (".pdf", ".xlsx", ".xls", ".csv", ".json"))


# ───────────────────────────────────────────────────────────────────────────────
# Downloader
# ───────────────────────────────────────────────────────────────────────────────

async def _download_with_retry(
    client: httpx.AsyncClient,
    url: str,
    fresh_hours: int,
    force: bool,
    retries: int = 1,
) -> Optional[Tuple[str, str]]:
    """
    Fetch URL and save to disk.
    Returns (saved_path, original_url) or None on failure.
    Retries once on transient network errors.
    """
    attempt = 0
    while True:
        try:
            base = _safe_name(url)

            # GET directly (HEAD often blocked). Follow redirects.
            r = await client.get(
                url,
                follow_redirects=True,
                headers={**HEADERS, "Referer": _origin_referer(url)},
            )

            # Accept only real 2xx responses
            if not (200 <= r.status_code < 300):
                print(f"❌ {url} → HTTP {r.status_code}, skipping")
                return None

            ct = (r.headers.get("content-type") or "").lower()
            cd = r.headers.get("content-disposition")
            ext = _guess_extension(url, ct, cd)
            out_path, which = _target_paths(base, ext)

            # Cache / freshness
            if not force and _exists_recent(out_path, fresh_hours):
                return (out_path, url)

            # Size guard (if Content-Length present)
            try:
                clen = int(r.headers.get("content-length", "0"))
                if clen and clen > MAX_DOWNLOAD_BYTES:
                    print(f"❌ {url} → Skipped (too large: {clen} bytes)")
                    return None
            except Exception:
                pass

            # Decide how to save
            if ext == ".html":
                text = r.text or ""
                # Block pages / tiny placeholders
                if (BLOCK_HTML_IF_URL_LOOKS_BINARY and _url_looks_binary(url)) or \
                   len(text.encode("utf-8")) < MIN_HTML_BYTES or \
                   _looks_like_block_page(text):
                    print(f"❌ {url} → HTML looks blocked/invalid (len={len(text)}), skipping")
                    return None

                with open(out_path, "w", encoding="utf-8", errors="ignore") as f:
                    f.write(text)
                print(f"✅ {url} → html saved: {os.path.basename(out_path)}")
            else:
                content = r.content or b""
                if len(content) == 0:
                    print(f"❌ {url} → Empty body for binary, skipping")
                    return None
                if len(content) > MAX_DOWNLOAD_BYTES:
                    print(f"❌ {url} → Skipped (download exceeded max size)")
                    return None
                with open(out_path, "wb") as f:
                    f.write(content)
                print(f"✅ {url} → file saved: {os.path.basename(out_path)} ({len(content)} bytes)")

            return (out_path, url)

        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as e:
            if attempt < retries:
                attempt += 1
                continue
            print(f"❌ {url} → Timeout/Protocol error after retries: {e}")
            return None
        except httpx.HTTPError as e:
            print(f"❌ {url} → HTTP error: {e}")
            return None
        except Exception as e:
            print(f"❌ {url} → Unexpected error: {e}")
            return None


# ───────────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────────

async def scrape_and_download(
    urls: List[str],
    fresh_hours: int = 0,
    force: bool = True,
) -> List[str]:
    """
    Download ONLY the provided URLs (no crawling).
    - Saves HTML to data/html, and files (pdf/xlsx/xls/csv/json) to data/files
    - Returns list of saved absolute paths
    - Updates scraping/output/download_manifest.json (filename → URL)
    - Skips non-2xx, blocked 403/“Access Denied” HTML wrappers, and tiny placeholder HTML.
    """
    saved: List[str] = []
    if not urls:
        return saved

    # Normalize input URLs and drop empties/duplicates
    seen_input = set()
    targets = []
    for u in urls:
        if isinstance(u, str):
            uu = u.strip()
            if uu and uu not in seen_input:
                seen_input.add(uu)
                targets.append(uu)
    if not targets:
        return saved

    # Local manifest (merge into global at end)
    local_manifest: Dict[str, str] = {}

    limits = httpx.Limits(max_keepalive_connections=8, max_connections=8)
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers=HEADERS, limits=limits) as client:
        tasks = [_download_with_retry(client, u, fresh_hours, force, retries=1) for u in targets]
        for coro in asyncio.as_completed(tasks):
            res = await coro
            if res:
                path, orig_url = res
                abs_path = os.path.abspath(path)
                saved.append(abs_path)
                # manifest key = filename only (not full path)
                local_manifest[os.path.basename(path)] = orig_url

    # Merge manifest changes (only for successfully saved files)
    if local_manifest:
        manifest = _load_manifest()
        manifest.update(local_manifest)
        _save_manifest(manifest)

    return saved
