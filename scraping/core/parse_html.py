# scraping/core/parse_html.py
from __future__ import annotations

import os
from typing import Dict
from bs4 import BeautifulSoup


def extract_text_from_html(html_dir: str = "data/html", out_dir: str = "data/text") -> Dict[str, int]:
    """
    Convert HTML pages in `html_dir` into plain, analysis-friendly text files in `out_dir`,
    preserving lists and basic table content.

    Returns a small stats dict: {"processed": N, "written": M}
    """
    os.makedirs(out_dir, exist_ok=True)

    processed = 0
    written = 0

    if not os.path.isdir(html_dir):
        # Nothing to do; keep a stable return shape
        return {"processed": 0, "written": 0}

    for fname in os.listdir(html_dir):
        if not fname.endswith(".html"):
            continue
        processed += 1

        fpath = os.path.join(html_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        # Remove boilerplate / non-content
        for tag in soup(["nav", "footer", "script", "style", "header", "aside", "form"]):
            tag.decompose()

        # Preserve bullet points from <ul>
        for ul in soup.find_all("ul"):
            for li in ul.find_all("li"):
                if li.parent:
                    li.insert_before(soup.new_string("• " + li.get_text(" ", strip=True)))
                li.decompose()
            ul.decompose()

        # Preserve enumerated lists from <ol>
        for ol in soup.find_all("ol"):
            for idx, li in enumerate(ol.find_all("li"), 1):
                if li.parent:
                    li.insert_before(soup.new_string(f"{idx}. " + li.get_text(" ", strip=True)))
                li.decompose()
            ol.decompose()

        # Preserve simple table content as line blocks
        for table in soup.find_all("table"):
            table.insert_before(soup.new_string("===TABLE_START==="))
            for row in table.find_all("tr"):
                cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
                if cells:
                    table.insert_before(soup.new_string(" | ".join(cells)))
            table.insert_after(soup.new_string("===TABLE_END==="))
            table.decompose()

        # Drop empty divs/spans
        for tag in soup.find_all(["div", "span"]):
            if not tag.get_text(strip=True):
                tag.decompose()

        # Extract and clean plain text
        text = soup.get_text("\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        out_path = os.path.join(out_dir, fname.replace(".html", ".txt"))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(clean_text)
        written += 1

    return {"processed": processed, "written": written}
