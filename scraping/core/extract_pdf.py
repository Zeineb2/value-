# scraping/core/extract_pdf.py
from __future__ import annotations

import os
import fitz  # PyMuPDF
import time
import pdfplumber
from pdf2image import convert_from_path
import pytesseract

from .extract_text import extract_sentences  # uses taxonomy auto-update internally
from ..utils.indicator_matcher import match_indicators


def is_scanned_pdf(doc):
    return not doc.is_reflowable


def extract_tables_with_pdfplumber(pdf_path, indicators):
    text_blocks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    print(f"📊 Table found on page {page_num + 1}")
                    for row in table:
                        if not row:
                            continue
                        row_text = " | ".join(cell.strip() if cell else "" for cell in row)
                        if match_indicators(row_text, indicators):
                            text_blocks.append(row_text)
            if text_blocks:
                for preview in text_blocks[:3]:
                    print("🧾 Table Row Preview:", preview)
    except Exception as e:
        print(f"⚠️ Failed to parse tables with pdfplumber: {e}")
    return "\n".join(text_blocks)


def extract_text_with_ocr(pdf_path):
    print("🔁 Performing OCR on scanned PDF...")
    try:
        images = convert_from_path(pdf_path, dpi=300)
        ocr_text = "\n".join(pytesseract.image_to_string(img, lang="eng") for img in images)
        return ocr_text
    except Exception as e:
        print(f"❌ OCR failed: {e}")
        return ""


def extract_text_with_fitz(doc):
    full_text = ""
    for page in doc:
        page_text = page.get_text("text")
        if page_text:
            full_text += page_text.replace("\n", " ").strip() + " "
    return full_text


def extract_from_pdfs(indicators, folder):
    results = []
    os.makedirs("output", exist_ok=True)

    for filename in os.listdir(folder):
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(folder, filename)
        print(f"\n📄 Processing file: {filename}")
        start = time.time()

        try:
            doc = fitz.Document(pdf_path)
            scanned = is_scanned_pdf(doc)
            full_text = extract_text_with_fitz(doc)

            if len(full_text) < 500 or scanned:
                print(f"ℹ️ Low text or scanned detected. Trying table extraction with pdfplumber for {filename}")
                table_text = extract_tables_with_pdfplumber(pdf_path, indicators)
                full_text += "\n" + table_text

            if not full_text.strip():
                ocr_text = extract_text_with_ocr(pdf_path)
                full_text += "\n" + ocr_text

            if not full_text.strip():
                print(f"⚠️ No usable text found in: {filename}")
                continue

            print("📌 Sample text preview:")
            print(full_text[:300], "...\n")

            # IMPORTANT: extract_sentences already handles taxonomy auto-grow + alias map rebuild per file
            extracted = extract_sentences(full_text, indicators, filename)
            print(f"✅ {len(extracted)} sentences extracted from {filename}")
            results.extend(extracted)
            print(f"✅ Finished {filename} in {time.time() - start:.2f}s")

        except fitz.FileDataError as e:
            print(f"⚠️ Skipping problematic PDF layer in {filename}: {e}")
        except Exception as e:
            print(f"❌ PDF extract failed {filename}: {e}")

    return results
