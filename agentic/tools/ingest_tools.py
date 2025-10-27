# agentic/tools/ingest_tools.py
from __future__ import annotations

import os
from urllib.parse import urlparse
from typing import List

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from agentic.config import FAISS_DIR, EMBED_MODEL, EMBED_DEVICE, REQUEST_TIMEOUT

_embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": EMBED_DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

def _ensure_index():
    if os.path.isdir(FAISS_DIR):
        return FAISS.load_local(FAISS_DIR, _embeddings, allow_dangerous_deserialization=True)
    return FAISS.from_texts(["__bootstrap__"], _embeddings)

def _readable_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    return " ".join(soup.get_text(separator=" ").split())

def _is_http_url(url: str) -> bool:
    try:
        p = urlparse(url)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False

@tool("ingest_url")
def ingest_url(url: str) -> str:
    """
    Download/scrape ONE URL (HTML; PDFs/XLSX should be handled by your dedicated scraper),
    chunk it, and add to FAISS. Returns a short status message.
    """
    url = (url or "").strip()
    if not _is_http_url(url):
        return "ERROR: Request URL is missing an 'http://' or 'https://' protocol."

    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
            r = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
        text = _readable_text(r.text)
        if not text or len(text) < 300:
            return "Fetched but text too short — skipped."

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=120, add_start_index=True
        )
        docs = splitter.create_documents([text], metadatas=[{"source": url}])

        vs = _ensure_index()
        vs.add_documents(docs)
        vs.save_local(FAISS_DIR)

        return f"Ingested {len(docs)} chunks from {url}."
    except httpx.TimeoutException:
        return "ERROR: ingest_url failed → network timeout."
    except httpx.RequestError as e:
        return f"ERROR: ingest_url failed → network error: {e}"
    except Exception as e:
        return f"ERROR: ingest_url failed → {e}"
