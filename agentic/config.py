# agentic/config.py
from __future__ import annotations
import os

# ============ LLM / Provider ============
PROVIDER: str = os.getenv("PROVIDER", "ollama").strip().lower()

# Ollama
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# OpenAI (non-Azure)
OPENAI_MODEL: str | None = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

# Azure OpenAI
AZURE_OPENAI_API_KEY: str | None = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT: str | None = os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g. https://<name>.openai.azure.com
AZURE_OPENAI_DEPLOYMENT: str | None = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Agent guardrails
MAX_TOOL_STEPS: int = int(os.getenv("MAX_TOOL_STEPS", "8"))

# ============ Embeddings / FAISS ============
FAISS_DIR: str = os.getenv("FAISS_DIR", "vectorization/faiss_index")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
EMBED_DEVICE: str = os.getenv("EMBED_DEVICE", "cpu")  # "cpu" or "cuda"
TOP_K: int = int(os.getenv("TOP_K", "6"))

# ============ Scraping / Pipelines ============
SERPER_API_KEY: str | None = os.getenv("SERPER_API_KEY")
SERPER_COUNTRY: str = os.getenv("SERPER_COUNTRY", "tn")

# HTTP & scraping defaults used by ingest/pipeline tools
REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "90"))         # seconds
SCRAPE_CONCURRENCY: int = int(os.getenv("SCRAPE_CONCURRENCY", "4"))    # optional, if used
DOWNLOAD_DIR: str = os.getenv("DOWNLOAD_DIR", "data/files")            # optional, if used
MAX_DOWNLOAD_MB: int = int(os.getenv("MAX_DOWNLOAD_MB", "25"))         # optional, if used

# Debug print (optional)
if os.getenv("CONFIG_DEBUG", "").lower() in {"1", "true", "yes"}:
    print("[agentic.config] PROVIDER=", PROVIDER)
    print("[agentic.config] OLLAMA_MODEL=", OLLAMA_MODEL)
    print("[agentic.config] OPENAI_MODEL=", OPENAI_MODEL)
    print("[agentic.config] AZURE_OPENAI_DEPLOYMENT=", AZURE_OPENAI_DEPLOYMENT)
    print("[agentic.config] FAISS_DIR=", FAISS_DIR)
    print("[agentic.config] EMBED_MODEL=", EMBED_MODEL)
    print("[agentic.config] EMBED_DEVICE=", EMBED_DEVICE)
    print("[agentic.config] TOP_K=", TOP_K)
    print("[agentic.config] REQUEST_TIMEOUT=", REQUEST_TIMEOUT)
