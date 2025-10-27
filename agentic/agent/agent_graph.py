# agentic/agent/agent_graph.py
from __future__ import annotations
from typing import Any, List
import json

# === LLM backends (choose via config) ===
from agentic.config import (
    PROVIDER,
    OLLAMA_MODEL,
    OPENAI_MODEL,
    OPENAI_API_KEY,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    MAX_TOOL_STEPS,
)

# Optional deps (import if available)
try:
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception:
    ChatOpenAI = None  # type: ignore

try:
    from langchain_ollama import ChatOllama  # type: ignore
except Exception:
    ChatOllama = None  # type: ignore

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent

# === Tools ===
# We'll import the module so we can call the original function safely.
import agentic.tools.vector_tools as vt
from agentic.tools.hybrid_ingest import hybrid_ingest  # @tool

# Optional: URL picker (also @tool-decorated)
_pick_urls_tool = None
try:
    from agentic.tools.url_pick import pick_urls_tool as _pick_urls_tool  # @tool
except Exception:
    _pick_urls_tool = None  # not critical

# Provide a robust wrapper that the agent will see as "vector_search"
try:
    from langchain.tools import tool
except Exception:
    from langchain_core.tools import tool  # type: ignore


def _coerce_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


@tool("vector_search", return_direct=False)
def vector_search_safe(query: str = "", k: str | int = "") -> str:
    """
    Safe semantic search over the local FAISS store.
    - `query` may be omitted; if blank, returns {"hits": []} instead of raising.
    - `k` is coerced from str/int; defaults to module's DEFAULT_TOP_K.
    """
    if not isinstance(query, str):
        query = str(query or "")
    if not query.strip():
        # Avoid Pydantic errors when the model forgets to pass query
        return json.dumps({"hits": []}, ensure_ascii=False)

    kval = _coerce_int(k, getattr(vt, "DEFAULT_TOP_K", 6))
    try:
        # delegate to the original tool's underlying function
        # (the @tool in langchain exposes .func for the wrapped callable)
        return vt.vector_search.func(query=query, k=kval)  # type: ignore[attr-defined]
    except Exception as e:
        # As a last resort, report no hits (so the agent proceeds to web ingest)
        return json.dumps({"hits": [], "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)


def _get_llm():
    """
    Returns a ChatModel according to PROVIDER in config.
    Supported: 'ollama' (default), 'openai', 'azure'
    """
    provider = (PROVIDER or "ollama").lower()

    if provider == "openai":
        if ChatOpenAI is None:
            raise RuntimeError("langchain-openai not installed")
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing in environment/config")
        return ChatOpenAI(model=OPENAI_MODEL or "gpt-4o-mini", temperature=0)

    if provider == "azure":
        if ChatOpenAI is None:
            raise RuntimeError("langchain-openai not installed")
        if not (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT):
            raise RuntimeError("Azure OpenAI credentials incomplete in config")
        return ChatOpenAI(
            model=AZURE_OPENAI_DEPLOYMENT,
            base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}",
            api_key=AZURE_OPENAI_API_KEY,
            temperature=0,
        )

    # default: ollama
    if ChatOllama is None:
        raise RuntimeError("langchain-ollama not installed")
    return ChatOllama(model=OLLAMA_MODEL or "llama3.1:8b", temperature=0)


def _build_prompt():
    """
    Strong, explicit orchestration for Scenario 2:
    - Try FAISS (vector_search)
    - If insufficient, run hybrid_ingest with allow_discovery=False (OFFICIAL-ONLY)
      then re-query FAISS
    - If still nothing, run hybrid_ingest with allow_discovery=True (BROAD)
      then re-query FAISS
    """
    system = (
        "You answer quantitative indicator questions using a local FAISS knowledge base. "
        "If FAISS is missing or stale, you MUST discover the exact indicator online, ingest it, "
        "and re-query FAISS.\n\n"
        "Step A — FAISS:\n"
        "  • Call vector_search(query, k). ALWAYS pass both `query` and `k`.\n"
        "  • If the user asks for latest/current/date of last change/YoY, only accept hits that are"
        "    recent (~last 18 months) or from official domains (ins.tn, bct.gov.tn, imf.org/data.imf.org,"
        "    worldbank.org/databank.worldbank.org, oecd.org, afdb.org). If no acceptable hits, proceed.\n\n"
        "Step B — Web ingest (two attempts):\n"
        "  B1) Call hybrid_ingest(question, allow_discovery=False)  # OFFICIAL-ONLY. "
        "      Then call vector_search(query, k) again. If still no usable hit:\n"
        "  B2) Call hybrid_ingest(question, allow_discovery=True)   # BROAD DISCOVERY. "
        "      Then call vector_search(query, k) again.\n\n"
        "Step C — Answer:\n"
        "  • Provide a single numeric value (if applicable), the reference date (month/quarter/year), and "
        "    the official source URL. Never invent numbers or dates. If still unknown after Step B, state that "
        "    you could not find a trustworthy value.\n\n"
        "IMPORTANT:\n"
        "  • Use EXACT tool signatures: vector_search(query, k) and hybrid_ingest(question, allow_discovery).\n"
        "  • Do NOT call tools with missing parameters."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    return prompt


def build_agent() -> AgentExecutor:
    llm = _get_llm()
    prompt = _build_prompt()

    # Only include @tool-decorated objects
    # IMPORTANT: expose our SAFE wrapper *as* 'vector_search'
    tools = [vector_search_safe, hybrid_ingest]
    # Optional URL picker (helps LLM see discovery capability explicitly)
    if _pick_urls_tool is not None:
        tools.insert(1, _pick_urls_tool)

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

    max_iters = MAX_TOOL_STEPS if isinstance(MAX_TOOL_STEPS, int) and MAX_TOOL_STEPS > 0 else 8

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=max_iters,
        return_intermediate_steps=False,
    )
    return executor


def run_agent(query: str, chat_history: List[dict] | None = None) -> str:
    """Synchronous single-turn run. Returns the final string answer."""
    agent = build_agent()
    result = agent.invoke({"input": query, "chat_history": chat_history or []})
    return result.get("output", "").strip()
