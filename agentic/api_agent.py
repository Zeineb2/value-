# agentic/api_agent.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import traceback
import json

from agentic.agent.agent_graph import run_agent, build_agent
from agentic import config as CFG

router = APIRouter(prefix="/agent", tags=["agent"])


class ChatBody(BaseModel):
    query: str
    chat_history: list[dict] | None = None


@router.get("/diag")
def diag():
    """Basic health/info to debug issues quickly."""
    info = {
        "provider": CFG.PROVIDER,
        "faiss_dir": CFG.FAISS_DIR,
        "embed_model": CFG.EMBED_MODEL,
        "embed_device": CFG.EMBED_DEVICE,
        "top_k": CFG.TOP_K,
    }
    try:
        agent = build_agent()
        info["tools"] = [t.name for t in (agent.tools or [])]
        info["agent_ok"] = True
    except Exception as e:
        info["agent_ok"] = False
        info["agent_error"] = str(e)
        info["agent_traceback"] = traceback.format_exc()
    return info


@router.post("/chat")
def chat(body: ChatBody):
    try:
        answer = run_agent(body.query, chat_history=body.chat_history)
        return {"answer": answer}
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=json.dumps(
                {
                    "error": str(e),
                    "traceback": tb[-5000:],  # trim if huge
                }
            ),
        )
