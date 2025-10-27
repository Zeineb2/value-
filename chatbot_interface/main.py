# chatbot_interface/main.py
# Minimal app that serves a one-input chat UI and mounts your Agent router at /agent/chat

import os
from pathlib import Path
import logging
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ✅ Mount your existing agent (this exposes /agent/chat)
from agentic.api_agent import router as agent_router

ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(ROOT / "templates"))

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
app.include_router(agent_router)  # ← keeps /agent/chat available

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ui")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})
