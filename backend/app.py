"""
FastAPI backend.

Endpoints:
    POST /chat          — send a message, get a character response
    GET  /characters    — list available personas

Requires ANTHROPIC_API_KEY in environment (or a .env file at project root).
Run with:
    uvicorn backend.app:app --reload
"""

import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.retriever import retrieve, format_context
from backend.persona import get_system_prompt, list_characters

load_dotenv(Path(__file__).parent.parent / ".env")

app = FastAPI(title="Percy RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "src"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── request / response models ────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str   # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    character: str = "percy"
    history: list[ChatMessage] = []

class SourceDoc(BaseModel):
    source: str
    score: float

class ChatResponse(BaseModel):
    reply: str
    character: str
    sources: list[SourceDoc]


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def serve_index():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"status": "Percy RAG API running. Open /docs for the API explorer."}


@app.get("/characters")
def characters():
    return {"characters": list_characters()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set.")

    # 1. Retrieve relevant wiki context
    hits = retrieve(req.message, top_k=5)
    context = format_context(hits)

    # 2. Build system prompt with persona + context
    system_prompt = get_system_prompt(req.character, context)

    # 3. Build message list (history + current turn)
    messages = [{"role": m.role, "content": m.content} for m in req.history]
    messages.append({"role": "user", "content": req.message})

    # 4. Call Claude
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
    )

    reply = response.content[0].text
    sources = [SourceDoc(source=h["source"], score=round(h["score"], 3)) for h in hits]

    return ChatResponse(reply=reply, character=req.character, sources=sources)
