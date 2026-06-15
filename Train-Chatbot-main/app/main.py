from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.conversation import ConversationManager
from app.models import ChatRequest, ChatResponse
from app.storage import ChatStorage


app = FastAPI(title="Railway Chatbot Base", version="0.1.0")
storage = ChatStorage()
conversation_manager = ConversationManager(storage)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return Path("app/static/index.html").read_text(encoding="utf-8")


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return conversation_manager.handle(request.session_id, request.message)
