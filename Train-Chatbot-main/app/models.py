from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Intent = Literal["ticket_search", "delay_prediction", "arrival_time_prediction", "contingency_advice", "unknown"]
RiskLevel = Literal["low", "medium", "high"]


class ChatRequest(BaseModel):
    session_id: str = Field(default="", description="Client-generated or server-generated session id")
    message: str = Field(..., min_length=1, description="User message")


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: Intent
    state: str
    slots: dict[str, Any]
    complete: bool = False
    prediction: dict[str, Any] | None = None
