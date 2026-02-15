"""Pydantic models for chat, inspections, and comments."""

from typing import Optional, List
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")
    session_id: Optional[str] = Field(None, description="Existing session ID (auto-generated if omitted)")
    model: Optional[str] = Field("fast", description="Model tier: 'fast' or 'pro'")


class ChatMessage(BaseModel):
    id: int
    session_id: str
    role: str
    content: str
    segment_ids: str = ""
    created_at: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    segment_ids: List[str] = Field(default_factory=list)


class ChatSession(BaseModel):
    id: str
    created_at: str
    title: str = ""
    messages: List[ChatMessage] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Inspections
# ---------------------------------------------------------------------------

class InspectionCreate(BaseModel):
    segment_id: str
    inspector: str = ""
    date: str = Field(..., description="Inspection date (YYYY-MM-DD)")
    weather: str = ""
    condition: str = Field(..., description="good / fair / poor / critical")
    findings: str = ""


class Inspection(BaseModel):
    id: int
    segment_id: str
    inspector: str
    date: str
    weather: str
    condition: str
    findings: str
    created_at: str


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------

class CommentCreate(BaseModel):
    segment_id: str
    author: str = ""
    content: str = Field(..., min_length=1)


class Comment(BaseModel):
    id: int
    segment_id: str
    author: str
    content: str
    created_at: str
