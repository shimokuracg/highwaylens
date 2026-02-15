"""Chat API router — Gemini-powered AI assistant."""

import uuid
import logging
from typing import List

from fastapi import APIRouter, HTTPException
from src.db.database import get_db
from src.db.models import ChatRequest, ChatResponse, ChatSession, ChatMessage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def send_message(body: ChatRequest):
    """Send a message to the AI assistant and receive a response."""
    from src.agent.gemini_agent import GeminiAgent

    session_id = body.session_id or str(uuid.uuid4())

    # Ensure session exists
    with get_db() as conn:
        existing = conn.execute("SELECT id FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
        if not existing:
            title = body.message[:50]
            conn.execute("INSERT INTO chat_sessions (id, title) VALUES (?, ?)", (session_id, title))

    # Load recent history (last 20 messages)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY id DESC LIMIT 20",
            (session_id,),
        ).fetchall()
    history = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    # Save user message
    with get_db() as conn:
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content) VALUES (?, 'user', ?)",
            (session_id, body.message),
        )

    # Call Gemini agent
    agent = GeminiAgent()
    result = await agent.chat(body.message, history, model_tier=body.model or "pro")

    reply_text = result["reply"]
    segment_ids = result.get("segment_ids", [])

    # Save assistant message
    with get_db() as conn:
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, segment_ids) VALUES (?, 'assistant', ?, ?)",
            (session_id, reply_text, ",".join(segment_ids)),
        )

    return ChatResponse(session_id=session_id, reply=reply_text, segment_ids=segment_ids)


@router.get("/sessions", response_model=List[ChatSession])
async def list_sessions():
    """List all chat sessions."""
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM chat_sessions ORDER BY created_at DESC").fetchall()
    return [ChatSession(id=r["id"], created_at=r["created_at"], title=r["title"]) for r in rows]


@router.get("/sessions/{session_id}", response_model=ChatSession)
async def get_session(session_id: str):
    """Get a chat session with its message history."""
    with get_db() as conn:
        sess = conn.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        msgs = conn.execute(
            "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
    return ChatSession(
        id=sess["id"],
        created_at=sess["created_at"],
        title=sess["title"],
        messages=[ChatMessage(**dict(m)) for m in msgs],
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session and its messages."""
    with get_db() as conn:
        conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        cur = conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": session_id}
