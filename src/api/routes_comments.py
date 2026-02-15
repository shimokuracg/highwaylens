"""Comment CRUD API router."""

from typing import List
from fastapi import APIRouter, Query, HTTPException
from src.db.database import get_db
from src.db.models import CommentCreate, Comment

router = APIRouter(prefix="/api/v1/comments", tags=["Comments"])


@router.post("", response_model=Comment)
async def add_comment(body: CommentCreate):
    """Add a comment to a segment."""
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO slope_comments (segment_id, author, content) VALUES (?, ?, ?)",
            (body.segment_id, body.author, body.content),
        )
        row = conn.execute("SELECT * FROM slope_comments WHERE id = ?", (cur.lastrowid,)).fetchone()
    return Comment(**dict(row))


@router.get("", response_model=List[Comment])
async def get_comments(segment_id: str = Query(..., description="Segment ID")):
    """Get comments for a segment."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM slope_comments WHERE segment_id = ? ORDER BY created_at DESC",
            (segment_id,),
        ).fetchall()
    return [Comment(**dict(r)) for r in rows]


@router.delete("/{comment_id}")
async def delete_comment(comment_id: int):
    """Delete a comment."""
    with get_db() as conn:
        cur = conn.execute("DELETE FROM slope_comments WHERE id = ?", (comment_id,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Comment not found")
    return {"deleted": comment_id}
