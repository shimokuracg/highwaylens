"""Inspection records CRUD API router."""

from typing import List
from fastapi import APIRouter, Query, HTTPException
from src.db.database import get_db
from src.db.models import InspectionCreate, Inspection

router = APIRouter(prefix="/api/v1/inspections", tags=["Inspections"])


@router.post("", response_model=Inspection)
async def create_inspection(body: InspectionCreate):
    """Register an inspection record."""
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO inspections (segment_id, inspector, date, weather, condition, findings) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (body.segment_id, body.inspector, body.date, body.weather, body.condition, body.findings),
        )
        row = conn.execute("SELECT * FROM inspections WHERE id = ?", (cur.lastrowid,)).fetchone()
    return Inspection(**dict(row))


@router.get("", response_model=List[Inspection])
async def get_inspections(segment_id: str = Query(..., description="Segment ID")):
    """Get inspection records for a segment."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM inspections WHERE segment_id = ? ORDER BY date DESC",
            (segment_id,),
        ).fetchall()
    return [Inspection(**dict(r)) for r in rows]


@router.get("/all", response_model=List[Inspection])
async def get_all_inspections(limit: int = Query(50, ge=1, le=200)):
    """全セグメント横断の点検記録一覧（新しい順）"""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM inspections ORDER BY date DESC, created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [Inspection(**dict(r)) for r in rows]


@router.delete("/{inspection_id}")
async def delete_inspection(inspection_id: int):
    """Delete an inspection record."""
    with get_db() as conn:
        cur = conn.execute("DELETE FROM inspections WHERE id = ?", (inspection_id,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Inspection not found")
    return {"deleted": inspection_id}
