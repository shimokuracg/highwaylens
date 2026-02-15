"""Agent tool definitions for Gemini Function Calling.

Each tool function takes simple arguments and returns dicts
that are serialised as JSON for the model.
"""

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


def _get_segments():
    """Lazy-import to avoid circular imports at module level."""
    from src.api.main import SAMPLE_SEGMENTS
    return SAMPLE_SEGMENTS


# ---- Tool implementations ------------------------------------------------

def get_segment(segment_id: str) -> dict:
    """Return details for a specific segment."""
    for seg in _get_segments():
        if seg.segment_id == segment_id:
            return {
                "segment_id": seg.segment_id,
                "lat": seg.lat,
                "lon": seg.lon,
                "level": seg.level,
                "message": seg.message,
                "is_reliable": seg.is_reliable,
                "insights": seg.insights,
            }
    return {"error": f"Segment '{segment_id}' not found"}


def search_segments(
    level: Optional[str] = None,
    route: Optional[str] = None,
    min_score: Optional[int] = None,
) -> dict:
    """Search segments by level, route prefix, and/or minimum score. Returns up to 10."""
    results = list(_get_segments())

    if level:
        results = [s for s in results if s.level == level.lower()]
    if route:
        prefix = route.upper()
        results = [s for s in results if s.segment_id.startswith(prefix)]
    if min_score is not None:
        results = [s for s in results if s.score >= min_score]

    results.sort(key=lambda s: s.score, reverse=True)
    results = results[:10]

    return {
        "count": len(results),
        "segments": [
            {
                "segment_id": s.segment_id,
                "level": s.level,
                "lat": s.lat,
                "lon": s.lon,
                "message": s.message,
            }
            for s in results
        ],
    }


def get_stats() -> dict:
    """Return overall monitoring statistics."""
    segments = _get_segments()
    levels: dict[str, int] = {}
    for s in segments:
        levels[s.level] = levels.get(s.level, 0) + 1
    return {
        "total_segments": len(segments),
        "segments_by_level": levels,
        "high_risk_count": sum(1 for s in segments if s.level in ("red", "orange")),
    }


def get_inspections(segment_id: str) -> dict:
    """Return inspection records for a segment."""
    from src.db.database import get_db
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM inspections WHERE segment_id = ? ORDER BY date DESC LIMIT 20",
            (segment_id,),
        ).fetchall()
    return {
        "segment_id": segment_id,
        "count": len(rows),
        "inspections": [dict(r) for r in rows],
    }


def get_comments(segment_id: str) -> dict:
    """Return comments for a segment."""
    from src.db.database import get_db
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM slope_comments WHERE segment_id = ? ORDER BY created_at DESC LIMIT 20",
            (segment_id,),
        ).fetchall()
    return {
        "segment_id": segment_id,
        "count": len(rows),
        "comments": [dict(r) for r in rows],
    }


def web_search(query: str) -> dict:
    """Search the internet via Gemini grounding with Google Search."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return {"error": "No API key configured for search"}

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        candidate = response.candidates[0]
        text = "".join(
            part.text for part in candidate.content.parts
            if part.text
        )

        # Extract source URLs from grounding metadata
        sources = []
        gm = candidate.grounding_metadata
        if gm and gm.grounding_chunks:
            for chunk in gm.grounding_chunks:
                if chunk.web and chunk.web.uri:
                    sources.append({"title": chunk.web.title or "", "url": chunk.web.uri})

        return {"query": query, "result": text, "sources": sources}
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return {"query": query, "error": str(e)}


# ---- Tool registry (name -> callable) ------------------------------------

TOOL_REGISTRY = {
    "get_segment": get_segment,
    "search_segments": search_segments,
    "get_stats": get_stats,
    "get_inspections": get_inspections,
    "get_comments": get_comments,
    "web_search": web_search,
}

# ---- Gemini function declarations ----------------------------------------

TOOL_DECLARATIONS = [
    {
        "name": "get_segment",
        "description": "Get detailed information for a specific monitored slope segment by its ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "segment_id": {
                    "type": "string",
                    "description": "Segment ID, e.g. SHINTOMEI_0014 or TOMEI_0087",
                },
            },
            "required": ["segment_id"],
        },
    },
    {
        "name": "search_segments",
        "description": "Search slope segments by risk level and/or route. Returns up to 10 results.",
        "parameters": {
            "type": "object",
            "properties": {
                "level": {
                    "type": "string",
                    "description": "Risk level filter: green, yellow, orange, or red",
                },
                "route": {
                    "type": "string",
                    "description": "Route prefix: TOMEI or SHINTOMEI",
                },
            },
        },
    },
    {
        "name": "get_stats",
        "description": "Get overall monitoring statistics: total segments, counts by level, high-risk count.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_inspections",
        "description": "Get inspection records for a specific segment.",
        "parameters": {
            "type": "object",
            "properties": {
                "segment_id": {
                    "type": "string",
                    "description": "Segment ID",
                },
            },
            "required": ["segment_id"],
        },
    },
    {
        "name": "get_comments",
        "description": "Get comments posted for a specific segment.",
        "parameters": {
            "type": "object",
            "properties": {
                "segment_id": {
                    "type": "string",
                    "description": "Segment ID",
                },
            },
            "required": ["segment_id"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the internet for information about slope engineering, landslide countermeasures, weather forecasts, geological knowledge, or any topic not covered by HighwayLens's internal data.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query in Japanese or English",
                },
            },
            "required": ["query"],
        },
    },
]
