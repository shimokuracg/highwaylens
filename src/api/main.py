"""
HighwayLens REST API v2
=======================

FastAPI application providing REST endpoints for the HighwayLens system.
v2 additions: AI chat, inspections, comments.

Run:
    uvicorn src.api.main:app --reload
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from src.api.auth import (
    auth_middleware, verify_password, set_session_cookie, clear_session_cookie,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    from src.db.database import init_db
    init_db()
    logger.info("HighwayLens API v2 starting up...")
    yield
    logger.info("HighwayLens API v2 shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="HighwayLens API",
    description="Highway Infrastructure Monitoring System API — v2 with AI chat",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication middleware
@app.middleware("http")
async def _auth_middleware(request: Request, call_next):
    return await auth_middleware(request, call_next)

# Mount v2 routers
from src.api.routes_chat import router as chat_router
from src.api.routes_inspections import router as inspections_router
from src.api.routes_comments import router as comments_router
app.include_router(chat_router)
app.include_router(inspections_router)
app.include_router(comments_router)


# ============================================================================
# Pydantic Models
# ============================================================================

class SegmentBase(BaseModel):
    """Base segment model"""
    segment_id: str = Field(..., description="Unique segment identifier")
    lat: float = Field(..., description="Latitude of segment center")
    lon: float = Field(..., description="Longitude of segment center")


class RiskScore(SegmentBase):
    """Risk score for a segment"""
    score: int = Field(..., ge=0, le=100, description="Risk score (0-100)")
    level: str = Field(..., description="Risk level: green, yellow, orange, red")
    message: str = Field("", description="Human-readable risk message")
    coherence: float = Field(1.0, description="InSAR data quality (0-1)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    is_reliable: bool = Field(True, description="Whether result is reliable")
    insights: List[dict] = Field(default_factory=list, description="Data-driven insights")


class Alert(BaseModel):
    """Alert model"""
    alert_id: str = Field(..., description="Unique alert identifier")
    segment_id: str = Field(..., description="Related segment ID")
    level: str = Field(..., description="Alert level")
    message: str = Field(..., description="Alert message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    acknowledged: bool = Field(False)


class Statistics(BaseModel):
    """System statistics"""
    total_segments: int
    segments_by_level: dict
    average_score: float
    high_risk_count: int
    last_updated: datetime


# ============================================================================
# Dynamic Data Generation
# ============================================================================

def generate_segments():
    """Load segments from deformation GeoJSON files with real InSAR data.

    Monitors two highways:
    1. Tomei Expressway (東名) - tomei_deformation.geojson
    2. Shin-Tomei Expressway (新東名) - shintomei_deformation.geojson
    """
    import json

    try:
        from src.analytics.risk_calculator import RiskScoreCalculator, SegmentData
        from src.data_acquisition.real_data_loader import (
            get_real_terrain, get_real_weather, get_real_geology
        )

        calculator = RiskScoreCalculator()
        results = []
        data_dir = Path(__file__).parent.parent.parent / "data" / "processed"

        # =====================================================================
        # Part 1: Tomei Expressway (東名)
        # =====================================================================
        tomei_path = data_dir / "tomei_deformation.geojson"
        if tomei_path.exists():
            with open(tomei_path) as f:
                data = json.load(f)
            for feature in data['features']:
                props = feature['properties']
                coords = feature['geometry']['coordinates']
                extra_info = {}
                if props.get('ic_from') and props.get('ic_to'):
                    extra_info['ic_section'] = f"{props['ic_from']} → {props['ic_to']}"
                result = _build_segment_from_deformation(
                    segment_id=props['segment_id'],
                    lat=coords[1], lon=coords[0],
                    deformation_rate=props.get('Deformation_Rate1'),
                    calculator=calculator,
                    get_real_terrain=get_real_terrain,
                    get_real_weather=get_real_weather,
                    get_real_geology=get_real_geology,
                    highway_name="東名",
                    extra_info=extra_info if extra_info else None,
                )
                results.append(result)
            logger.info(f"[東名] Loaded {len(data['features'])} segments from deformation GeoJSON")
        else:
            logger.warning(f"Tomei deformation file not found: {tomei_path}")

        # =====================================================================
        # Part 2: Shin-Tomei Expressway (新東名)
        # =====================================================================
        shintomei_path = data_dir / "shintomei_deformation.geojson"
        if shintomei_path.exists():
            with open(shintomei_path) as f:
                data = json.load(f)
            for feature in data['features']:
                props = feature['properties']
                coords = feature['geometry']['coordinates']
                extra_info = {}
                if props.get('tiers'):
                    extra_info['tiers'] = props['tiers']
                    extra_info['length_m'] = props.get('length_m', 0)
                if props.get('ic_from') and props.get('ic_to'):
                    extra_info['ic_section'] = f"{props['ic_from']} → {props['ic_to']}"
                if props.get('direction'):
                    extra_info['direction'] = props['direction']
                if props.get('kp_center'):
                    extra_info['kp_center'] = props['kp_center']
                tier_bonus = min((props.get('tiers', 1) - 1) * 3, 15)
                fallback_slope = min(15 + (props.get('tiers', 1) * 4), 45)

                result = _build_segment_from_deformation(
                    segment_id=props.get('slope_id', props.get('segment_id')),
                    lat=coords[1], lon=coords[0],
                    deformation_rate=props.get('Deformation_Rate1'),
                    calculator=calculator,
                    get_real_terrain=get_real_terrain,
                    get_real_weather=get_real_weather,
                    get_real_geology=get_real_geology,
                    highway_name="新東名",
                    extra_info=extra_info,
                    score_adjustment=tier_bonus,
                    fallback_slope=fallback_slope,
                )
                results.append(result)
            logger.info(f"[新東名] Loaded {len(data['features'])} segments from deformation GeoJSON")
        else:
            logger.warning(f"Shin-Tomei deformation file not found: {shintomei_path}")

        logger.info(f"Total monitoring segments: {len(results)} (東名 + 新東名)")
        return sorted(results, key=lambda x: x.score, reverse=True)

    except Exception as e:
        logger.error(f"Error generating segments: {e}")
        import traceback
        traceback.print_exc()
        return [
            RiskScore(segment_id="ERROR_0001", lat=35.62, lon=138.92, score=25, level="green", message="Error loading data"),
        ]


def _build_segment_from_deformation(
    segment_id: str,
    lat: float,
    lon: float,
    deformation_rate: float,
    calculator,
    get_real_terrain,
    get_real_weather,
    get_real_geology,
    highway_name: str = "",
    extra_info: dict = None,
    score_adjustment: int = 0,
    fallback_slope: float = None,
) -> RiskScore:
    """Calculate risk for a segment using real InSAR deformation rate."""

    from src.analytics.risk_calculator import SegmentData

    # --- No SAR data: partial risk assessment using terrain/weather/geology ---
    if deformation_rate is None:
        terrain = get_real_terrain(lat, lon, fallback_slope=fallback_slope)
        weather = get_real_weather(lat, lon)
        geology = get_real_geology(lat, lon)

        # Calculate component scores using the calculator's scoring methods
        slope_score = calculator._score_slope(terrain["slope_deg"])
        rainfall_score = calculator._score_rainfall(weather["rainfall_48h_mm"], weather.get("rainfall_7d_mm", 0))
        geology_score = calculator._score_geology(geology["geology"])

        # Renormalize weights for available components (total 0.45 → 1.0)
        raw_weights = {"slope": 0.15, "rainfall": 0.15, "geology": 0.15}
        w_total = sum(raw_weights.values())  # 0.45
        partial_score = (
            slope_score * raw_weights["slope"]
            + rainfall_score * raw_weights["rainfall"]
            + geology_score * raw_weights["geology"]
        ) / w_total

        adjusted_score = min(100, max(0, int(partial_score) + score_adjustment))
        if adjusted_score >= 76:
            level = "red"
        elif adjusted_score >= 51:
            level = "orange"
        elif adjusted_score >= 26:
            level = "yellow"
        else:
            level = "green"

        # Build insights — SAR unavailable + available data insights
        partial_insights = []
        partial_insights.append({
            "icon": "📡", "label": "衛星観測",
            "value": "SAR データ未取得",
            "detail": "地形・気象・地質による暫定評価",
            "severity": "medium"
        })

        # Terrain insight
        if terrain["slope_deg"] > 35:
            partial_insights.append({"icon": "⛰️", "label": "地形", "value": "急傾斜地", "detail": f"傾斜角 {terrain['slope_deg']:.0f}°", "severity": "high"})
        elif terrain["slope_deg"] > 20:
            partial_insights.append({"icon": "⛰️", "label": "地形", "value": "やや急な傾斜", "detail": f"傾斜角 {terrain['slope_deg']:.0f}°", "severity": "medium"})
        else:
            partial_insights.append({"icon": "⛰️", "label": "地形", "value": "緩やかな傾斜", "detail": f"傾斜角 {terrain['slope_deg']:.0f}°", "severity": "low"})

        # Rainfall insight (AMeDAS real-time)
        rain_24h = weather.get("rainfall_24h_mm", weather["rainfall_48h_mm"])
        station_name = weather.get("station_name", "")
        if "取得失敗" in weather.get("source", ""):
            partial_insights.append({"icon": "🌧️", "label": "気象", "value": "AMeDAS取得失敗", "detail": f"観測局: {station_name}", "severity": "low"})
        elif rain_24h > 80:
            partial_insights.append({"icon": "🌧️", "label": "気象", "value": "大雨の影響あり", "detail": f"24時間 {rain_24h:.0f}mm ({station_name})", "severity": "high"})
        elif rain_24h > 30:
            partial_insights.append({"icon": "🌧️", "label": "気象", "value": "降雨の影響あり", "detail": f"24時間 {rain_24h:.0f}mm ({station_name})", "severity": "medium"})
        elif rain_24h > 0:
            partial_insights.append({"icon": "🌧️", "label": "気象", "value": "小雨", "detail": f"24時間 {rain_24h:.0f}mm ({station_name})", "severity": "low"})
        else:
            partial_insights.append({"icon": "☀️", "label": "気象", "value": "降雨なし", "detail": f"24時間 0mm ({station_name})", "severity": "low"})

        # Geology insight
        from src.analytics.risk_calculator import GEOLOGY_NAMES_JA
        geo_type = geology["geology"]
        geo_name = GEOLOGY_NAMES_JA.get(geo_type, geo_type)
        geo_risk = calculator.GEOLOGY_RISK.get(geo_type, 50)
        partial_insights.append({"icon": "🪨", "label": "地質", "value": geo_name, "detail": "地質調査データより", "severity": "high" if geo_risk > 60 else "medium" if geo_risk > 40 else "low"})

        if extra_info:
            if extra_info.get('tiers'):
                partial_insights.append({
                    "icon": "🏗️", "label": "法面構造",
                    "value": f"{extra_info['tiers']}段の法面",
                    "detail": f"延長 {extra_info['length_m']:.0f}m" if extra_info.get('length_m') else "",
                    "severity": "high" if extra_info['tiers'] >= 5 else "medium" if extra_info['tiers'] >= 3 else "low"
                })
            if extra_info.get('ic_section'):
                partial_insights.append({"icon": "🛣️", "label": "区間", "value": extra_info['ic_section'], "severity": "low"})
            if extra_info.get('direction'):
                partial_insights.append({"icon": "🔄", "label": "方向", "value": extra_info['direction'], "severity": "low"})
            if extra_info.get('kp_center'):
                partial_insights.append({"icon": "📍", "label": "キロポスト", "value": f"KP {extra_info['kp_center']:.1f}", "severity": "low"})

        source_tag = f" [{highway_name}: DEM+AMeDAS+地質, SAR待ち]"

        return RiskScore(
            segment_id=segment_id,
            lat=lat, lon=lon,
            score=adjusted_score,
            level=level,
            message=f"{'【暫定】' if adjusted_score >= 26 else '【正常・暫定】'}SAR未取得・他データで評価{source_tag}",
            coherence=0.0,
            timestamp=datetime.utcnow(),
            is_reliable=False,
            insights=partial_insights,
        )

    # --- Has observation data: full risk calculation ---
    velocity_mm_year = deformation_rate * 1000
    coherence = 0.7

    terrain = get_real_terrain(lat, lon, fallback_slope=fallback_slope)
    weather = get_real_weather(lat, lon)
    geology = get_real_geology(lat, lon)

    acceleration = 0.0

    segment_data = SegmentData(
        segment_id=segment_id,
        lat=lat,
        lon=lon,
        deformation_rate_mm_year=velocity_mm_year,
        deformation_acceleration=acceleration,
        slope_angle_degrees=terrain["slope_deg"],
        rainfall_48h_mm=weather["rainfall_48h_mm"],
        rainfall_7d_mm=weather.get("rainfall_7d_mm", 0),
        geology_type=geology["geology"],
        coherence=coherence,
    )

    result = calculator.calculate(segment_data)

    adjusted_score = min(100, max(0, result.score + score_adjustment))
    if adjusted_score >= 76:
        level = "red"
    elif adjusted_score >= 51:
        level = "orange"
    elif adjusted_score >= 26:
        level = "yellow"
    else:
        level = "green"

    # Data sources
    source_tag = f" [{highway_name}: SAR+DEM+AMeDAS+地質]"

    # Insights — risk calculator already generates 衛星観測 insight from deformation_rate
    insights = result.insights or []
    if extra_info:
        if extra_info.get('tiers'):
            insights.append({
                "icon": "🏗️", "label": "法面構造",
                "value": f"{extra_info['tiers']}段の法面",
                "detail": f"延長 {extra_info['length_m']:.0f}m" if extra_info.get('length_m') else "",
                "severity": "high" if extra_info['tiers'] >= 5 else "medium" if extra_info['tiers'] >= 3 else "low"
            })
        if extra_info.get('ic_section'):
            insights.append({
                "icon": "🛣️", "label": "区間",
                "value": extra_info['ic_section'],
                "severity": "low"
            })
        if extra_info.get('direction'):
            insights.append({"icon": "🔄", "label": "方向", "value": extra_info['direction'], "severity": "low"})
        if extra_info.get('kp_center'):
            insights.append({"icon": "📍", "label": "キロポスト", "value": f"KP {extra_info['kp_center']:.1f}", "severity": "low"})

    return RiskScore(
        segment_id=segment_id,
        lat=lat, lon=lon,
        score=adjusted_score,
        level=level,
        message=result.message + source_tag,
        coherence=coherence,
        timestamp=datetime.utcnow(),
        is_reliable=deformation_rate is not None,
        insights=insights
    )


def get_current_segments():
    """Get cached or fresh segments."""
    global _cached_segments, _cache_time
    now = datetime.utcnow()

    # Refresh cache every 60 seconds
    if '_cached_segments' not in globals() or (now - _cache_time).seconds > 60:
        _cached_segments = generate_segments()
        _cache_time = now

    return _cached_segments


_cached_segments = None
_cache_time = datetime.utcnow()

# Initialize on import
SAMPLE_SEGMENTS = generate_segments()

SAMPLE_ALERTS = []
for seg in SAMPLE_SEGMENTS:
    if seg.level in ['orange', 'red']:
        SAMPLE_ALERTS.append(Alert(
            alert_id=f"ALERT_{seg.segment_id}",
            segment_id=seg.segment_id,
            level=seg.level,
            message=seg.message,
            timestamp=datetime.utcnow()
        ))


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """API root - basic info"""
    return {
        "name": "HighwayLens API",
        "version": "0.1.0",
        "description": "Highway Slope Failure Prediction System",
        "docs": "/docs",
        "status": "operational"
    }


@app.get("/health", tags=["Info"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/v1/segments", response_model=List[RiskScore], tags=["Segments"])
async def get_all_segments(
    level: Optional[str] = Query(None, description="Filter by risk level"),
    min_score: Optional[int] = Query(None, ge=0, le=100, description="Minimum risk score"),
    limit: Optional[int] = Query(None, ge=1, le=1000, description="Maximum results")
):
    """
    Get risk scores for all monitored segments.

    Optional filters:
    - level: Filter by risk level (green, yellow, orange, red)
    - min_score: Only return segments with score >= min_score
    - limit: Maximum number of results
    """
    results = SAMPLE_SEGMENTS.copy()

    # Apply filters
    if level:
        results = [s for s in results if s.level == level.lower()]

    if min_score is not None:
        results = [s for s in results if s.score >= min_score]

    # Sort by score descending
    results.sort(key=lambda x: x.score, reverse=True)

    # Apply limit
    if limit:
        results = results[:limit]

    return results


@app.get("/api/v1/segments/{segment_id}", response_model=RiskScore, tags=["Segments"])
async def get_segment(segment_id: str):
    """Get risk score for a specific segment."""
    for segment in SAMPLE_SEGMENTS:
        if segment.segment_id == segment_id:
            return segment

    raise HTTPException(status_code=404, detail=f"Segment '{segment_id}' not found")


@app.get("/api/v1/alerts", response_model=List[Alert], tags=["Alerts"])
async def get_alerts(
    level: Optional[str] = Query(None, description="Filter by alert level"),
    acknowledged: Optional[bool] = Query(None, description="Filter by acknowledged status")
):
    """
    Get active alerts.

    Optional filters:
    - level: Filter by alert level
    - acknowledged: Filter by acknowledged status
    """
    results = SAMPLE_ALERTS.copy()

    if level:
        results = [a for a in results if a.level == level.lower()]

    if acknowledged is not None:
        results = [a for a in results if a.acknowledged == acknowledged]

    return results


@app.get("/api/v1/geojson", tags=["GeoJSON"])
async def get_geojson():
    """
    Get all segments as GeoJSON FeatureCollection.

    Useful for map display in frontend applications.
    """
    features = []

    for segment in SAMPLE_SEGMENTS:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [segment.lon, segment.lat]
            },
            "properties": {
                "segment_id": segment.segment_id,
                "score": segment.score,
                "level": segment.level,
                "message": segment.message,
                "coherence": segment.coherence,
                "timestamp": segment.timestamp.isoformat()
            }
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "total_features": len(features),
            "generated_at": datetime.utcnow().isoformat()
        }
    }


@app.get("/api/v1/stats", response_model=Statistics, tags=["Statistics"])
async def get_statistics():
    """Get system statistics."""
    segments = SAMPLE_SEGMENTS

    levels = {}
    for s in segments:
        levels[s.level] = levels.get(s.level, 0) + 1

    avg_score = sum(s.score for s in segments) / len(segments) if segments else 0
    high_risk = sum(1 for s in segments if s.score >= 75)

    return Statistics(
        total_segments=len(segments),
        segments_by_level=levels,
        average_score=round(avg_score, 1),
        high_risk_count=high_risk,
        last_updated=datetime.utcnow()
    )


@app.get("/api/v1/routes", tags=["Geometry"])
async def get_expressway_routes():
    """Get expressway route coordinates for map display."""
    from src.data_acquisition.osm_loader import get_tomei_coords, get_shintomei_coords

    routes = {}
    tomei = get_tomei_coords()
    if tomei:
        routes["tomei"] = {
            "name": "東名高速道路",
            "name_en": "Tomei Expressway",
            "ref": "E1",
            "coordinates": [[c[1], c[0]] for c in tomei],  # [lat, lon] for Leaflet
        }
    shintomei = get_shintomei_coords()
    if shintomei:
        routes["shintomei"] = {
            "name": "新東名高速道路",
            "name_en": "Shin-Tomei Expressway",
            "ref": "E1A",
            "coordinates": [[c[1], c[0]] for c in shintomei],  # [lat, lon] for Leaflet
        }
    return routes


@app.get("/dashboard", tags=["Visualization"])
async def get_dashboard():
    """Serve live dashboard."""
    dashboard_path = Path(__file__).parent.parent.parent / "output" / "dashboard.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path, media_type='text/html')
    raise HTTPException(status_code=404, detail="Dashboard not found")


@app.get("/api/v1/map", tags=["Visualization"])
async def get_map_html():
    """
    Generate and return interactive map HTML.

    Note: For production, generate maps asynchronously.
    """
    try:
        from src.utils.visualization import create_risk_map

        segments = [
            {
                'segment_id': s.segment_id,
                'lat': s.lat,
                'lon': s.lon,
                'score': s.score,
                'level': s.level,
                'message': s.message
            }
            for s in SAMPLE_SEGMENTS
        ]

        output_path = 'output/api_map.html'
        create_risk_map(segments, output_path=output_path)

        return FileResponse(output_path, media_type='text/html')

    except ImportError:
        return JSONResponse(
            status_code=501,
            content={"error": "Visualization dependencies not installed"}
        )
    except Exception as e:
        logger.error(f"Error generating map: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Authentication Routes
# ============================================================================

class LoginRequest(BaseModel):
    password: str


@app.get("/login", tags=["Auth"])
async def login_page():
    """Serve login page."""
    login_path = Path(__file__).parent.parent.parent / "output" / "login.html"
    if login_path.exists():
        return FileResponse(login_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="Login page not found")


@app.post("/api/v1/auth/login", tags=["Auth"])
async def login(body: LoginRequest):
    """Authenticate with shared password."""
    if verify_password(body.password):
        response = JSONResponse(content={"status": "ok"})
        return set_session_cookie(response)
    return JSONResponse(status_code=401, content={"detail": "Invalid password"})


@app.get("/api/v1/auth/logout", tags=["Auth"])
async def logout():
    """Clear session and redirect to login."""
    response = RedirectResponse(url="/login", status_code=302)
    return clear_session_cookie(response)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
