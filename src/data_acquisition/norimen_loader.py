"""
Norimen (Slope) Data Loader
===========================

Loads slope/embankment data from NEXCO contest data (7norimen.xlsx)
and converts KP (kilometer post) references to lat/lon coordinates.

The norimen data contains 394 engineered slope locations on Shin-Tomei Expressway.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import math

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NorimenSlope:
    """Represents a single norimen (slope/embankment) location."""
    slope_id: str
    road: str
    ic_from: str
    ic_to: str
    direction: str
    kp_from: float
    kp_to: float
    length_m: float
    tiers: int
    lat: float = None
    lon: float = None

    @property
    def kp_center(self) -> float:
        """Center KP of the slope."""
        return (self.kp_from + self.kp_to) / 2


def get_shintomei_coords() -> List[Tuple[float, float]]:
    """
    Get Shin-Tomei Expressway coordinates from OpenStreetMap.

    Delegates to OSMHighwayLoader.get_shintomei_expressway() which
    handles full-route coverage, proper ref/name filtering, and
    single-direction extraction.

    Returns:
        List of (lon, lat) tuples along the highway
    """
    from src.data_acquisition.osm_loader import get_shintomei_coords as _get

    return _get()


def calculate_cumulative_distance(coords: List[Tuple[float, float]]) -> List[float]:
    """
    Calculate cumulative distance along a route.

    Args:
        coords: List of (lon, lat) tuples

    Returns:
        List of cumulative distances in km from the start
    """
    distances = [0.0]

    for i in range(1, len(coords)):
        lon1, lat1 = coords[i-1]
        lon2, lat2 = coords[i]

        # Haversine formula
        R = 6371  # Earth radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        d = R * c

        distances.append(distances[-1] + d)

    return distances


def _normalize_ic_name(name: str) -> str:
    """Normalize full-width ASCII characters to half-width for IC name matching."""
    # Full-width A-Z (U+FF21..U+FF3A) → half-width A-Z
    # Full-width a-z (U+FF41..U+FF5A) → half-width a-z
    result = []
    for ch in name:
        cp = ord(ch)
        if 0xFF21 <= cp <= 0xFF3A:
            result.append(chr(cp - 0xFF21 + ord('A')))
        elif 0xFF41 <= cp <= 0xFF5A:
            result.append(chr(cp - 0xFF41 + ord('a')))
        else:
            result.append(ch)
    return ''.join(result)


def _load_ic_coords_from_geojson() -> Dict[str, Tuple[float, float]]:
    """Load Shin-Tomei IC coordinates from MLIT N06-24_Joint.geojson."""
    joint_path = (
        Path(__file__).parent.parent.parent
        / "data/external/mlit_n06/N06-24/UTF-8/N06-24_Joint.geojson"
    )

    shin_tomei_ics = {
        '新静岡', '藤枝岡部', '島田金谷', '森掛川',
        '浜松浜北', '浜松いなさJCT', '新城', '豊田東',
    }

    ic_coords = {}
    if not joint_path.exists():
        logger.warning(f"Joint.geojson not found: {joint_path}")
        return ic_coords

    with open(joint_path) as f:
        geojson = json.load(f)

    for feature in geojson['features']:
        name = feature['properties'].get('N06_018', '')
        if name in shin_tomei_ics and name not in ic_coords:
            c = feature['geometry']['coordinates']
            ic_coords[name] = (c[0], c[1])

    logger.info(f"Loaded {len(ic_coords)} IC coords from Joint.geojson")
    return ic_coords


def _derive_ic_kps(df: pd.DataFrame) -> Dict[str, float]:
    """
    Derive IC boundary KP values from xlsx section transitions.

    Each IC section (e.g. 新静岡→藤枝岡部) has a KP range. The IC boundary
    KP is the midpoint of the gap between adjacent sections.
    """
    sections = []
    for ic_from in df['ＩＣ(自)'].unique():
        subset = df[df['ＩＣ(自)'] == ic_from]
        ic_to = _normalize_ic_name(subset['ＩＣ(至)'].iloc[0])
        ic_from_norm = _normalize_ic_name(ic_from)
        kp_min = subset['管理_ＫＰ(自)'].min()
        kp_max = subset['管理_ＫＰ(至)'].max()
        sections.append((ic_from_norm, ic_to, kp_min, kp_max))

    sections.sort(key=lambda s: s[2])

    ic_kps = {}
    # First IC: min KP of first section
    ic_kps[sections[0][0]] = sections[0][2]
    # Interior ICs: midpoint of gap between adjacent sections
    for i in range(len(sections) - 1):
        ic_name = sections[i][1]
        ic_kps[ic_name] = (sections[i][3] + sections[i + 1][2]) / 2
    # Last IC: max KP of last section
    ic_kps[sections[-1][1]] = sections[-1][3]
    # 豊田東: beyond xlsx data range (estimated)
    ic_kps['豊田東'] = 252.0

    logger.info(
        f"Derived IC boundary KPs from xlsx: "
        + ", ".join(f"{k}={v:.1f}" for k, v in sorted(ic_kps.items(), key=lambda x: x[1]))
    )
    return ic_kps


def calibrate_kp_to_distance(
    coords: List[Tuple[float, float]],
    distances: List[float],
    df: pd.DataFrame = None,
) -> List[Tuple[float, float]]:
    """
    Build a piecewise-linear KP ↔ cumulative-distance mapping using
    known IC locations along the Shin-Tomei.

    Uses MLIT Joint.geojson for precise IC coordinates and derives
    IC boundary KP values from xlsx section transitions.

    Shin-Tomei KP increases east→west (Ebina KP0 → Toyota KP~254),
    but MLIT route coords run west→east (index 0 = Toyota side).

    Returns:
        Sorted list of (kp, distance_from_west) reference pairs for
        piecewise interpolation.
    """
    # Load precise IC coordinates from Joint.geojson
    ic_coords = _load_ic_coords_from_geojson()

    # Derive IC boundary KPs from xlsx section transitions
    if df is not None:
        ic_kps = _derive_ic_kps(df)
    else:
        ic_kps = {
            '新静岡': 120.8, '藤枝岡部': 138.2, '島田金谷': 153.4,
            '森掛川': 170.4, '浜松浜北': 182.2, '浜松いなさJCT': 198.6,
            '新城': 210.0, '豊田東': 252.0,
        }

    # Fallback coordinates if Joint.geojson unavailable
    fallback_coords = {
        '新静岡': (138.378, 35.042), '藤枝岡部': (138.261, 34.916),
        '島田金谷': (138.121, 34.850), '森掛川': (137.943, 34.825),
        '浜松浜北': (137.814, 34.834), '浜松いなさJCT': (137.660, 34.882),
        '新城': (137.538, 34.925), '豊田東': (137.163, 35.033),
    }

    ic_order = [
        '新静岡', '藤枝岡部', '島田金谷', '森掛川',
        '浜松浜北', '浜松いなさJCT', '新城', '豊田東',
    ]

    ref_points: List[Tuple[float, float]] = []
    for name in ic_order:
        ic_lon, ic_lat = ic_coords.get(name, fallback_coords[name])
        ic_kp = ic_kps[name]

        best_i, best_d = 0, float("inf")
        for i, (lon, lat) in enumerate(coords):
            d = (lon - ic_lon) ** 2 + (lat - ic_lat) ** 2
            if d < best_d:
                best_d = d
                best_i = i

        match_km = math.sqrt(best_d) * 111
        ref_points.append((ic_kp, distances[best_i]))
        logger.info(
            f"  IC {name}: KP {ic_kp:.1f} → route dist {distances[best_i]:.1f} km "
            f"(match {match_km:.2f} km)"
        )

    # Sort by KP ascending (east → west)
    ref_points.sort(key=lambda p: p[0])

    logger.info(
        f"KP calibration: {len(ref_points)} reference ICs, "
        f"KP {ref_points[0][0]:.0f}..{ref_points[-1][0]:.0f}"
    )
    return ref_points


def kp_to_coords(
    kp: float,
    coords: List[Tuple[float, float]],
    distances: List[float],
    ref_points: List[Tuple[float, float]],
) -> Tuple[float, float]:
    """
    Convert a KP value to lat/lon coordinates using piecewise-linear
    interpolation between calibrated IC reference points.

    Args:
        kp: Kilometer post value
        coords: Highway coordinates (west→east)
        distances: Cumulative distances along the highway
        ref_points: Sorted (kp, distance) pairs from calibrate_kp_to_distance

    Returns:
        (lat, lon) tuple
    """
    # Piecewise-linear interpolation of KP → distance
    if kp <= ref_points[0][0]:
        # Extrapolate from first two points
        kp1, d1 = ref_points[0]
        kp2, d2 = ref_points[1]
        target_dist = d1 + (kp - kp1) * (d2 - d1) / (kp2 - kp1)
    elif kp >= ref_points[-1][0]:
        # Extrapolate from last two points
        kp1, d1 = ref_points[-2]
        kp2, d2 = ref_points[-1]
        target_dist = d1 + (kp - kp1) * (d2 - d1) / (kp2 - kp1)
    else:
        # Find the bracketing segment
        for j in range(1, len(ref_points)):
            if ref_points[j][0] >= kp:
                kp1, d1 = ref_points[j - 1]
                kp2, d2 = ref_points[j]
                ratio = (kp - kp1) / (kp2 - kp1)
                target_dist = d1 + ratio * (d2 - d1)
                break
        else:
            target_dist = ref_points[-1][1]

    # Clamp to route range
    target_dist = max(0.0, min(target_dist, distances[-1]))

    # Convert distance to coordinates
    for i in range(1, len(distances)):
        if distances[i] >= target_dist:
            segment_start = distances[i-1]
            segment_end = distances[i]
            ratio = (target_dist - segment_start) / (segment_end - segment_start) if segment_end > segment_start else 0

            lon = coords[i-1][0] + ratio * (coords[i][0] - coords[i-1][0])
            lat = coords[i-1][1] + ratio * (coords[i][1] - coords[i-1][1])

            return lat, lon

    return coords[-1][1], coords[-1][0]


def load_norimen_data(
    excel_path: str = None
) -> List[NorimenSlope]:
    """
    Load norimen (slope) data from the contest Excel file.

    Args:
        excel_path: Path to 7norimen.xlsx

    Returns:
        List of NorimenSlope objects with coordinates
    """
    if excel_path is None:
        excel_path = Path(__file__).parent.parent.parent.parent / "data/other_all/7norimen.xlsx"

    excel_path = Path(excel_path)
    if not excel_path.exists():
        logger.error(f"Norimen data file not found: {excel_path}")
        return []

    # Load Excel data
    df = pd.read_excel(excel_path)
    logger.info(f"Loaded {len(df)} slope records from {excel_path}")

    # Get highway coordinates
    coords = get_shintomei_coords()
    if not coords:
        logger.error("Could not get Shin-Tomei coordinates")
        return []

    # Calculate cumulative distances
    distances = calculate_cumulative_distance(coords)
    total_dist = distances[-1]
    logger.info(f"Highway route length: {total_dist:.1f} km")

    min_kp_in_data = df['管理_ＫＰ(自)'].min()
    max_kp_in_data = df['管理_ＫＰ(至)'].max()
    logger.info(
        f"Norimen data KP range: {min_kp_in_data:.1f} - {max_kp_in_data:.1f} "
        f"({max_kp_in_data - min_kp_in_data:.1f} km)"
    )

    # Calibrate KP ↔ route distance using known IC reference points
    ref_points = calibrate_kp_to_distance(coords, distances, df)

    # Convert each slope to NorimenSlope with coordinates
    slopes = []
    for idx, row in df.iterrows():
        kp_center = (row['管理_ＫＰ(自)'] + row['管理_ＫＰ(至)']) / 2

        lat, lon = kp_to_coords(kp_center, coords, distances, ref_points)

        slope = NorimenSlope(
            slope_id=f"NORI_{idx+1:04d}",
            road=row['道路'],
            ic_from=row['ＩＣ(自)'],
            ic_to=row['ＩＣ(至)'],
            direction=row['上下線区分'],
            kp_from=row['管理_ＫＰ(自)'],
            kp_to=row['管理_ＫＰ(至)'],
            length_m=row['延長'] if pd.notna(row['延長']) else 0,
            tiers=int(row['段数']) if pd.notna(row['段数']) else 1,
            lat=lat,
            lon=lon
        )
        slopes.append(slope)

    logger.info(f"Converted {len(slopes)} slopes to coordinates")

    return slopes


def get_norimen_segments_for_monitoring(
    min_tiers: int = 3,
    min_length_m: float = 50
) -> List[Dict]:
    """
    Get norimen slopes that should be prioritized for monitoring.

    Higher tiers and longer slopes are generally higher risk.

    Args:
        min_tiers: Minimum number of tiers to include
        min_length_m: Minimum length in meters to include

    Returns:
        List of segment dictionaries for the risk calculator
    """
    slopes = load_norimen_data()

    # Filter by priority criteria
    priority_slopes = [
        s for s in slopes
        if s.tiers >= min_tiers or s.length_m >= min_length_m
    ]

    logger.info(f"Selected {len(priority_slopes)} priority slopes (tiers >= {min_tiers} or length >= {min_length_m}m)")

    # Convert to segment format
    segments = []
    for slope in priority_slopes:
        segments.append({
            'segment_id': slope.slope_id,
            'lat': slope.lat,
            'lon': slope.lon,
            'kp_from': slope.kp_from,
            'kp_to': slope.kp_to,
            'length_m': slope.length_m,
            'tiers': slope.tiers,
            'direction': slope.direction,
            'ic_section': f"{slope.ic_from} → {slope.ic_to}",
            'source': 'norimen_contest_data'
        })

    return segments


if __name__ == "__main__":
    # Test loading
    print("Loading norimen data...")
    slopes = load_norimen_data()

    print(f"\nLoaded {len(slopes)} slopes")
    print("\nFirst 5 slopes:")
    for slope in slopes[:5]:
        print(f"  {slope.slope_id}: KP {slope.kp_from:.2f}-{slope.kp_to:.2f}, "
              f"{slope.tiers} tiers, {slope.length_m:.0f}m, "
              f"coords: ({slope.lat:.4f}, {slope.lon:.4f})")

    print("\nPriority slopes for monitoring:")
    priority = get_norimen_segments_for_monitoring()
    print(f"  Total priority slopes: {len(priority)}")
