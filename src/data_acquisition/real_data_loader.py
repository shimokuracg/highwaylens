"""
Real Data Loader
================

Load real data from various sources:
- DEM/Slope: SRTM 1-arcsec (30m) from SNAP cache
- Weather: JMA AMeDAS real-time data via bosai JSON API
- Geology: GSI geological classification
"""

import json as _json
import numpy as np
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
import zipfile
import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealDEMLoader:
    """Load real SRTM DEM data and calculate slopes."""

    SNAP_DEM_DIR = Path.home() / ".snap/auxdata/dem/SRTM 1Sec HGT"

    def __init__(self):
        self.dem_cache = {}  # Cache loaded DEM tiles
        self.slope_cache = {}  # Cache calculated slopes

    def _get_tile_name(self, lat: float, lon: float) -> str:
        """Get SRTM tile name for a coordinate."""
        lat_prefix = 'N' if lat >= 0 else 'S'
        lon_prefix = 'E' if lon >= 0 else 'W'
        lat_int = int(abs(lat))
        lon_int = int(abs(lon))
        return f"{lat_prefix}{lat_int:02d}{lon_prefix}{lon_int:03d}"

    def _load_hgt_tile(self, tile_name: str) -> Optional[np.ndarray]:
        """Load SRTM HGT tile from SNAP cache."""
        if tile_name in self.dem_cache:
            return self.dem_cache[tile_name]

        # Try to find the tile
        zip_path = self.SNAP_DEM_DIR / f"{tile_name}.SRTMGL1.hgt.zip"

        if not zip_path.exists():
            logger.warning(f"DEM tile not found: {zip_path}")
            return None

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                hgt_name = f"{tile_name}.hgt"
                with zf.open(hgt_name) as hgt_file:
                    # SRTM 1-arcsec is 3601x3601 pixels, 2 bytes per pixel, big-endian
                    data = hgt_file.read()
                    dem = np.frombuffer(data, dtype='>i2').reshape(3601, 3601)
                    # Replace void values (-32768) with NaN
                    dem = dem.astype(np.float32)
                    dem[dem == -32768] = np.nan
                    self.dem_cache[tile_name] = dem
                    logger.info(f"Loaded DEM tile: {tile_name}")
                    return dem
        except Exception as e:
            logger.error(f"Error loading DEM tile {tile_name}: {e}")
            return None

    def get_elevation(self, lat: float, lon: float) -> float:
        """Get elevation at a point (meters)."""
        tile_name = self._get_tile_name(lat, lon)
        dem = self._load_hgt_tile(tile_name)

        if dem is None:
            return np.nan

        # Calculate pixel position within tile
        # SRTM tiles start at lower-left corner
        lat_frac = lat - int(lat) if lat >= 0 else 1 + (lat - int(lat))
        lon_frac = lon - int(lon) if lon >= 0 else 1 + (lon - int(lon))

        row = int((1 - lat_frac) * 3600)  # Top to bottom
        col = int(lon_frac * 3600)

        row = max(0, min(3600, row))
        col = max(0, min(3600, col))

        return float(dem[row, col])

    def get_slope(self, lat: float, lon: float, window_size: int = 3) -> float:
        """
        Calculate slope at a point using surrounding DEM values.

        Uses Horn's method for slope calculation.
        Returns slope in degrees.
        """
        tile_name = self._get_tile_name(lat, lon)
        dem = self._load_hgt_tile(tile_name)

        if dem is None:
            return np.nan

        # Calculate pixel position
        lat_frac = lat - int(lat) if lat >= 0 else 1 + (lat - int(lat))
        lon_frac = lon - int(lon) if lon >= 0 else 1 + (lon - int(lon))

        row = int((1 - lat_frac) * 3600)
        col = int(lon_frac * 3600)

        # Ensure we have enough margin for window
        half = window_size // 2
        if row < half or row >= 3601 - half or col < half or col >= 3601 - half:
            return np.nan

        # Extract window
        window = dem[row-half:row+half+1, col-half:col+half+1]

        if np.any(np.isnan(window)):
            return np.nan

        # Cell size in meters (approximately 30m at these latitudes)
        cell_size = 30.0

        # Horn's method for slope
        # dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 * cell_size)
        # dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 * cell_size)
        # where window is:
        # a b c
        # d e f
        # g h i

        a, b, c = window[0, 0], window[0, 1], window[0, 2]
        d, e, f = window[1, 0], window[1, 1], window[1, 2]
        g, h, i = window[2, 0], window[2, 1], window[2, 2]

        dzdx = ((c + 2*f + i) - (a + 2*d + g)) / (8 * cell_size)
        dzdy = ((g + 2*h + i) - (a + 2*b + c)) / (8 * cell_size)

        slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
        slope_deg = np.degrees(slope_rad)

        return float(slope_deg)

    def get_terrain_data(self, lat: float, lon: float, fallback_slope: float = None) -> Dict:
        """Get complete terrain data for a location.

        Args:
            lat: Latitude
            lon: Longitude
            fallback_slope: Optional slope value to use if DEM is unavailable
                           (e.g., estimated from norimen tier count)
        """
        elevation = self.get_elevation(lat, lon)
        slope = self.get_slope(lat, lon)

        # Use fallback slope if DEM data unavailable
        if np.isnan(slope):
            if fallback_slope is not None:
                slope = fallback_slope
                source = "推定値 (法面データより)"
            else:
                # Default estimate for highway slope areas
                slope = 20.0
                source = "推定値 (デフォルト)"
        else:
            source = "SRTM 1-arcsec (30m)"

        return {
            "elevation_m": elevation if not np.isnan(elevation) else 100,
            "slope_deg": slope,
            "source": source
        }


class RealWeatherLoader:
    """Fetch real-time weather data from JMA AMeDAS bosai JSON API.

    Uses the publicly available JMA endpoints (no authentication required):
    - latest_time.txt → current observation timestamp
    - data/map/{ts}00.json → all-station observation data

    Data is cached and refreshed every 10 minutes (AMeDAS update interval).
    """

    LATEST_TIME_URL = "https://www.jma.go.jp/bosai/amedas/data/latest_time.txt"
    MAP_DATA_URL = "https://www.jma.go.jp/bosai/amedas/data/map/{ts}00.json"
    CACHE_TTL_SECONDS = 600  # 10 minutes

    # Corrected AMeDAS station IDs for Tomei/Shin-Tomei corridor
    # Verified against amedastable.json
    STATIONS = {
        # Tomei Expressway area (East → West)
        "ebina":     {"id": "46091", "name": "海老名", "lat": 35.4333, "lon": 139.3867},
        "odawara":   {"id": "46166", "name": "小田原", "lat": 35.2767, "lon": 139.1550},
        "gotemba":   {"id": "50136", "name": "御殿場", "lat": 35.3050, "lon": 138.9267},
        "mishima":   {"id": "50206", "name": "三島",   "lat": 35.1133, "lon": 138.9250},
        "fuji":      {"id": "50196", "name": "富士",   "lat": 35.2217, "lon": 138.6717},
        # Shin-Tomei Expressway area (East → West)
        "shizuoka":  {"id": "50331", "name": "静岡",   "lat": 34.9750, "lon": 138.4033},
        "kakegawa":  {"id": "50466", "name": "掛川",   "lat": 34.7683, "lon": 137.9950},
        "iwata":     {"id": "50536", "name": "磐田",   "lat": 34.6917, "lon": 137.8800},
        "hamamatsu": {"id": "50456", "name": "浜松",   "lat": 34.7533, "lon": 137.7117},
        "tenryu":    {"id": "50386", "name": "天竜",   "lat": 34.8900, "lon": 137.8133},
    }

    def __init__(self):
        self._cache: Dict[str, Dict] = {}  # station_id -> precip data
        self._cache_time: Optional[datetime] = None
        self._observed_at: str = ""
        self._fetch_latest()

    def _fetch_latest(self):
        """Fetch latest AMeDAS observation data for all stations."""
        try:
            # 1. Get latest observation timestamp
            req = urllib.request.Request(
                self.LATEST_TIME_URL,
                headers={"User-Agent": "HighwayLens/2.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                latest_str = resp.read().decode().strip()

            latest_dt = datetime.fromisoformat(latest_str)
            ts = latest_dt.strftime("%Y%m%d%H%M")

            # 2. Fetch all-station map data
            url = self.MAP_DATA_URL.format(ts=ts)
            req = urllib.request.Request(url, headers={"User-Agent": "HighwayLens/2.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = _json.loads(resp.read().decode())

            # 3. Extract precipitation data for our stations
            self._cache = {}
            for station_info in self.STATIONS.values():
                sid = station_info["id"]
                if sid in data:
                    obs = data[sid]
                    self._cache[sid] = {
                        "precipitation1h": self._extract(obs, "precipitation1h"),
                        "precipitation3h": self._extract(obs, "precipitation3h"),
                        "precipitation24h": self._extract(obs, "precipitation24h"),
                        "temp": self._extract(obs, "temp"),
                        "humidity": self._extract(obs, "humidity"),
                    }

            self._cache_time = datetime.now(timezone.utc)
            self._observed_at = latest_dt.isoformat()
            logger.info(
                f"AMeDAS data fetched: {ts} ({len(self._cache)}/{len(self.STATIONS)} stations)"
            )

        except Exception as e:
            logger.warning(f"AMeDAS fetch failed: {e}")
            # Keep any previous cache intact; first-run will have empty cache

    @staticmethod
    def _extract(obs: dict, key: str) -> float:
        """Extract a value from AMeDAS observation, checking quality flag.

        AMeDAS values are [value, quality_flag].  quality_flag == 0 means OK.
        """
        val = obs.get(key)
        if val is None:
            return 0.0
        if isinstance(val, list) and len(val) >= 2:
            if val[1] == 0 and val[0] is not None:
                return float(val[0])
        return 0.0

    def _maybe_refresh(self):
        """Refresh cache if stale (older than CACHE_TTL_SECONDS)."""
        if self._cache_time is None:
            self._fetch_latest()
            return
        elapsed = (datetime.now(timezone.utc) - self._cache_time).total_seconds()
        if elapsed > self.CACHE_TTL_SECONDS:
            self._fetch_latest()

    def get_nearest_station(self, lat: float, lon: float) -> str:
        """Find nearest AMeDAS station by Euclidean distance."""
        min_dist = float("inf")
        nearest = "shizuoka"

        for name, info in self.STATIONS.items():
            dist = (lat - info["lat"]) ** 2 + (lon - info["lon"]) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest = name

        return nearest

    def get_weather_data(self, lat: float, lon: float) -> Dict:
        """Get real-time weather data for a location from nearest AMeDAS station."""
        self._maybe_refresh()

        station = self.get_nearest_station(lat, lon)
        station_info = self.STATIONS[station]
        sid = station_info["id"]

        precip = self._cache.get(sid)
        if precip:
            rain_24h = precip["precipitation24h"]
            rain_1h = precip["precipitation1h"]
            return {
                "rainfall_1h_mm": rain_1h,
                "rainfall_24h_mm": rain_24h,
                "rainfall_48h_mm": rain_24h,  # 24h as proxy for scoring
                "rainfall_7d_mm": 0,
                "station_name": station_info["name"],
                "station_id": sid,
                "observed_at": self._observed_at,
                "source": f"JMA AMeDAS ({station_info['name']})",
            }

        return {
            "rainfall_1h_mm": 0,
            "rainfall_24h_mm": 0,
            "rainfall_48h_mm": 0,
            "rainfall_7d_mm": 0,
            "station_name": station_info["name"],
            "station_id": sid,
            "observed_at": "",
            "source": "AMeDAS (取得失敗)",
        }


class RealGeologyLoader:
    """Load geological data based on GSI geological maps."""

    # Geological units along Tomei and Shin-Tomei Expressways
    # Based on GSI 1:200,000 geological map
    # Simplified classification for slope stability assessment
    GEOLOGY_ZONES = [
        # Tomei Expressway area (Kanagawa - East Shizuoka)
        # (min_lon, max_lon, min_lat, max_lat, geology_type, description)
        (139.30, 139.40, 35.40, 35.50, "alluvial", "Sagami River alluvial plain"),
        (139.15, 139.30, 35.30, 35.45, "sandstone", "Tanzawa Group - sandstone/mudstone"),
        (139.00, 139.15, 35.25, 35.35, "granite", "Tanzawa plutonic rocks"),
        (138.90, 139.05, 35.20, 35.30, "volcanic_ash", "Fuji volcanic deposits"),
        (138.85, 138.95, 35.10, 35.25, "volcanic_ash", "Hakone volcanic deposits"),
        (138.80, 138.90, 35.05, 35.15, "alluvial", "Numazu coastal plain"),

        # Shin-Tomei Expressway area (Central Shizuoka - West Shizuoka)
        # Based on GSI geological map of Shizuoka Prefecture
        (138.30, 138.50, 34.90, 35.00, "sandstone", "Shimanto Belt - sandstone"),
        (138.10, 138.30, 34.80, 34.95, "mudstone", "Shimanto Belt - mudstone/shale"),
        (137.90, 138.10, 34.75, 34.90, "sandstone", "Setogawa Group - sandstone"),
        (137.70, 137.90, 34.70, 34.85, "mudstone", "Setogawa Group - mudstone"),
        (137.50, 137.70, 34.80, 34.95, "sandstone", "Ryoke metamorphic belt"),
        (137.45, 137.65, 34.85, 34.95, "granite", "Ryoke granitic rocks"),
    ]

    # Stability factors by geology type (lower = less stable)
    STABILITY_FACTORS = {
        "alluvial": 0.9,      # Generally stable, but can liquefy
        "sandstone": 0.7,     # Moderate stability
        "mudstone": 0.5,      # Less stable, prone to sliding
        "granite": 0.85,      # Stable when intact
        "volcanic_ash": 0.6,  # Unstable when wet
        "volcanic_rock": 0.8, # Generally stable
        "shale": 0.5,         # Prone to sliding when weathered
        "tuff": 0.55,         # Variable stability
    }

    def get_geology(self, lat: float, lon: float) -> str:
        """Get geological unit at a location."""
        for min_lon, max_lon, min_lat, max_lat, geology, desc in self.GEOLOGY_ZONES:
            if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
                return geology

        # Default based on general area
        if lon > 139.2:
            return "sandstone"
        elif lon > 139.0:
            return "granite"
        elif lon > 138.0:
            return "volcanic_ash"
        elif lon > 137.5:
            return "sandstone"  # Shin-Tomei area - Shimanto/Setogawa
        else:
            return "mudstone"   # West Shin-Tomei area

    def get_geology_data(self, lat: float, lon: float) -> Dict:
        """Get complete geological data for a location."""
        geology = self.get_geology(lat, lon)
        stability = self.STABILITY_FACTORS.get(geology, 0.7)

        return {
            "geology": geology,
            "stability_factor": stability,
            "source": "GSI 1:200,000 geological map"
        }


class RealDataIntegrator:
    """Integrate all real data sources."""

    def __init__(self):
        self.dem_loader = RealDEMLoader()
        self.weather_loader = RealWeatherLoader()
        self.geology_loader = RealGeologyLoader()
        logger.info("Real data integrator initialized")

    def get_all_data(self, lat: float, lon: float) -> Dict:
        """Get all real data for a location."""
        terrain = self.dem_loader.get_terrain_data(lat, lon)
        weather = self.weather_loader.get_weather_data(lat, lon)
        geology = self.geology_loader.get_geology_data(lat, lon)

        return {
            "terrain": terrain,
            "weather": weather,
            "geology": geology,
            "lat": lat,
            "lon": lon
        }


# Singleton instance
_real_data = None

def get_real_data_integrator() -> RealDataIntegrator:
    """Get singleton real data integrator."""
    global _real_data
    if _real_data is None:
        _real_data = RealDataIntegrator()
    return _real_data


def get_real_terrain(lat: float, lon: float, fallback_slope: float = None) -> Dict:
    """Get real terrain data."""
    return get_real_data_integrator().dem_loader.get_terrain_data(lat, lon, fallback_slope=fallback_slope)


def get_real_weather(lat: float, lon: float) -> Dict:
    """Get real weather data."""
    return get_real_data_integrator().weather_loader.get_weather_data(lat, lon)


def get_real_geology(lat: float, lon: float) -> Dict:
    """Get real geology data."""
    return get_real_data_integrator().geology_loader.get_geology_data(lat, lon)


if __name__ == "__main__":
    # Test the loaders
    print("Testing Real Data Loader\n")

    integrator = RealDataIntegrator()

    # Test points along Tomei Expressway
    test_points = [
        (35.42, 139.35, "Atsugi"),
        (35.35, 139.22, "Hadano"),
        (35.28, 139.05, "Mountain Pass"),
        (35.25, 138.95, "Gotemba"),
        (35.12, 138.87, "Numazu"),
    ]

    for lat, lon, name in test_points:
        print(f"\n=== {name} ({lat:.2f}, {lon:.2f}) ===")
        data = integrator.get_all_data(lat, lon)

        print(f"Elevation: {data['terrain']['elevation_m']:.0f} m")
        print(f"Slope: {data['terrain']['slope_deg']:.1f}°")
        print(f"Station: {data['weather']['station_name']} ({data['weather']['source']})")
        print(f"Geology: {data['geology']['geology']}")
