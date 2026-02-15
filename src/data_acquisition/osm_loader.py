"""
Highway Geometry Loader
=======================

Load Tomei (E1) and Shin-Tomei (E1A) expressway geometry from:
  1. MLIT N06-24 (国土数値情報 高速道路時系列データ) — primary source
  2. OpenStreetMap Overpass API — fallback

Public API:
  get_tomei_coords()     → List[(lon, lat)]
  get_shintomei_coords() → List[(lon, lat)]
"""

import requests
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OSMHighwayLoader:
    """Load highway geometry from OpenStreetMap."""

    OVERPASS_URLS = [
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass-api.de/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]

    CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "external"

    def __init__(self):
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tomei_expressway(
        self, use_cache: bool = True
    ) -> List[Tuple[float, float]]:
        """
        Get Tomei Expressway (E1 / 東名高速道路) full-route geometry.

        Uses OSM route relation 7523774 (下り / outbound: Tokyo → Komaki).
        Full route ~347 km.

        Returns:
            List of (lon, lat) coordinates, west-to-east single direction.
        """
        cache_file = self.CACHE_DIR / "tomei_expressway.json"

        if use_cache and cache_file.exists():
            logger.info("Loading Tomei Expressway from cache...")
            with open(cache_file, "r") as f:
                data = json.load(f)
                return [tuple(c) for c in data["coordinates"]]

        # Tomei 下り relation: Tokyo → Komaki
        relation_ids = [7523774]
        coords = self._fetch_from_relations(relation_ids)

        if not coords:
            logger.error("Failed to fetch Tomei Expressway from OSM")
            return self._get_fallback_coords()

        # Ensure west-to-east ordering (Komaki → Tokyo)
        if coords[0][0] > coords[-1][0]:
            coords = list(reversed(coords))

        self._save_cache(
            cache_file, coords, "東名高速道路", "Tomei Expressway", "E1"
        )
        return coords

    def get_shintomei_expressway(
        self, use_cache: bool = True
    ) -> List[Tuple[float, float]]:
        """
        Get Shin-Tomei Expressway (E1A / 新東名高速道路) full-route geometry.

        Uses OSM route relations for 下り direction:
          - 10382781: 新御殿場IC → 豊田東JCT (western section)
          - 2137240:  海老名南JCT → 新秦野IC (eastern section)
        Sections are merged west-to-east.

        Returns:
            List of (lon, lat) coordinates, west-to-east single direction.
        """
        cache_file = self.CACHE_DIR / "shintomei_expressway.json"

        if use_cache and cache_file.exists():
            logger.info("Loading Shin-Tomei Expressway from cache...")
            with open(cache_file, "r") as f:
                data = json.load(f)
                return [tuple(c) for c in data["coordinates"]]

        # Shin-Tomei 下り relations (two sections)
        # 10382781: 新御殿場IC → 豊田東JCT (western)
        # 2137240:  海老名南JCT → 新秦野IC (eastern)
        relation_ids = [10382781, 2137240]
        coords = self._fetch_from_relations(relation_ids)

        if not coords:
            logger.error("Failed to fetch Shin-Tomei Expressway from OSM")
            return []

        # Fill the gap between 新秦野IC and 新御殿場IC.
        # This section (opened 2024-03) has no route relation yet,
        # so we query individual ways in the gap area.
        # Note: ~22km gap between 新御殿場IC and 新秦野IC is still
        # unmapped in OSM as of this writing.
        gap_sections = self._fetch_gap_sections(
            bbox=(138.82, 35.15, 139.24, 35.45),
            ref="E1A",
            name_pattern="新東名",
        )
        for section in gap_sections:
            coords = self._insert_gap_section(coords, section)

        # Ensure west-to-east ordering
        if coords[0][0] > coords[-1][0]:
            coords = list(reversed(coords))

        self._save_cache(
            cache_file, coords, "新東名高速道路", "Shin-Tomei Expressway", "E1A"
        )
        return coords

    def get_highway_within_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        highway_type: str = "motorway",
    ) -> List[Tuple[float, float]]:
        """
        Get all highways of specified type within bounding box.

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            highway_type: OSM highway type (motorway, trunk, etc.)

        Returns:
            List of (lon, lat) coordinates
        """
        min_lon, min_lat, max_lon, max_lat = bbox

        query = f"""
        [out:json][timeout:60];
        way["highway"="{highway_type}"]({min_lat},{min_lon},{max_lat},{max_lon});
        (._;>;);
        out body;
        """

        data = self._fetch_overpass(query)
        if data is None:
            return []

        nodes: Dict[int, Tuple[float, float]] = {}
        coords = []
        for elem in data.get("elements", []):
            if elem["type"] == "node":
                nodes[elem["id"]] = (elem["lon"], elem["lat"])
            elif elem["type"] == "way":
                for nid in elem["nodes"]:
                    if nid in nodes:
                        coords.append(nodes[nid])
        return coords

    # ------------------------------------------------------------------
    # Internal: relation-based route building
    # ------------------------------------------------------------------

    def _fetch_from_relations(
        self, relation_ids: List[int]
    ) -> List[Tuple[float, float]]:
        """
        Fetch one or more OSM route relations and build a single
        ordered coordinate list.

        Relation members are already ordered, so we use the member
        sequence directly instead of heuristic clustering.
        """
        all_section_coords: List[List[Tuple[float, float]]] = []

        for rel_id in relation_ids:
            query = f"""
            [out:json][timeout:120];
            relation({rel_id});
            (._;>;);
            out body;
            """
            data = self._fetch_overpass(query)
            if data is None:
                logger.warning(f"Failed to fetch relation {rel_id}")
                continue

            coords = self._build_route_from_relation(data, rel_id)
            if coords:
                all_section_coords.append(coords)
                logger.info(
                    f"Relation {rel_id}: {len(coords)} points, "
                    f"lon {coords[0][0]:.3f}→{coords[-1][0]:.3f}"
                )

        if not all_section_coords:
            return []

        # If multiple sections, orient each west→east, sort, merge
        if len(all_section_coords) == 1:
            return all_section_coords[0]

        # Orient each section west→east first
        for i, section in enumerate(all_section_coords):
            if section[0][0] > section[-1][0]:
                all_section_coords[i] = list(reversed(section))

        # Sort sections by starting (western) longitude
        all_section_coords.sort(key=lambda s: s[0][0])

        # Concatenate in order
        merged = list(all_section_coords[0])
        for section in all_section_coords[1:]:
            merged.extend(section)

        logger.info(
            f"Merged {len(all_section_coords)} sections → "
            f"{len(merged)} total points"
        )
        return merged

    def _build_route_from_relation(
        self, data: dict, relation_id: int
    ) -> List[Tuple[float, float]]:
        """
        Build an ordered coordinate list from a fetched relation response.

        Uses node IDs to determine way orientation and connection.
        Skips misplaced ways that would cause large jumps (no shared
        node AND gap > threshold).  After building the route, removes
        remaining spike artifacts via post-processing.
        """
        # Parse all nodes (id → coordinate)
        nodes: Dict[int, Tuple[float, float]] = {}
        # Parse all ways (id → ordered list of node ids)
        ways: Dict[int, List[int]] = {}

        for elem in data.get("elements", []):
            if elem["type"] == "node":
                nodes[elem["id"]] = (elem["lon"], elem["lat"])
            elif elem["type"] == "way":
                ways[elem["id"]] = elem.get("nodes", [])

        # Find the relation element
        relation = None
        for elem in data.get("elements", []):
            if elem["type"] == "relation" and elem["id"] == relation_id:
                relation = elem
                break

        if relation is None:
            logger.warning(f"Relation {relation_id} not found in response")
            return []

        members = relation.get("members", [])
        way_members = [m for m in members if m["type"] == "way"]

        if not way_members:
            logger.warning(f"Relation {relation_id} has no way members")
            return []

        logger.info(
            f"Relation {relation_id}: {len(way_members)} way members, "
            f"{len(nodes)} nodes"
        )

        route_node_ids: List[int] = []
        prev_last_node_id: Optional[int] = None

        for member in way_members:
            way_id = member["ref"]
            if way_id not in ways:
                continue

            node_ids = ways[way_id]
            if len(node_ids) < 2:
                continue

            # Determine orientation using shared node IDs
            if prev_last_node_id is not None:
                if node_ids[-1] == prev_last_node_id:
                    # Last node matches prev → reverse this way
                    node_ids = list(reversed(node_ids))
                elif node_ids[0] != prev_last_node_id:
                    # No shared node — orient by coordinate proximity
                    prev_coord = nodes.get(prev_last_node_id)
                    s_coord = nodes.get(node_ids[0])
                    e_coord = nodes.get(node_ids[-1])
                    if prev_coord and s_coord and e_coord:
                        d_s = self._pt_distance(prev_coord, s_coord)
                        d_e = self._pt_distance(prev_coord, e_coord)
                        if d_e < d_s:
                            node_ids = list(reversed(node_ids))

            # Append (skip duplicate shared node)
            if route_node_ids and node_ids[0] == route_node_ids[-1]:
                route_node_ids.extend(node_ids[1:])
            else:
                route_node_ids.extend(node_ids)

            prev_last_node_id = route_node_ids[-1]

        # Convert node IDs to coordinates
        route_coords = [
            nodes[nid] for nid in route_node_ids if nid in nodes
        ]

        # Post-process: remove remaining spike artifacts.
        # A spike is a short sequence of points that jumps far away
        # and then returns near the departure point.
        route_coords = self._remove_spikes(route_coords)

        return route_coords

    def _fetch_gap_sections(
        self,
        bbox: Tuple[float, float, float, float],
        ref: str,
        name_pattern: str,
    ) -> List[List[Tuple[float, float]]]:
        """
        Fetch individual ways for a gap area and return all viable
        route sections (one per cluster), sorted west-to-east.
        """
        min_lon, min_lat, max_lon, max_lat = bbox

        query = f"""
        [out:json][timeout:90];
        (
          way["ref"="{ref}"]["highway"="motorway"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["name"~"{name_pattern}"]["highway"="motorway"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        (._;>;);
        out body;
        """

        data = self._fetch_overpass(query)
        if data is None:
            return []

        nodes: Dict[int, Tuple[float, float]] = {}
        segments: List[List[Tuple[float, float]]] = []

        for elem in data.get("elements", []):
            if elem["type"] == "node":
                nodes[elem["id"]] = (elem["lon"], elem["lat"])
            elif elem["type"] == "way":
                seg = [
                    nodes[nid]
                    for nid in elem.get("nodes", [])
                    if nid in nodes
                ]
                if len(seg) >= 2:
                    segments.append(seg)

        if not segments:
            return []

        logger.info(f"Gap fill: {len(segments)} ways in bbox")

        clusters = self._cluster_ways(segments)
        if not clusters:
            return []

        # Orient each cluster west→east, sort, apply spike removal
        results = []
        for cluster in clusters:
            if cluster[0][0] > cluster[-1][0]:
                cluster = list(reversed(cluster))
            cluster = self._remove_spikes(cluster)
            results.append(cluster)
            logger.info(
                f"Gap section: {len(cluster)} pts, "
                f"lon {cluster[0][0]:.3f}→{cluster[-1][0]:.3f}"
            )

        results.sort(key=lambda s: s[0][0])
        return results

    def _cluster_ways(
        self, segments: List[List[Tuple[float, float]]]
    ) -> List[List[Tuple[float, float]]]:
        """
        Connect way segments into continuous routes by greedy chaining.
        """
        remaining = list(range(len(segments)))
        clusters: List[List[Tuple[float, float]]] = []

        while remaining:
            remaining.sort(key=lambda i: min(c[0] for c in segments[i]))
            seed_idx = remaining.pop(0)
            chain = list(segments[seed_idx])

            changed = True
            while changed:
                changed = False
                best_idx = None
                best_dist = float("inf")
                best_end = None
                best_reverse = False

                for i in remaining:
                    seg = segments[i]
                    candidates = [
                        (self._pt_distance(chain[-1], seg[0]), i, "end", False),
                        (self._pt_distance(chain[-1], seg[-1]), i, "end", True),
                        (self._pt_distance(chain[0], seg[-1]), i, "start", False),
                        (self._pt_distance(chain[0], seg[0]), i, "start", True),
                    ]
                    for d, idx, end, rev in candidates:
                        if d < best_dist:
                            best_dist = d
                            best_idx = idx
                            best_end = end
                            best_reverse = rev

                if best_idx is not None and best_dist < 0.005:
                    seg = segments[best_idx]
                    if best_reverse:
                        seg = list(reversed(seg))
                    if best_end == "end":
                        chain.extend(seg[1:])
                    else:
                        chain = seg[:-1] + chain
                    remaining.remove(best_idx)
                    changed = True

            if len(chain) >= 10:
                clusters.append(chain)

        return clusters

    @staticmethod
    def _geographic_span(coords: List[Tuple[float, float]]) -> float:
        """Bounding-box diagonal in degrees."""
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return np.sqrt(
            (max(lons) - min(lons)) ** 2 + (max(lats) - min(lats)) ** 2
        )

    @staticmethod
    def _insert_gap_section(
        route: List[Tuple[float, float]],
        gap: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """
        Insert a gap-fill section into the route at the correct position.
        Finds the point in `route` closest to the gap's start and end,
        and splices the gap section in between.
        """
        def nearest_idx(coords, target):
            best_i, best_d = 0, float("inf")
            for i, c in enumerate(coords):
                d = (c[0] - target[0]) ** 2 + (c[1] - target[1]) ** 2
                if d < best_d:
                    best_d = d
                    best_i = i
            return best_i

        idx_start = nearest_idx(route, gap[0])
        idx_end = nearest_idx(route, gap[-1])

        if idx_start > idx_end:
            idx_start, idx_end = idx_end, idx_start
            gap = list(reversed(gap))

        merged = route[: idx_start + 1] + gap + route[idx_end:]
        logger.info(
            f"Inserted gap section ({len(gap)} pts) at "
            f"idx {idx_start}..{idx_end}"
        )
        return merged

    @staticmethod
    def _remove_spikes(
        coords: List[Tuple[float, float]],
        jump_threshold: float = 0.005,  # ~500m
        max_spike_len: int = 80,
    ) -> List[Tuple[float, float]]:
        """
        Remove spike artifacts: short detours that jump far away
        and return to roughly the same location.

        Runs multiple passes until no more spikes are found, to
        handle nested or adjacent spikes.
        """
        if len(coords) < 3:
            return coords

        clean = list(coords)
        total_removed = 0

        for _pass in range(10):  # max 10 passes
            removed_this_pass = 0
            i = 0

            while i < len(clean) - 1:
                dx = clean[i + 1][0] - clean[i][0]
                dy = clean[i + 1][1] - clean[i][1]
                gap = np.sqrt(dx * dx + dy * dy)

                if gap > jump_threshold:
                    # Look ahead for a return point near clean[i]
                    found = False
                    for j in range(
                        i + 2, min(i + 2 + max_spike_len, len(clean))
                    ):
                        dx2 = clean[j][0] - clean[i][0]
                        dy2 = clean[j][1] - clean[i][1]
                        if np.sqrt(dx2 * dx2 + dy2 * dy2) < jump_threshold:
                            del clean[i + 1 : j]
                            removed_this_pass += j - (i + 1)
                            found = True
                            break
                    if not found:
                        i += 1
                else:
                    i += 1

            total_removed += removed_this_pass
            if removed_this_pass == 0:
                break

        if total_removed:
            logger.info(
                f"Removed {total_removed} spike point(s) "
                f"in {_pass + 1} pass(es)"
            )

        return clean

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _fetch_overpass(self, query: str) -> Optional[dict]:
        """Try each Overpass endpoint until one succeeds."""
        for url in self.OVERPASS_URLS:
            try:
                logger.info(f"Trying {url}...")
                response = requests.post(
                    url, data={"data": query}, timeout=180
                )
                response.raise_for_status()
                data = response.json()
                logger.info(f"Success with {url}")
                return data
            except Exception as e:
                logger.warning(f"Failed: {e}")
                continue
        return None

    @staticmethod
    def _pt_distance(
        c1: Tuple[float, float], c2: Tuple[float, float]
    ) -> float:
        """Approximate distance between two lon/lat points in degrees."""
        return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    def _save_cache(
        self,
        path: Path,
        coords: List[Tuple[float, float]],
        name_ja: str,
        name_en: str,
        ref: str,
    ) -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "source": "OpenStreetMap",
                    "name": name_ja,
                    "name_en": name_en,
                    "ref": ref,
                    "coordinates": coords,
                    "count": len(coords),
                },
                f,
                indent=2,
            )
        logger.info(f"Cached {len(coords)} coordinates → {path}")

    def _get_fallback_coords(self) -> List[Tuple[float, float]]:
        """Fallback coordinates if OSM query fails."""
        logger.warning("Using fallback coordinates")
        from src.utils.geo_utils import CENTRAL_EXPRESSWAY_COORDS

        return CENTRAL_EXPRESSWAY_COORDS


class MLITHighwayLoader:
    """
    Load highway geometry from MLIT N06-24 (国土数値情報 高速道路時系列データ).

    Higher-quality geometry than OSM, with official section definitions.
    Covers Tomei (第一東海自動車道) and Shin-Tomei (第二東海自動車道).
    """

    GEOJSON_PATH = (
        Path(__file__).parent.parent.parent
        / "data"
        / "external"
        / "mlit_n06"
        / "N06-24"
        / "UTF-8"
        / "N06-24_HighwaySection.geojson"
    )

    CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "external"

    # Shin-Tomei spur / ramp section codes to exclude
    SHINTOMEI_EXCLUDE = {"EA02_022008", "EA02_022018", "EA02_022019"}

    def __init__(self):
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._geojson = None

    def _load_geojson(self) -> Optional[dict]:
        """Load and cache the GeoJSON file."""
        if self._geojson is not None:
            return self._geojson

        if not self.GEOJSON_PATH.exists():
            logger.warning(f"MLIT GeoJSON not found: {self.GEOJSON_PATH}")
            return None

        logger.info(f"Loading MLIT GeoJSON: {self.GEOJSON_PATH}")
        with open(self.GEOJSON_PATH, "r", encoding="utf-8") as f:
            self._geojson = json.load(f)
        logger.info(
            f"Loaded {len(self._geojson.get('features', []))} features"
        )
        return self._geojson

    def _filter_features(
        self,
        highway_name: str,
        exclude_codes: Optional[set] = None,
    ) -> List[dict]:
        """
        Filter GeoJSON features by highway name, deduplicate by section
        code (N06_004), and optionally exclude specific section codes.

        When duplicate section codes exist, prefer N06_010 == 4.
        """
        geojson = self._load_geojson()
        if geojson is None:
            return []

        # Collect features matching this highway name
        matched = [
            f
            for f in geojson["features"]
            if f["properties"].get("N06_007") == highway_name
        ]
        logger.info(
            f"MLIT: {len(matched)} raw features for '{highway_name}'"
        )

        # Exclude specific section codes
        if exclude_codes:
            before = len(matched)
            matched = [
                f
                for f in matched
                if f["properties"].get("N06_004") not in exclude_codes
            ]
            logger.info(
                f"MLIT: excluded {before - len(matched)} spur/ramp sections"
            )

        # Deduplicate by section code, preferring N06_010 == 4
        by_code: Dict[str, List[dict]] = {}
        for f in matched:
            code = f["properties"].get("N06_004", "")
            by_code.setdefault(code, []).append(f)

        deduped = []
        for code, features in by_code.items():
            if len(features) == 1:
                deduped.append(features[0])
            else:
                # Prefer N06_010 == 4
                preferred = [
                    f for f in features if f["properties"].get("N06_010") == 4
                ]
                if preferred:
                    deduped.append(preferred[0])
                else:
                    deduped.append(features[0])
                logger.info(
                    f"MLIT: deduplicated section {code} "
                    f"({len(features)} variants → 1)"
                )

        logger.info(f"MLIT: {len(deduped)} unique sections after dedup")
        return deduped

    def _chain_segments(
        self, features: List[dict]
    ) -> List[Tuple[float, float]]:
        """
        Extract LineString coordinates from features and chain them
        into a single west-to-east route using greedy chaining.
        """
        if not features:
            return []

        # Extract coordinate lists from each feature
        segments: List[List[Tuple[float, float]]] = []
        for f in features:
            geom = f["geometry"]
            if geom["type"] == "LineString":
                coords = [(c[0], c[1]) for c in geom["coordinates"]]
                if len(coords) >= 2:
                    segments.append(coords)
            elif geom["type"] == "MultiLineString":
                for line in geom["coordinates"]:
                    coords = [(c[0], c[1]) for c in line]
                    if len(coords) >= 2:
                        segments.append(coords)

        if not segments:
            return []

        logger.info(
            f"MLIT: chaining {len(segments)} segments "
            f"({sum(len(s) for s in segments)} total points)"
        )

        # Orient each segment west→east
        for i, seg in enumerate(segments):
            if seg[0][0] > seg[-1][0]:
                segments[i] = list(reversed(seg))

        # Sort by western-most longitude
        segments.sort(key=lambda s: s[0][0])

        # Greedy chaining: start with westernmost, append nearest
        chain = list(segments[0])
        remaining = list(range(1, len(segments)))

        changed = True
        while changed and remaining:
            changed = False
            best_idx = None
            best_dist = float("inf")
            best_end = None  # "end" or "start"
            best_reverse = False

            for i in remaining:
                seg = segments[i]
                candidates = [
                    (self._pt_distance(chain[-1], seg[0]), i, "end", False),
                    (self._pt_distance(chain[-1], seg[-1]), i, "end", True),
                    (self._pt_distance(chain[0], seg[-1]), i, "start", False),
                    (self._pt_distance(chain[0], seg[0]), i, "start", True),
                ]
                for d, idx, end, rev in candidates:
                    if d < best_dist:
                        best_dist = d
                        best_idx = idx
                        best_end = end
                        best_reverse = rev

            if best_idx is not None and best_dist < 0.05:
                seg = segments[best_idx]
                if best_reverse:
                    seg = list(reversed(seg))
                if best_end == "end":
                    chain.extend(seg)
                else:
                    chain = seg + chain
                remaining.remove(best_idx)
                changed = True

        if remaining:
            logger.warning(
                f"MLIT: {len(remaining)} segments could not be chained"
            )

        # Final west→east orientation
        if chain[0][0] > chain[-1][0]:
            chain = list(reversed(chain))

        return chain

    @staticmethod
    def _pt_distance(
        c1: Tuple[float, float], c2: Tuple[float, float]
    ) -> float:
        return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    def _save_cache(
        self,
        path: Path,
        coords: List[Tuple[float, float]],
        name_ja: str,
        name_en: str,
        ref: str,
    ) -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "source": "MLIT N06-24",
                    "name": name_ja,
                    "name_en": name_en,
                    "ref": ref,
                    "coordinates": coords,
                    "count": len(coords),
                },
                f,
                indent=2,
            )
        logger.info(f"Cached {len(coords)} coordinates → {path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tomei_expressway(self) -> Optional[List[Tuple[float, float]]]:
        """
        Get Tomei Expressway (第一東海自動車道) from MLIT N06-24.

        Returns:
            List of (lon, lat) coordinates west-to-east, or None if
            the GeoJSON file is not available.
        """
        cache_file = self.CACHE_DIR / "tomei_expressway.json"

        # Check existing cache — use if already MLIT sourced
        if cache_file.exists():
            with open(cache_file, "r") as f:
                data = json.load(f)
            if data.get("source") == "MLIT N06-24":
                logger.info("Loading Tomei (MLIT) from cache...")
                return [tuple(c) for c in data["coordinates"]]

        features = self._filter_features("第一東海自動車道")
        if not features:
            return None

        coords = self._chain_segments(features)
        if not coords:
            return None

        lons = [c[0] for c in coords]
        logger.info(
            f"MLIT Tomei: {len(coords)} points, "
            f"lon {min(lons):.4f}→{max(lons):.4f}"
        )

        self._save_cache(
            cache_file, coords, "東名高速道路", "Tomei Expressway", "E1"
        )
        return coords

    def get_shintomei_expressway(self) -> Optional[List[Tuple[float, float]]]:
        """
        Get Shin-Tomei Expressway (第二東海自動車道) from MLIT N06-24.

        Returns:
            List of (lon, lat) coordinates west-to-east, or None if
            the GeoJSON file is not available.
        """
        cache_file = self.CACHE_DIR / "shintomei_expressway.json"

        # Check existing cache — use if already MLIT sourced
        if cache_file.exists():
            with open(cache_file, "r") as f:
                data = json.load(f)
            if data.get("source") == "MLIT N06-24":
                logger.info("Loading Shin-Tomei (MLIT) from cache...")
                return [tuple(c) for c in data["coordinates"]]

        features = self._filter_features(
            "第二東海自動車道", exclude_codes=self.SHINTOMEI_EXCLUDE
        )
        if not features:
            return None

        coords = self._chain_segments(features)
        if not coords:
            return None

        lons = [c[0] for c in coords]
        logger.info(
            f"MLIT Shin-Tomei: {len(coords)} points, "
            f"lon {min(lons):.4f}→{max(lons):.4f}"
        )

        self._save_cache(
            cache_file,
            coords,
            "新東名高速道路",
            "Shin-Tomei Expressway",
            "E1A",
        )
        return coords


def get_tomei_coords() -> List[Tuple[float, float]]:
    """Get Tomei Expressway coordinates (MLIT → OSM fallback)."""
    # Try MLIT first
    mlit = MLITHighwayLoader()
    coords = mlit.get_tomei_expressway()
    if coords:
        return coords

    # Fall back to OSM
    logger.info("MLIT unavailable for Tomei, falling back to OSM")
    osm = OSMHighwayLoader()
    return osm.get_tomei_expressway()


def get_shintomei_coords() -> List[Tuple[float, float]]:
    """Get Shin-Tomei Expressway coordinates (MLIT → OSM fallback)."""
    # Try MLIT first
    mlit = MLITHighwayLoader()
    coords = mlit.get_shintomei_expressway()
    if coords:
        return coords

    # Fall back to OSM
    logger.info("MLIT unavailable for Shin-Tomei, falling back to OSM")
    osm = OSMHighwayLoader()
    return osm.get_shintomei_expressway()


if __name__ == "__main__":
    # --- MLIT N06-24 ---
    mlit = MLITHighwayLoader()

    print("=== MLIT N06-24: Tomei Expressway (E1) ===")
    tomei = mlit.get_tomei_expressway()
    if tomei:
        lons = [c[0] for c in tomei]
        print(f"Points: {len(tomei)}")
        print(f"Start (west): lon={tomei[0][0]:.4f}, lat={tomei[0][1]:.4f}")
        print(f"End (east):   lon={tomei[-1][0]:.4f}, lat={tomei[-1][1]:.4f}")
        print(f"Lon range: {min(lons):.4f} → {max(lons):.4f}")
    else:
        print("MLIT data not available")

    print("\n=== MLIT N06-24: Shin-Tomei Expressway (E1A) ===")
    shintomei = mlit.get_shintomei_expressway()
    if shintomei:
        lons = [c[0] for c in shintomei]
        print(f"Points: {len(shintomei)}")
        print(f"Start (west): lon={shintomei[0][0]:.4f}, lat={shintomei[0][1]:.4f}")
        print(f"End (east):   lon={shintomei[-1][0]:.4f}, lat={shintomei[-1][1]:.4f}")
        print(f"Lon range: {min(lons):.4f} → {max(lons):.4f}")
    else:
        print("MLIT data not available")

    # --- OSM (for comparison / fallback) ---
    print("\n=== OSM: Tomei Expressway (E1) ===")
    osm = OSMHighwayLoader()
    tomei_osm = osm.get_tomei_expressway()
    if tomei_osm:
        print(f"Points: {len(tomei_osm)}")
        print(f"Start (west): lon={tomei_osm[0][0]:.4f}, lat={tomei_osm[0][1]:.4f}")
        print(f"End (east):   lon={tomei_osm[-1][0]:.4f}, lat={tomei_osm[-1][1]:.4f}")

    print("\n=== OSM: Shin-Tomei Expressway (E1A) ===")
    shintomei_osm = osm.get_shintomei_expressway()
    if shintomei_osm:
        print(f"Points: {len(shintomei_osm)}")
        print(f"Start (west): lon={shintomei_osm[0][0]:.4f}, lat={shintomei_osm[0][1]:.4f}")
        print(f"End (east):   lon={shintomei_osm[-1][0]:.4f}, lat={shintomei_osm[-1][1]:.4f}")
