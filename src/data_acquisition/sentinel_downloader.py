"""
Sentinel-1 SAR Data Downloader
==============================

Downloads Sentinel-1 SLC data from ASF DAAC for InSAR processing.

Requirements:
    - NASA Earthdata account (free): https://urs.earthdata.nasa.gov/
    - Set environment variables:
        EARTHDATA_USER=your_username
        EARTHDATA_PASS=your_password
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

from tqdm import tqdm

# ASF Search is optional
try:
    import asf_search as asf
    HAS_ASF = True
except ImportError:
    asf = None
    HAS_ASF = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SARScene:
    """Represents a single SAR scene"""
    scene_id: str
    platform: str
    acquisition_date: datetime
    orbit_number: int
    orbit_direction: str
    path_number: int
    frame_number: int
    url: str
    file_size: int
    geometry: dict


class SentinelDownloader:
    """
    Download Sentinel-1 SLC data from Alaska Satellite Facility (ASF)

    Example:
        >>> downloader = SentinelDownloader()
        >>> scenes = downloader.search(
        ...     bbox=[138.4, 35.5, 138.9, 36.1],
        ...     start_date="2023-01-01",
        ...     end_date="2024-01-01"
        ... )
        >>> downloader.download(scenes[:5], output_dir="data/raw/sentinel1")
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Initialize with NASA Earthdata credentials.

        Args:
            username: Earthdata username (or set EARTHDATA_USER env var)
            password: Earthdata password (or set EARTHDATA_PASS env var)
        """
        if not HAS_ASF:
            logger.warning(
                "asf_search not installed. Install with: pip install asf_search"
            )
            self.session = None
            return

        self.username = username or os.getenv('EARTHDATA_USER')
        self.password = password or os.getenv('EARTHDATA_PASS')
        self.session = None

        if self.username and self.password:
            self._authenticate()
        else:
            logger.warning(
                "No credentials provided. Set EARTHDATA_USER and EARTHDATA_PASS "
                "environment variables or provide username/password."
            )

    def _authenticate(self):
        """Authenticate with ASF"""
        try:
            self.session = asf.ASFSession()
            self.session.auth_with_creds(self.username, self.password)
            logger.info("Successfully authenticated with ASF")
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise

    def search(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        platform: str = "Sentinel-1",
        beam_mode: str = "IW",
        processing_level: str = "SLC",
        polarization: str = "VV+VH",
        orbit_direction: Optional[str] = None,
        max_results: int = 100
    ) -> List[SARScene]:
        """
        Search for Sentinel-1 scenes.

        Args:
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            platform: Satellite platform
            beam_mode: Acquisition mode (IW for most of Japan)
            processing_level: Product type (SLC for InSAR)
            polarization: Polarization mode
            orbit_direction: ASCENDING or DESCENDING
            max_results: Maximum number of results

        Returns:
            List of SARScene objects
        """
        # Convert bbox to WKT
        min_lon, min_lat, max_lon, max_lat = bbox
        wkt = f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"

        logger.info(f"Searching for {platform} data...")
        logger.info(f"  AOI: {bbox}")
        logger.info(f"  Date range: {start_date} to {end_date}")

        # Build search options
        search_opts = {
            'platform': platform,
            'processingLevel': processing_level,
            'beamMode': beam_mode,
            'intersectsWith': wkt,
            'start': start_date,
            'end': end_date,
            'maxResults': max_results
        }

        if orbit_direction:
            search_opts['flightDirection'] = orbit_direction

        # Execute search
        results = asf.search(**search_opts)

        logger.info(f"Found {len(results)} scenes")

        # Convert to SARScene objects
        scenes = []
        for r in results:
            props = r.properties
            scenes.append(SARScene(
                scene_id=props.get('sceneName', ''),
                platform=props.get('platform', ''),
                acquisition_date=datetime.fromisoformat(
                    props.get('startTime', '').replace('Z', '+00:00')
                ),
                orbit_number=props.get('orbit', 0),
                orbit_direction=props.get('flightDirection', ''),
                path_number=props.get('pathNumber', 0),
                frame_number=props.get('frameNumber', 0),
                url=props.get('url', ''),
                file_size=props.get('bytes', 0),
                geometry=r.geometry
            ))

        # Sort by date
        scenes.sort(key=lambda x: x.acquisition_date)

        return scenes

    def download(
        self,
        scenes: List[SARScene],
        output_dir: str,
        parallel: int = 2
    ) -> List[Path]:
        """
        Download scenes to local directory.

        Args:
            scenes: List of SARScene objects to download
            output_dir: Output directory path
            parallel: Number of parallel downloads

        Returns:
            List of downloaded file paths
        """
        if not self.session:
            raise ValueError("Not authenticated. Provide credentials first.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {len(scenes)} scenes to {output_dir}")

        # Convert scenes back to ASF results for download
        scene_ids = [s.scene_id for s in scenes]
        results = asf.granule_search(scene_ids)

        # Download
        results.download(
            path=str(output_path),
            session=self.session,
            processes=parallel
        )

        # Return list of downloaded files
        downloaded = list(output_path.glob("*.zip"))
        logger.info(f"Downloaded {len(downloaded)} files")

        return downloaded

    def create_pairs(
        self,
        scenes: List[SARScene],
        max_temporal_baseline: int = 48,
        max_perpendicular_baseline: float = 200.0
    ) -> List[Tuple[SARScene, SARScene]]:
        """
        Create interferogram pairs from scenes.

        Args:
            scenes: List of SARScene objects (sorted by date)
            max_temporal_baseline: Maximum days between acquisitions
            max_perpendicular_baseline: Maximum perpendicular baseline in meters

        Returns:
            List of (master, slave) scene tuples
        """
        pairs = []

        for i, master in enumerate(scenes[:-1]):
            for slave in scenes[i+1:]:
                # Check temporal baseline
                temporal_baseline = (slave.acquisition_date - master.acquisition_date).days

                if temporal_baseline > max_temporal_baseline:
                    break  # Scenes are sorted, no need to check further

                if temporal_baseline >= 12:  # Minimum 12 days (1 orbit cycle)
                    pairs.append((master, slave))
                    logger.debug(
                        f"Pair: {master.scene_id} - {slave.scene_id} "
                        f"(Δt={temporal_baseline} days)"
                    )

        logger.info(f"Created {len(pairs)} interferogram pairs")
        return pairs

    def get_coverage_info(self, scenes: List[SARScene]) -> dict:
        """
        Get coverage statistics for scenes.

        Args:
            scenes: List of SARScene objects

        Returns:
            Dictionary with coverage statistics
        """
        if not scenes:
            return {}

        dates = [s.acquisition_date for s in scenes]
        orbits = set(s.orbit_direction for s in scenes)
        paths = set(s.path_number for s in scenes)

        return {
            'total_scenes': len(scenes),
            'date_range': {
                'start': min(dates).isoformat(),
                'end': max(dates).isoformat(),
                'span_days': (max(dates) - min(dates)).days
            },
            'orbit_directions': list(orbits),
            'paths': list(paths),
            'average_interval_days': (max(dates) - min(dates)).days / len(scenes) if len(scenes) > 1 else 0,
            'total_size_gb': sum(s.file_size for s in scenes) / (1024**3)
        }


def main():
    """CLI entry point for testing"""
    import argparse

    parser = argparse.ArgumentParser(description='Search and download Sentinel-1 data')
    parser.add_argument('--bbox', nargs=4, type=float, required=True,
                        help='Bounding box: min_lon min_lat max_lon max_lat')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', default='data/raw/sentinel1', help='Output directory')
    parser.add_argument('--download', action='store_true', help='Download scenes')
    parser.add_argument('--max-scenes', type=int, default=10, help='Max scenes to download')

    args = parser.parse_args()

    downloader = SentinelDownloader()

    # Search
    scenes = downloader.search(
        bbox=args.bbox,
        start_date=args.start,
        end_date=args.end
    )

    # Print coverage info
    info = downloader.get_coverage_info(scenes)
    print("\n=== Coverage Info ===")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Print scenes
    print(f"\n=== Found Scenes ({len(scenes)}) ===")
    for s in scenes[:10]:
        print(f"  {s.acquisition_date.date()} | {s.orbit_direction:10} | {s.scene_id}")

    if len(scenes) > 10:
        print(f"  ... and {len(scenes) - 10} more")

    # Download if requested
    if args.download and scenes:
        to_download = scenes[:args.max_scenes]
        print(f"\nDownloading {len(to_download)} scenes...")
        downloader.download(to_download, args.output)


if __name__ == "__main__":
    main()
