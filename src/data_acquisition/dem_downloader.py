"""
DEM (Digital Elevation Model) Downloader
=========================================

Downloads ALOS AW3D30 DEM tiles from JAXA or OpenTopography.

The ALOS World 3D - 30m (AW3D30) is a global digital surface model
with 30-meter resolution, derived from ALOS PRISM stereo images.
"""

import os
import logging
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

import requests
import numpy as np
from tqdm import tqdm

try:
    import rasterio
    from rasterio.merge import merge
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DEMTile:
    """Represents a DEM tile"""
    tile_name: str
    lat: int
    lon: int
    filepath: Optional[Path] = None


class DEMDownloader:
    """
    Download ALOS AW3D30 DEM tiles.

    The tiles are organized by 1x1 degree cells, named like:
    N035E138 (North 35, East 138)

    Example:
        >>> downloader = DEMDownloader()
        >>> tiles = downloader.get_required_tiles(bbox=[138.4, 35.5, 138.9, 36.1])
        >>> downloader.download_tiles(tiles, output_dir="data/raw/dem")
        >>> dem_path = downloader.merge_tiles(tiles, output_path="data/raw/dem/merged.tif")
    """

    # OpenTopography API (easier access, no registration needed for small areas)
    OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem"

    # Alternative: JAXA direct (requires registration)
    JAXA_BASE_URL = "https://www.eorc.jaxa.jp/ALOS/aw3d30/data/release_v2303"

    def __init__(self, output_dir: str = "data/raw/dem"):
        """
        Initialize DEM downloader.

        Args:
            output_dir: Default output directory for downloaded tiles
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_required_tiles(self, bbox: List[float]) -> List[DEMTile]:
        """
        Get list of required DEM tiles for bounding box.

        Args:
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]

        Returns:
            List of DEMTile objects needed to cover the bbox
        """
        min_lon, min_lat, max_lon, max_lat = bbox

        tiles = []
        for lat in range(int(np.floor(min_lat)), int(np.ceil(max_lat))):
            for lon in range(int(np.floor(min_lon)), int(np.ceil(max_lon))):
                tile_name = self._get_tile_name(lat, lon)
                tiles.append(DEMTile(
                    tile_name=tile_name,
                    lat=lat,
                    lon=lon
                ))

        logger.info(f"Required tiles: {[t.tile_name for t in tiles]}")
        return tiles

    def _get_tile_name(self, lat: int, lon: int) -> str:
        """
        Generate tile name from coordinates.

        Args:
            lat: Latitude (integer)
            lon: Longitude (integer)

        Returns:
            Tile name (e.g., "N035E138")
        """
        ns = 'N' if lat >= 0 else 'S'
        ew = 'E' if lon >= 0 else 'W'
        return f"{ns}{abs(lat):03d}{ew}{abs(lon):03d}"

    def download_from_opentopography(
        self,
        bbox: List[float],
        output_path: str,
        dem_type: str = "AW3D30"
    ) -> Path:
        """
        Download DEM directly from OpenTopography API.

        This is the easiest method - downloads a merged DEM for the entire bbox.
        No registration required for moderate-sized areas.

        Args:
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            output_path: Output file path
            dem_type: DEM type (AW3D30, SRTMGL1, etc.)

        Returns:
            Path to downloaded DEM file
        """
        min_lon, min_lat, max_lon, max_lat = bbox

        params = {
            "demtype": dem_type,
            "south": min_lat,
            "north": max_lat,
            "west": min_lon,
            "east": max_lon,
            "outputFormat": "GTiff"
        }

        logger.info(f"Downloading DEM from OpenTopography...")
        logger.info(f"  Bbox: {bbox}")
        logger.info(f"  DEM type: {dem_type}")

        response = requests.get(self.OPENTOPO_URL, params=params, stream=True)

        if response.status_code == 200:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            total_size = int(response.headers.get('content-length', 0))

            with open(output_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

            logger.info(f"Downloaded DEM to: {output_file}")
            return output_file
        else:
            raise Exception(f"Download failed: {response.status_code} - {response.text}")

    def download_tiles(
        self,
        tiles: List[DEMTile],
        output_dir: Optional[str] = None
    ) -> List[DEMTile]:
        """
        Download individual DEM tiles from JAXA.

        Note: JAXA requires registration. For easier access, use
        download_from_opentopography() instead.

        Args:
            tiles: List of DEMTile objects to download
            output_dir: Output directory (uses default if not specified)

        Returns:
            List of DEMTile objects with filepath set
        """
        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        downloaded = []
        for tile in tqdm(tiles, desc="Downloading tiles"):
            # Construct URL (JAXA structure)
            # Note: This is a simplified URL - actual structure may vary
            url = f"{self.JAXA_BASE_URL}/{tile.tile_name}/{tile.tile_name}_AVE_DSM.tif"

            output_path = out_dir / f"{tile.tile_name}_DSM.tif"

            if output_path.exists():
                logger.info(f"Tile already exists: {tile.tile_name}")
                tile.filepath = output_path
                downloaded.append(tile)
                continue

            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    tile.filepath = output_path
                    downloaded.append(tile)
                    logger.info(f"Downloaded: {tile.tile_name}")
                else:
                    logger.warning(f"Failed to download {tile.tile_name}: {response.status_code}")
            except Exception as e:
                logger.error(f"Error downloading {tile.tile_name}: {e}")

        return downloaded

    def merge_tiles(
        self,
        tiles: List[DEMTile],
        output_path: str
    ) -> Optional[Path]:
        """
        Merge multiple DEM tiles into a single file.

        Args:
            tiles: List of DEMTile objects (with filepath set)
            output_path: Output file path

        Returns:
            Path to merged DEM file
        """
        if not HAS_RASTERIO:
            logger.error("rasterio not installed. Cannot merge tiles.")
            return None

        filepaths = [t.filepath for t in tiles if t.filepath and t.filepath.exists()]

        if not filepaths:
            logger.error("No tile files found to merge")
            return None

        logger.info(f"Merging {len(filepaths)} tiles...")

        # Open all tiles
        src_files = [rasterio.open(fp) for fp in filepaths]

        # Merge
        mosaic, out_transform = merge(src_files)

        # Get metadata from first file
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "compress": "lzw"
        })

        # Write merged file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_file, 'w', **out_meta) as dest:
            dest.write(mosaic)

        # Close source files
        for src in src_files:
            src.close()

        logger.info(f"Merged DEM saved to: {output_file}")
        return output_file

    def calculate_slope(
        self,
        dem_path: str,
        output_path: str
    ) -> Optional[Path]:
        """
        Calculate slope from DEM.

        Args:
            dem_path: Path to DEM file
            output_path: Output slope file path

        Returns:
            Path to slope file
        """
        if not HAS_RASTERIO:
            logger.error("rasterio not installed.")
            return None

        logger.info("Calculating slope...")

        with rasterio.open(dem_path) as src:
            dem = src.read(1)
            transform = src.transform
            crs = src.crs

            # Calculate pixel size in meters (approximate for geographic CRS)
            pixel_size_x = abs(transform[0]) * 111320  # degrees to meters at equator
            pixel_size_y = abs(transform[4]) * 111320

            # Calculate gradients
            dy, dx = np.gradient(dem, pixel_size_y, pixel_size_x)

            # Calculate slope in degrees
            slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

            # Write output
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            meta = src.meta.copy()
            meta.update({
                'dtype': 'float32',
                'compress': 'lzw'
            })

            with rasterio.open(output_file, 'w', **meta) as dst:
                dst.write(slope.astype(np.float32), 1)

        logger.info(f"Slope saved to: {output_file}")
        return output_file


def main():
    """CLI entry point for testing"""
    import argparse

    parser = argparse.ArgumentParser(description='Download DEM data')
    parser.add_argument('--bbox', nargs=4, type=float, required=True,
                        help='Bounding box: min_lon min_lat max_lon max_lat')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--source', choices=['opentopo', 'jaxa'], default='opentopo',
                        help='Data source')

    args = parser.parse_args()

    downloader = DEMDownloader()

    if args.source == 'opentopo':
        dem_path = downloader.download_from_opentopography(
            bbox=args.bbox,
            output_path=args.output
        )
    else:
        tiles = downloader.get_required_tiles(bbox=args.bbox)
        downloader.download_tiles(tiles)
        dem_path = downloader.merge_tiles(tiles, output_path=args.output)

    # Calculate slope
    if dem_path:
        slope_path = str(dem_path).replace('.tif', '_slope.tif')
        downloader.calculate_slope(str(dem_path), slope_path)


if __name__ == "__main__":
    main()
