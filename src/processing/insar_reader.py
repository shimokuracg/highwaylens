"""
InSAR Data Reader
=================

Read processed interferogram data from SNAP and extract deformation values.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
import xml.etree.ElementTree as ET

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InSARReader:
    """
    Read InSAR interferogram data from SNAP BEAM-DIMAP format.

    The interferogram contains:
    - Coherence: Quality metric (0-1)
    - I/Q components: Real/imaginary parts of complex interferogram
    - Phase can be computed as: atan2(Q, I)
    - Displacement = phase * wavelength / (4 * pi)
    """

    # Sentinel-1 C-band wavelength in meters
    WAVELENGTH = 0.055465  # ~5.5 cm

    def __init__(self, dim_file: str):
        """
        Initialize reader with BEAM-DIMAP .dim file.

        Args:
            dim_file: Path to .dim file
        """
        self.dim_file = Path(dim_file)
        self.data_dir = self.dim_file.with_suffix('.data')

        if not self.dim_file.exists():
            raise FileNotFoundError(f"DIM file not found: {self.dim_file}")

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Parse metadata
        self.metadata = self._parse_dim()

        # Find data files
        self.coherence_file = None
        self.i_file = None
        self.q_file = None

        for f in self.data_dir.glob("*.img"):
            name = f.name.lower()
            if name.startswith("coh"):
                self.coherence_file = f
            elif name.startswith("i_"):
                self.i_file = f
            elif name.startswith("q_"):
                self.q_file = f

        logger.info(f"InSAR Reader initialized: {self.dim_file.name}")
        logger.info(f"  Coherence: {self.coherence_file.name if self.coherence_file else 'Not found'}")
        logger.info(f"  I component: {self.i_file.name if self.i_file else 'Not found'}")
        logger.info(f"  Q component: {self.q_file.name if self.q_file else 'Not found'}")

    def _parse_dim(self) -> dict:
        """Parse BEAM-DIMAP metadata."""
        tree = ET.parse(self.dim_file)
        root = tree.getroot()

        metadata = {}

        # Get raster dimensions
        raster_dim = root.find(".//Raster_Dimensions")
        if raster_dim is not None:
            metadata['width'] = int(raster_dim.findtext("NCOLS", "0"))
            metadata['height'] = int(raster_dim.findtext("NROWS", "0"))

        # Get geocoding
        geocoding = root.find(".//Coordinate_Reference_System")
        if geocoding is not None:
            metadata['crs'] = geocoding.findtext("GEO_TABLES", "WGS84")

        # Get geo-position
        geopos = root.find(".//Geoposition")
        if geopos is not None:
            # Try to get tie-point grid info
            pass

        return metadata

    def _read_envi_header(self, hdr_file: Path) -> dict:
        """Parse ENVI header file."""
        header = {}
        with open(hdr_file, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.split('=', 1)
                    header[key.strip()] = value.strip()
        return header

    def _read_img(self, img_file: Path) -> np.ndarray:
        """Read ENVI .img file as numpy array."""
        hdr_file = img_file.with_suffix('.hdr')
        header = self._read_envi_header(hdr_file)

        samples = int(header.get('samples', 0))
        lines = int(header.get('lines', 0))
        dtype_code = int(header.get('data type', 4))
        byte_order = int(header.get('byte order', 0))  # 0=little, 1=big

        # ENVI data type codes with byte order prefix
        # '<' = little-endian, '>' = big-endian
        prefix = '>' if byte_order == 1 else '<'

        dtype_map = {
            1: f'{prefix}u1',   # uint8
            2: f'{prefix}i2',   # int16
            3: f'{prefix}i4',   # int32
            4: f'{prefix}f4',   # float32
            5: f'{prefix}f8',   # float64
            12: f'{prefix}u2', # uint16
        }
        dtype = dtype_map.get(dtype_code, f'{prefix}f4')

        # Read binary data
        data = np.fromfile(img_file, dtype=dtype)
        data = data.reshape((lines, samples))

        return data

    def get_coherence(self) -> np.ndarray:
        """Read coherence map."""
        if self.coherence_file is None:
            raise ValueError("Coherence file not found")
        return self._read_img(self.coherence_file)

    def get_interferogram(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read I and Q components of interferogram."""
        if self.i_file is None or self.q_file is None:
            raise ValueError("Interferogram files not found")

        i_data = self._read_img(self.i_file)
        q_data = self._read_img(self.q_file)

        return i_data, q_data

    def get_phase(self) -> np.ndarray:
        """Compute wrapped phase from I/Q components."""
        i_data, q_data = self.get_interferogram()
        phase = np.arctan2(q_data, i_data)
        return phase

    def get_displacement_los(self) -> np.ndarray:
        """
        Compute line-of-sight displacement from phase.

        Returns:
            Displacement in mm (negative = away from satellite / subsidence)
        """
        phase = self.get_phase()

        # Displacement = phase * wavelength / (4 * pi)
        # Convert to mm
        displacement_m = phase * self.WAVELENGTH / (4 * np.pi)
        displacement_mm = displacement_m * 1000

        return displacement_mm

    def get_tie_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get latitude/longitude tie point grids for geocoding."""
        lat_file = self.data_dir / "tie_point_grids" / "latitude.img"
        lon_file = self.data_dir / "tie_point_grids" / "longitude.img"

        if lat_file.exists() and lon_file.exists():
            lat = self._read_img(lat_file)
            lon = self._read_img(lon_file)
            return lat, lon

        return None, None

    def get_value_at_location(
        self,
        lat: float,
        lon: float,
        data: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Get InSAR values at a specific lat/lon location.

        Args:
            lat: Latitude
            lon: Longitude
            data: Optional pre-loaded data array

        Returns:
            Dict with coherence, phase, displacement values
        """
        # Get tie points for geocoding
        lat_grid, lon_grid = self.get_tie_points()

        if lat_grid is None:
            logger.warning("No tie points available for geocoding")
            return {"coherence": 0.0, "displacement_mm": 0.0}

        # Find nearest pixel
        # Interpolate tie points to full resolution
        from scipy.ndimage import zoom

        coherence = self.get_coherence()
        h, w = coherence.shape

        # Zoom tie points to full resolution
        zoom_h = h / lat_grid.shape[0]
        zoom_w = w / lat_grid.shape[1]

        lat_full = zoom(lat_grid, (zoom_h, zoom_w), order=1)
        lon_full = zoom(lon_grid, (zoom_w, zoom_w), order=1)

        # Find closest pixel
        dist = np.sqrt((lat_full - lat)**2 + (lon_full - lon)**2)
        row, col = np.unravel_index(np.argmin(dist), dist.shape)

        # Get values
        coh = coherence[row, col]

        i_data, q_data = self.get_interferogram()
        phase = np.arctan2(q_data[row, col], i_data[row, col])
        displacement_mm = phase * self.WAVELENGTH / (4 * np.pi) * 1000

        return {
            "coherence": float(coh),
            "phase_rad": float(phase),
            "displacement_mm": float(displacement_mm),
            "row": row,
            "col": col
        }

    def extract_segment_values(
        self,
        segments: list,
        verbose: bool = True
    ) -> Dict[str, dict]:
        """
        Extract InSAR values for highway segments.

        Args:
            segments: List of segment objects with lat_center, lon_center
            verbose: Print progress

        Returns:
            Dict mapping segment_id to InSAR values
        """
        from scipy.ndimage import zoom

        # Load all data
        if verbose:
            print("Loading InSAR data...")

        coherence = self.get_coherence()
        i_data, q_data = self.get_interferogram()

        # Compute phase and displacement
        phase = np.arctan2(q_data, i_data)
        displacement = phase * self.WAVELENGTH / (4 * np.pi) * 1000  # mm

        # Get tie points
        lat_grid, lon_grid = self.get_tie_points()

        if lat_grid is None:
            logger.error("Cannot extract values: no geocoding information")
            return {}

        h, w = coherence.shape
        zoom_h = h / lat_grid.shape[0]
        zoom_w = w / lat_grid.shape[1]

        # Extract values for each segment
        results = {}

        if verbose:
            print(f"Extracting values for {len(segments)} segments using tie-point mapping...")

        for seg in segments:
            target_lat = seg.lat_center
            target_lon = seg.lon_center
            
            # Estimate row, col by searching tie points first (much faster than search in full lat/lon grid)
            dist_tie = (lat_grid - target_lat)**2 + (lon_grid - target_lon)**2
            tie_row, tie_col = np.unravel_index(np.argmin(dist_tie), dist_tie.shape)
            
            # Map tie point coordinates back to full image coordinates
            row, col = int(tie_row * zoom_h), int(tie_col * zoom_w)



            # Check if within bounds
            if 0 <= row < h and 0 <= col < w:
                # Average over small window (3x3) for robustness
                r1, r2 = max(0, row-1), min(h, row+2)
                c1, c2 = max(0, col-1), min(w, col+2)

                coh_val = np.nanmean(coherence[r1:r2, c1:c2])
                disp_val = np.nanmean(displacement[r1:r2, c1:c2])

                results[seg.cell_id] = {
                    "coherence": float(coh_val),
                    "displacement_mm": float(disp_val),
                    "row": row,
                    "col": col
                }
            else:
                results[seg.cell_id] = {
                    "coherence": 0.0,
                    "displacement_mm": 0.0,
                    "row": -1,
                    "col": -1
                }

        if verbose:
            print(f"Extracted values for {len(results)} segments")

        return results


def main():
    """Test InSAR reader."""
    dim_file = "data/processed/interferograms/interferogram.dim"

    if not Path(dim_file).exists():
        print(f"Interferogram not found: {dim_file}")
        return

    reader = InSARReader(dim_file)

    # Load coherence
    print("\nLoading coherence...")
    coh = reader.get_coherence()
    print(f"  Shape: {coh.shape}")
    print(f"  Range: {np.nanmin(coh):.3f} - {np.nanmax(coh):.3f}")
    print(f"  Mean: {np.nanmean(coh):.3f}")

    # Load displacement
    print("\nComputing displacement...")
    disp = reader.get_displacement_los()
    print(f"  Shape: {disp.shape}")
    print(f"  Range: {np.nanmin(disp):.1f} - {np.nanmax(disp):.1f} mm")
    print(f"  Std: {np.nanstd(disp):.1f} mm")

    # Test segment extraction
    print("\nTesting segment extraction...")
    from src.utils.geo_utils import create_highway_segments, CENTRAL_EXPRESSWAY_COORDS

    segments = create_highway_segments(
        CENTRAL_EXPRESSWAY_COORDS,
        segment_length_km=5.0,
        prefix="CHUO"
    )

    values = reader.extract_segment_values(segments)

    print("\nSegment values:")
    for seg_id, data in list(values.items())[:5]:
        print(f"  {seg_id}: coh={data['coherence']:.2f}, disp={data['displacement_mm']:.1f}mm")


if __name__ == "__main__":
    main()
