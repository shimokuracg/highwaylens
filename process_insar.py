#!/usr/bin/env python
"""
InSAR Processing with SNAP GPT
==============================

Process Sentinel-1 SLC pairs to generate interferograms.
"""

import os
import subprocess
from pathlib import Path
import zipfile

# SNAP GPT path
GPT_PATH = "/Applications/esa-snap/bin/gpt"

def extract_safe(zip_path: Path, output_dir: Path) -> Path:
    """Extract .SAFE directory from zip file."""
    safe_dir = None

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Find .SAFE directory name
        for name in zf.namelist():
            if name.endswith('.SAFE/'):
                safe_dir = name.rstrip('/')
                break

        if safe_dir:
            # Check if already extracted
            extracted_path = output_dir / safe_dir
            if extracted_path.exists():
                print(f"  Already extracted: {safe_dir}")
                return extracted_path

            print(f"  Extracting: {safe_dir}")
            zf.extractall(output_dir)
            return extracted_path

    raise ValueError(f"No .SAFE directory found in {zip_path}")

def run_insar_processing(master_zip: Path, slave_zip: Path, output_dir: Path):
    """Run InSAR processing using SNAP GPT."""

    print("=" * 60)
    print("SlopeGuard InSAR Processing")
    print("=" * 60)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract SAFE directories
    print("\n[1/3] Extracting Sentinel-1 products...")
    master_safe = extract_safe(master_zip, master_zip.parent)
    slave_safe = extract_safe(slave_zip, slave_zip.parent)

    print(f"  Master: {master_safe.name}")
    print(f"  Slave:  {slave_safe.name}")

    # Output file
    output_file = output_dir / "interferogram"

    # Graph file
    graph_file = Path(__file__).parent / "config" / "insar_graph.xml"

    print("\n[2/3] Running InSAR processing with SNAP...")
    print("  This may take 15-30 minutes...")
    print(f"  Graph: {graph_file}")
    print(f"  Output: {output_file}")

    # Build GPT command
    cmd = [
        GPT_PATH,
        str(graph_file),
        f"-PmasterFile={master_safe}",
        f"-PslaveFile={slave_safe}",
        f"-PoutputFile={output_file}",
        "-x",  # Clear cache
    ]

    print("\n  Command:", " ".join(cmd[:3]), "...")

    # Run processing
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode == 0:
            print("\n[3/3] Processing complete!")
            print(f"  Output: {output_file}.dim")

            # List output files
            if output_file.with_suffix('.dim').exists():
                print("\n  Output files:")
                for f in output_file.parent.glob("interferogram*"):
                    print(f"    {f.name}")

            return output_file.with_suffix('.dim')
        else:
            print(f"\n  Error: {result.stderr[:500]}")
            return None

    except subprocess.TimeoutExpired:
        print("\n  Error: Processing timed out (>1 hour)")
        return None
    except Exception as e:
        print(f"\n  Error: {e}")
        return None

def main():
    # Find downloaded Sentinel-1 files
    data_dir = Path("data/raw/sentinel1")
    zip_files = sorted(data_dir.glob("*.zip"))

    if len(zip_files) < 2:
        print(f"Error: Need at least 2 Sentinel-1 zip files in {data_dir}")
        print(f"Found: {[f.name for f in zip_files]}")
        return

    # Use first two files (master = earlier date, slave = later date)
    master_zip = zip_files[0]
    slave_zip = zip_files[1]

    print(f"Master: {master_zip.name}")
    print(f"Slave:  {slave_zip.name}")

    # Process
    output_dir = Path("data/processed/interferograms")
    result = run_insar_processing(master_zip, slave_zip, output_dir)

    if result:
        print("\n" + "=" * 60)
        print("InSAR processing completed successfully!")
        print(f"Interferogram: {result}")
        print("=" * 60)
    else:
        print("\nProcessing failed. Check SNAP installation and input files.")

if __name__ == "__main__":
    main()
