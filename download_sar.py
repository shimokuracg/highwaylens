#!/usr/bin/env python
"""Download Sentinel-1 SAR data for InSAR processing."""

import os
os.environ['EARTHDATA_USER'] = 'solafune_charles'
os.environ['EARTHDATA_PASS'] = 'zJ4MdsC4S+dE!!f'

from src.data_acquisition.sentinel_downloader import SentinelDownloader
from pathlib import Path

def main():
    downloader = SentinelDownloader()

    if not downloader.session:
        print("Authentication failed!")
        return

    print("Authentication successful!")

    # Search for scenes
    bbox = [138.4, 35.5, 138.9, 36.0]
    scenes = downloader.search(
        bbox=bbox,
        start_date='2024-10-01',
        end_date='2024-12-31',
        orbit_direction='DESCENDING',
        max_results=10
    )

    # Get pair
    pairs = downloader.create_pairs(scenes, max_temporal_baseline=24)
    master, slave = pairs[0]

    # Download both scenes
    print(f'\nDownloading 2 Sentinel-1 scenes (~9.2GB)...')
    print(f'  Master: {master.scene_id}')
    print(f'  Slave:  {slave.scene_id}')
    print()

    output_dir = Path('data/raw/sentinel1')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download sequentially to avoid multiprocessing issues
    downloaded = downloader.download([master, slave], str(output_dir), parallel=1)

    print(f'\nDownloaded {len(downloaded)} files to {output_dir}')
    for f in downloaded:
        print(f'  {f.name} ({f.stat().st_size/1e9:.1f}GB)')

if __name__ == "__main__":
    main()
