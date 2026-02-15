"""Data acquisition modules for satellite and external data."""

# Conditional imports for optional dependencies
try:
    from .sentinel_downloader import SentinelDownloader
except ImportError:
    SentinelDownloader = None

try:
    from .dem_downloader import DEMDownloader
except ImportError:
    DEMDownloader = None

from .weather_fetcher import JMAWeatherFetcher

__all__ = ['SentinelDownloader', 'DEMDownloader', 'JMAWeatherFetcher']
