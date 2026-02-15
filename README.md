# SlopeGuard

**Highway Slope Failure Prediction System**

A satellite-based monitoring system for predicting slope failures on Japanese expressways using Sentinel-1 InSAR data, weather data, and AI risk scoring.

## Features

- **Satellite Monitoring**: Uses Sentinel-1 SAR data (free, 12-day revisit)
- **InSAR Analysis**: Detects mm-level ground deformation
- **AI Risk Scoring**: Combines multiple factors for risk assessment
- **Real-time Weather**: Integrates JMA precipitation data
- **Interactive Maps**: Web-based visualization dashboard
- **REST API**: Programmatic access to risk data

## Quick Start

### 1. Setup Environment

```bash
cd slopeguard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Demo

```bash
python main.py demo
```

This will:
- Create sample highway segments
- Fetch weather data from JMA
- Calculate simulated risk scores
- Generate an interactive map

### 3. Start API Server

```bash
python main.py api
```

Then open http://localhost:8000/docs for API documentation.

### 4. Download Real Data

```bash
# Set credentials (register at https://urs.earthdata.nasa.gov/)
export EARTHDATA_USER="your_username"
export EARTHDATA_PASS="your_password"

# Download data
python main.py download
```

## Project Structure

```
slopeguard/
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── config/
│   └── config.yaml            # Configuration
├── src/
│   ├── data_acquisition/      # Data downloaders
│   │   ├── sentinel_downloader.py
│   │   ├── dem_downloader.py
│   │   └── weather_fetcher.py
│   ├── processing/            # InSAR processing (TODO)
│   ├── analytics/             # Risk calculation
│   │   └── risk_calculator.py
│   ├── api/                   # REST API
│   │   └── main.py
│   └── utils/                 # Utilities
│       ├── geo_utils.py
│       └── visualization.py
├── data/                      # Data storage
├── output/                    # Generated outputs
└── tests/                     # Unit tests
```

## Risk Score Algorithm

The risk score (0-100) is calculated from weighted factors:

| Factor | Weight | Description |
|--------|--------|-------------|
| Deformation Rate | 35% | InSAR velocity (mm/year) |
| Acceleration | 20% | Change in deformation rate |
| Slope Angle | 15% | From DEM |
| Rainfall | 15% | 48-hour precipitation |
| Geology | 10% | Rock/soil type |
| Historical | 5% | Past events nearby |

### Risk Levels

- **GREEN** (0-25): Normal monitoring
- **YELLOW** (26-50): Increased monitoring
- **ORANGE** (51-75): Field inspection recommended
- **RED** (76-100): Immediate action required

## Data Sources

| Data | Source | Cost |
|------|--------|------|
| SAR Images | Sentinel-1 (ESA) | Free |
| DEM | ALOS AW3D30 (JAXA) | Free |
| Weather | JMA Open Data | Free |
| Geology | GSI Japan | Free |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/segments` | All monitored segments |
| GET | `/api/v1/segments/{id}` | Specific segment |
| GET | `/api/v1/alerts` | Active alerts |
| GET | `/api/v1/geojson` | GeoJSON export |
| GET | `/api/v1/stats` | Statistics |

## Development

### Run Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black src/

# Type checking
mypy src/
```

## License

This project is developed for the 2025 Highway DX Acceleration Program.

## Contact

[Your contact information]
