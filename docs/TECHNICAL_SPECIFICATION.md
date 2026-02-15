# SlopeGuard Technical Specification
# SlopeGuard 技術仕様書

**Version / バージョン**: 0.1.0
**Last Updated / 最終更新**: 2026-02-11
**Target User / 対象ユーザー**: NEXCO Central Japan / NEXCO中日本

---

## Table of Contents / 目次

1. [System Overview / システム概要](#1-system-overview--システム概要)
2. [Architecture / アーキテクチャ](#2-architecture--アーキテクチャ)
3. [Data Sources / データソース](#3-data-sources--データソース)
4. [Risk Calculation Algorithm / リスク計算アルゴリズム](#4-risk-calculation-algorithm--リスク計算アルゴリズム)
5. [API Specification / API仕様](#5-api-specification--api仕様)
6. [Dashboard / ダッシュボード](#6-dashboard--ダッシュボード)
7. [Deployment / デプロイメント](#7-deployment--デプロイメント)
8. [Data Flow / データフロー](#8-data-flow--データフロー)

---

## 1. System Overview / システム概要

### English

**SlopeGuard** is a satellite-based highway slope failure prediction system designed for Japanese expressway operators. The system integrates multiple data sources including:

- **Sentinel-1 InSAR**: Satellite radar for millimeter-level ground deformation detection
- **Digital Elevation Model (DEM)**: SRTM 30m resolution terrain analysis
- **Weather Data**: Japan Meteorological Agency (JMA) AMeDAS precipitation data
- **Geological Data**: Geological Survey of Japan (GSI) classifications
- **Historical Records**: MLIT/NEXCO past disaster database
- **Highway Geometry**: OpenStreetMap expressway coordinates

The system calculates risk scores (0-100) for monitored slope segments and classifies them into four alert levels for prioritized field inspection.

### 日本語

**SlopeGuard**は、日本の高速道路事業者向けに設計された衛星ベースの斜面崩壊予測システムです。以下の複数のデータソースを統合しています：

- **Sentinel-1 InSAR**: ミリメートル精度の地盤変動検出用衛星レーダー
- **数値標高モデル（DEM）**: SRTM 30m解像度の地形解析
- **気象データ**: 気象庁AMeDASの降水量データ
- **地質データ**: 産業技術総合研究所（GSI）の地質分類
- **災害履歴**: 国土交通省/NEXCO過去災害データベース
- **高速道路形状**: OpenStreetMapの高速道路座標

本システムは、監視対象の斜面セグメントごとにリスクスコア（0-100）を算出し、優先的な現地点検のために4段階の警戒レベルに分類します。

---

## 2. Architecture / アーキテクチャ

### Directory Structure / ディレクトリ構成

```
slopeguard/
├── main.py                          # CLI entry point / CLIエントリーポイント
├── config/
│   └── config.yaml                  # System configuration / システム設定
├── src/
│   ├── data_acquisition/            # Data loaders / データ取得
│   │   ├── sentinel_downloader.py   # Sentinel-1 SAR download
│   │   ├── dem_downloader.py        # DEM download
│   │   ├── weather_fetcher.py       # JMA AMeDAS API
│   │   ├── osm_loader.py            # OpenStreetMap loader
│   │   └── real_data_loader.py      # SRTM/Geology/History
│   ├── processing/
│   │   └── insar_reader.py          # InSAR data reader
│   ├── analytics/
│   │   └── risk_calculator.py       # Risk scoring engine
│   ├── api/
│   │   └── main.py                  # FastAPI REST API
│   └── utils/
│       ├── geo_utils.py             # Geospatial utilities
│       └── visualization.py         # Map generation
├── data/
│   ├── raw/                         # Raw data storage
│   ├── processed/                   # Processed outputs
│   └── external/                    # External data cache
└── output/
    └── dashboard.html               # Interactive dashboard
```

### Technology Stack / 技術スタック

| Component | Technology | Purpose |
|-----------|------------|---------|
| Backend API | FastAPI + Uvicorn | REST API server |
| Risk Engine | Python + NumPy | Risk calculation |
| Geospatial | Rasterio, GeoPandas, Shapely | Spatial analysis |
| InSAR Processing | ESA SNAP GPT | Interferogram generation |
| Frontend | HTML + Leaflet.js | Interactive map dashboard |
| Data Format | GeoJSON, BEAM-DIMAP, GeoTIFF | Data interchange |

---

## 3. Data Sources / データソース

### 3.1 Sentinel-1 InSAR / Sentinel-1 InSAR

| Attribute | Value |
|-----------|-------|
| Satellite | Sentinel-1A/B (ESA) |
| Band | C-band (5.405 GHz) |
| Wavelength | 5.5465 cm |
| Resolution | 5m × 20m (IW mode) |
| Revisit Period | 12 days |
| Data Provider | Alaska Satellite Facility (ASF) |
| Product Type | SLC (Single Look Complex) |

**Displacement Calculation / 変位計算:**
```
Displacement (mm) = Phase (rad) × λ / (4π)
                  = Phase × 5.5465 / (4π)
                  = Phase × 0.441 mm/rad
```

### 3.2 Digital Elevation Model / 数値標高モデル

| Attribute | Value |
|-----------|-------|
| Source | SRTM 1 Arc-Second |
| Resolution | 30 meters |
| Format | HGT (signed int16, big-endian) |
| Tile Size | 3601 × 3601 pixels (1° × 1°) |
| Void Value | -32768 |

**Slope Calculation (Horn's Method) / 傾斜計算（Hornの方法）:**
```
dz/dx = ((z3 + 2×z6 + z9) - (z1 + 2×z4 + z7)) / (8 × cell_size)
dz/dy = ((z1 + 2×z2 + z3) - (z7 + 2×z8 + z9)) / (8 × cell_size)
slope_degrees = arctan(√(dz/dx² + dz/dy²)) × 180/π
```

### 3.3 Weather Data / 気象データ

| Attribute | Value |
|-----------|-------|
| Source | Japan Meteorological Agency (JMA) |
| Network | AMeDAS (Automated Meteorological Data Acquisition System) |
| Parameters | Precipitation, Temperature, Humidity, Wind |
| Update Interval | 10 minutes |
| API Endpoint | `https://www.jma.go.jp/bosai/amedas/` |

**Monitored Parameters / 監視パラメータ:**
- 48-hour cumulative precipitation (mm) / 48時間累積降水量
- 7-day cumulative precipitation (mm) / 7日間累積降水量

### 3.4 Geological Data / 地質データ

| Attribute | Value |
|-----------|-------|
| Source | Geological Survey of Japan (GSI/AIST) |
| Classification | Rock/Soil type |
| Format | Vector database |

**Geology Risk Classifications / 地質リスク分類:**

| Geology Type | Risk Score | Stability |
|--------------|------------|-----------|
| Granite (花崗岩) | 15 | Stable |
| Andesite (安山岩) | 20 | Stable |
| Basalt (玄武岩) | 20 | Stable |
| Sandstone (砂岩) | 35 | Moderate |
| Shale (頁岩) | 45 | Moderate |
| Mudstone (泥岩) | 55 | Unstable |
| Tuff (凝灰岩) | 50 | Unstable |
| Volcanic Ash (火山灰) | 65 | Very Unstable |
| Colluvium (崖錐) | 75 | Very Unstable |
| Alluvium (沖積層) | 60 | Unstable |
| Fill (盛土) | 85 | Critical |
| Landslide Deposit (地すべり堆積物) | 90 | Critical |

### 3.5 Historical Disaster Data / 災害履歴データ

| Attribute | Value |
|-----------|-------|
| Source | MLIT, NEXCO Central Japan |
| Coverage | Past slope failures on Tomei/Chuo Expressway |
| Parameters | Location, Date, Type, Scale |

---

## 4. Risk Calculation Algorithm / リスク計算アルゴリズム

### 4.1 Weighted Factor Model / 重み付け要因モデル

The risk score is calculated as a weighted sum of six factors:
リスクスコアは6つの要因の重み付け合計として計算されます：

| Factor / 要因 | Weight / 重み | Description / 説明 |
|---------------|---------------|---------------------|
| Deformation Rate / 変動速度 | 35% | InSAR velocity (mm/year) |
| Deformation Acceleration / 変動加速度 | 20% | Rate of change (mm/year²) |
| Slope Angle / 傾斜角 | 15% | From DEM (degrees) |
| Rainfall / 降雨 | 15% | 48-hour precipitation (mm) |
| Geology / 地質 | 10% | Rock/Soil stability |
| Historical Events / 災害履歴 | 5% | Past disasters nearby |

**Formula / 計算式:**
```
Risk Score = Σ (Factor_Score × Weight)
           = 0.35×Def_Rate + 0.20×Accel + 0.15×Slope
           + 0.15×Rain + 0.10×Geo + 0.05×History
```

### 4.2 Factor Scoring Functions / 要因スコアリング関数

#### Deformation Rate (mm/year) / 変動速度

| Rate | Score |
|------|-------|
| < 2 | 0-10 |
| 2-5 | 10-30 |
| 5-10 | 30-60 |
| 10-20 | 60-80 |
| 20-30 | 80-90 |
| > 30 | 90-100 |

#### Slope Angle (degrees) / 傾斜角

| Angle | Score |
|-------|-------|
| < 10° | 0-10 |
| 10-20° | 10-30 |
| 20-30° | 30-50 |
| 30-40° | 50-80 |
| 40-50° | 80-95 |
| > 50° | 95-100 |

#### 48-hour Rainfall (mm) / 48時間降雨量

| Rainfall | Score |
|----------|-------|
| < 10 | 0-10 |
| 10-30 | 10-40 |
| 30-80 | 40-80 |
| 80-150 | 80-94 |
| > 150 | 94-100 |

### 4.3 Risk Levels / リスクレベル

| Level | Score Range | Japanese | English | Action |
|-------|-------------|----------|---------|--------|
| GREEN | 0-25 | 異常なし | Normal | Continue monitoring |
| YELLOW | 26-50 | 経過観察 | Caution | Enhanced monitoring |
| ORANGE | 51-75 | 要注意 | Warning | Field inspection recommended |
| RED | 76-100 | 要対応 | Critical | Immediate action required |

### 4.4 Data Quality Check / データ品質チェック

- **Minimum Coherence / 最小コヒーレンス**: 0.3
- If coherence < 0.3, result marked as "UNKNOWN" (unreliable)
- コヒーレンスが0.3未満の場合、結果は「不明」（信頼性低）とマークされます

---

## 5. API Specification / API仕様

### 5.1 Base URL

```
http://localhost:8000
```

### 5.2 Endpoints / エンドポイント

#### GET /api/v1/segments

Get all monitored segments with risk scores.
全監視セグメントのリスクスコアを取得。

**Query Parameters / クエリパラメータ:**

| Parameter | Type | Description |
|-----------|------|-------------|
| level | string | Filter by risk level (green/yellow/orange/red) |
| min_score | int | Minimum risk score (0-100) |
| limit | int | Maximum results (1-1000) |

**Response / レスポンス:**
```json
[
  {
    "segment_id": "SLOPE_0001",
    "lat": 35.4521,
    "lon": 139.0234,
    "score": 67,
    "level": "orange",
    "message": "【警戒】現地点検を推奨します",
    "coherence": 0.85,
    "timestamp": "2026-02-11T10:30:00Z",
    "is_reliable": true,
    "insights": [
      {
        "icon": "📡",
        "label": "衛星観測",
        "value": "地盤変動を検出",
        "detail": "年間 15mm の動き",
        "severity": "medium"
      }
    ]
  }
]
```

#### GET /api/v1/segments/{segment_id}

Get specific segment by ID.
特定セグメントをIDで取得。

#### GET /api/v1/alerts

Get active alerts (orange and red level segments).
アクティブなアラート（オレンジ・赤レベル）を取得。

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| level | string | Filter by alert level |
| acknowledged | bool | Filter by acknowledgment status |

#### GET /api/v1/geojson

Get all segments as GeoJSON FeatureCollection.
全セグメントをGeoJSON形式で取得。

**Response:**
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [139.0234, 35.4521]
      },
      "properties": {
        "segment_id": "SLOPE_0001",
        "score": 67,
        "level": "orange",
        "message": "...",
        "coherence": 0.85
      }
    }
  ],
  "metadata": {
    "total_features": 140,
    "generated_at": "2026-02-11T10:30:00Z"
  }
}
```

#### GET /api/v1/stats

Get system statistics.
システム統計を取得。

**Response:**
```json
{
  "total_segments": 140,
  "segments_by_level": {
    "red": 2,
    "orange": 15,
    "yellow": 45,
    "green": 78
  },
  "average_score": 32.5,
  "high_risk_count": 17,
  "last_updated": "2026-02-11T10:30:00Z"
}
```

#### GET /dashboard

Serve interactive dashboard HTML.
インタラクティブダッシュボードHTMLを提供。

#### GET /health

Health check endpoint.
ヘルスチェック用エンドポイント。

### 5.3 Response Models / レスポンスモデル

#### RiskScore

| Field | Type | Description |
|-------|------|-------------|
| segment_id | string | Unique segment identifier |
| lat | float | Latitude (WGS84) |
| lon | float | Longitude (WGS84) |
| score | int | Risk score (0-100) |
| level | string | Risk level (green/yellow/orange/red) |
| message | string | Human-readable message (Japanese) |
| coherence | float | InSAR data quality (0-1) |
| timestamp | datetime | Calculation timestamp |
| is_reliable | bool | Data reliability flag |
| insights | array | Data-driven insights for display |

---

## 6. Dashboard / ダッシュボード

### 6.1 Features / 機能

| Feature | Description (EN) | Description (JP) |
|---------|------------------|------------------|
| Bilingual UI | English/Japanese toggle | 日英切り替え |
| Risk Summary | Count by risk level | リスクレベル別件数 |
| Alert List | Actionable items list | 対応必要箇所一覧 |
| Interactive Map | Leaflet.js with markers | マーカー付きインタラクティブ地図 |
| SAR Coverage | Toggle satellite coverage area | 衛星観測範囲表示切替 |
| Data Insights | Per-segment analysis display | セグメント別分析表示 |

### 6.2 Map Layers / 地図レイヤー

- **Base Maps**: Esri Satellite, OpenStreetMap
- **Markers**: Color-coded by risk level
- **Popup**: Segment details with insights
- **Legend**: Risk level color reference

### 6.3 Color Scheme / カラースキーム

| Level | Color | Hex Code |
|-------|-------|----------|
| Critical (RED) | Red | #ef4444 |
| Warning (ORANGE) | Orange | #f97316 |
| Caution (YELLOW) | Yellow | #eab308 |
| Normal (GREEN) | Green | #22c55e |
| Unknown | Gray | #9ca3af |

---

## 7. Deployment / デプロイメント

### 7.1 Requirements / 要件

**System Requirements / システム要件:**
- Python 3.11+
- 8GB+ RAM (for InSAR processing)
- ESA SNAP 9.0+ (optional, for InSAR processing)

**Python Dependencies / Python依存関係:**
```
fastapi>=0.100.0
uvicorn>=0.23.0
numpy>=1.24.0
pandas>=2.0.0
rasterio>=1.3.0
geopandas>=0.13.0
shapely>=2.0.0
folium>=0.14.0
pydantic>=2.0.0
httpx>=0.24.0
pyyaml>=6.0
```

### 7.2 Running the API / API起動

```bash
# Development / 開発環境
cd /Users/cgs/dxaccelprog2025/slopeguard
source venv/bin/activate
uvicorn src.api.main:app --reload --port 8000

# Production / 本番環境
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 7.3 Docker Deployment / Dockerデプロイ

```bash
# Build and run / ビルドと実行
docker-compose up -d

# View logs / ログ確認
docker-compose logs -f slopeguard-api
```

**Docker Compose Configuration:**
```yaml
services:
  slopeguard-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./output:/app/output
    environment:
      - EARTHDATA_USER=${EARTHDATA_USER}
      - EARTHDATA_PASS=${EARTHDATA_PASS}
```

---

## 8. Data Flow / データフロー

### English

```
┌─────────────────────────────────────────────────────────────┐
│                  EXTERNAL DATA SOURCES                       │
├─────────────┬─────────────┬─────────────┬─────────────┬─────┤
│ Sentinel-1  │  SRTM DEM   │  JMA AMeDAS │    GSI      │ OSM │
│   (ESA)     │   (USGS)    │  (Weather)  │  (Geology)  │     │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴──┬──┘
       │             │             │             │         │
       ▼             ▼             ▼             ▼         ▼
┌─────────────────────────────────────────────────────────────┐
│              DATA ACQUISITION LAYER                          │
│  sentinel_downloader | real_data_loader | weather_fetcher   │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              PROCESSING LAYER                                │
│         insar_reader | slope calculation | geocoding         │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              ANALYTICS LAYER                                 │
│                   risk_calculator.py                         │
│  • 6-factor weighted model                                   │
│  • Score: 0-100                                              │
│  • Levels: GREEN / YELLOW / ORANGE / RED                     │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              API LAYER (FastAPI)                             │
│  /api/v1/segments | /api/v1/alerts | /api/v1/geojson        │
└────────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│   Interactive Dashboard │   │   External Systems      │
│   (Leaflet.js + HTML)   │   │   (NEXCO Operations)    │
└─────────────────────────┘   └─────────────────────────┘
```

### 日本語

```
┌─────────────────────────────────────────────────────────────┐
│                    外部データソース                           │
├─────────────┬─────────────┬─────────────┬─────────────┬─────┤
│ Sentinel-1  │  SRTM DEM   │  気象庁     │   地質調査所 │ OSM │
│   (ESA)     │   (USGS)    │  (AMeDAS)   │    (GSI)    │     │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴──┬──┘
       │             │             │             │         │
       ▼             ▼             ▼             ▼         ▼
┌─────────────────────────────────────────────────────────────┐
│                  データ取得レイヤー                           │
│  衛星ダウンローダー | 実データローダー | 気象フェッチャー       │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    処理レイヤー                               │
│       InSARリーダー | 傾斜計算 | ジオコーディング              │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   分析レイヤー                                │
│                 risk_calculator.py                           │
│  • 6要因重み付けモデル                                        │
│  • スコア: 0-100                                             │
│  • レベル: 異常なし / 経過観察 / 要注意 / 要対応               │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   APIレイヤー (FastAPI)                       │
│  /api/v1/segments | /api/v1/alerts | /api/v1/geojson        │
└────────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│ インタラクティブ         │   │   外部システム          │
│ ダッシュボード           │   │   (NEXCO運用系)         │
│ (Leaflet.js + HTML)     │   │                         │
└─────────────────────────┘   └─────────────────────────┘
```

---

## Appendix A: Monitoring Coverage / 付録A: 監視範囲

### Current Coverage / 現在の監視範囲

| Expressway | Coverage | Segments | Status |
|------------|----------|----------|--------|
| Tomei Expressway (東名高速) | Kanagawa - Shizuoka | ~140 | Active |
| Chuo Expressway (中央道) | Planned | - | Planned |

### Segment Selection Criteria / セグメント選定基準

- Slope angle ≥ 10° / 傾斜角10度以上
- Within 300m of highway centerline / 高速道路中心線から300m以内
- 200m sampling interval along highway / 高速道路沿い200m間隔でサンプリング

---

## Appendix B: Observation Schedule / 付録B: 観測スケジュール

| Parameter | Value |
|-----------|-------|
| Satellite Revisit | 12 days (Sentinel-1) |
| Weather Update | 10 minutes (JMA AMeDAS) |
| Risk Recalculation | 60 seconds (API cache) |
| Dashboard Refresh | 60 seconds (auto) |

---

## Appendix C: Accuracy and Limitations / 付録C: 精度と制限事項

### Accuracy / 精度

| Measurement | Accuracy |
|-------------|----------|
| InSAR Deformation | ±2-5 mm/year |
| DEM Elevation | ±5 m (SRTM) |
| Slope Angle | ±2° |
| Geolocation | ±30 m |

### Limitations / 制限事項

1. **InSAR Coherence Loss / InSARコヒーレンス低下**
   - Dense vegetation reduces coherence
   - 植生が密な地域ではコヒーレンスが低下

2. **Temporal Resolution / 時間分解能**
   - 12-day satellite revisit, not real-time
   - 12日周期の衛星観測であり、リアルタイムではない

3. **Weather Data Delay / 気象データ遅延**
   - 10-20 minute delay from real-time
   - リアルタイムから10-20分の遅延

4. **Geological Data Resolution / 地質データ解像度**
   - Regional classifications, may miss local variations
   - 地域分類であり、局所的な変化を捉えられない可能性

---

## Appendix D: Contact / 付録D: 連絡先

**Development Team / 開発チーム**: Highway DX Acceleration Program 2025
**Target Organization / 対象組織**: NEXCO Central Japan (NEXCO中日本)

---

*This document is automatically generated and may be updated as the system evolves.*
*本文書は自動生成されており、システムの進化に伴い更新される場合があります。*
