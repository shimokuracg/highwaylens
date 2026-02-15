"""
Visualization Utilities
=======================

Functions for creating maps, charts, and visual outputs.
"""

import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import folium
    from folium.plugins import HeatMap
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# Risk level colors
RISK_COLORS = {
    'green': '#4CAF50',
    'yellow': '#FFC107',
    'orange': '#FF9800',
    'red': '#F44336',
    'unknown': '#9E9E9E'
}


def create_risk_map(
    segments: List[Dict[str, Any]],
    center: Optional[List[float]] = None,
    zoom: int = 10,
    output_path: Optional[str] = None
) -> Any:
    """
    Create interactive risk map using Folium.

    Args:
        segments: List of dicts with keys: lat, lon, score, level, segment_id
        center: Map center [lat, lon], auto-calculated if None
        zoom: Initial zoom level
        output_path: Optional path to save HTML

    Returns:
        Folium Map object
    """
    if not HAS_FOLIUM:
        logger.error("folium not installed. Run: pip install folium")
        return None

    # Calculate center if not provided
    if center is None:
        lats = [s['lat'] for s in segments]
        lons = [s['lon'] for s in segments]
        center = [np.mean(lats), np.mean(lons)]

    # Create map
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles='cartodbpositron'
    )

    # Add satellite layer option
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False
    ).add_to(m)

    # Add segments as colored markers/circles
    for seg in segments:
        level = seg.get('level', 'unknown')
        color = RISK_COLORS.get(level, RISK_COLORS['unknown'])
        score = seg.get('score', 0)

        # Popup content
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 200px;">
            <h4 style="margin: 0 0 10px 0;">{seg.get('segment_id', 'Unknown')}</h4>
            <table style="width: 100%;">
                <tr>
                    <td><b>Risk Score:</b></td>
                    <td style="text-align: right;">{score}</td>
                </tr>
                <tr>
                    <td><b>Level:</b></td>
                    <td style="text-align: right; color: {color}; font-weight: bold;">
                        {level.upper()}
                    </td>
                </tr>
                <tr>
                    <td><b>Location:</b></td>
                    <td style="text-align: right;">
                        {seg['lat']:.4f}, {seg['lon']:.4f}
                    </td>
                </tr>
            </table>
            <p style="margin-top: 10px; font-size: 12px;">
                {seg.get('message', '')}
            </p>
        </div>
        """

        # Circle size based on risk
        radius = 200 + score * 3  # 200-500m radius

        folium.Circle(
            location=[seg['lat'], seg['lon']],
            radius=radius,
            popup=folium.Popup(popup_html, max_width=300),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.6,
            weight=2
        ).add_to(m)

    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid gray; font-family: Arial;">
        <p style="margin: 0 0 5px 0;"><b>Risk Level</b></p>
        <p style="margin: 2px 0;"><span style="color: #4CAF50;">●</span> Green (0-25)</p>
        <p style="margin: 2px 0;"><span style="color: #FFC107;">●</span> Yellow (26-50)</p>
        <p style="margin: 2px 0;"><span style="color: #FF9800;">●</span> Orange (51-75)</p>
        <p style="margin: 2px 0;"><span style="color: #F44336;">●</span> Red (76-100)</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add layer control
    folium.LayerControl().add_to(m)

    # Save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(output_path)
        logger.info(f"Saved map to: {output_path}")

    return m


def create_heatmap(
    points: List[Dict[str, Any]],
    value_key: str = 'score',
    center: Optional[List[float]] = None,
    zoom: int = 10,
    output_path: Optional[str] = None
) -> Any:
    """
    Create heat map visualization.

    Args:
        points: List of dicts with lat, lon, and value_key
        value_key: Key for intensity value
        center: Map center
        zoom: Initial zoom
        output_path: Optional save path

    Returns:
        Folium Map object
    """
    if not HAS_FOLIUM:
        logger.error("folium not installed")
        return None

    if center is None:
        lats = [p['lat'] for p in points]
        lons = [p['lon'] for p in points]
        center = [np.mean(lats), np.mean(lons)]

    m = folium.Map(location=center, zoom_start=zoom, tiles='cartodbpositron')

    # Prepare heat data
    heat_data = [
        [p['lat'], p['lon'], p.get(value_key, 0) / 100]  # Normalize to 0-1
        for p in points
    ]

    HeatMap(
        heat_data,
        radius=20,
        blur=15,
        gradient={0.2: 'green', 0.4: 'yellow', 0.6: 'orange', 0.8: 'red'}
    ).add_to(m)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(output_path)
        logger.info(f"Saved heatmap to: {output_path}")

    return m


def plot_deformation(
    dates: List,
    values: List[float],
    segment_id: str = "",
    output_path: Optional[str] = None
) -> Any:
    """
    Plot deformation time series.

    Args:
        dates: List of dates
        values: List of deformation values (mm)
        segment_id: Segment identifier for title
        output_path: Optional save path

    Returns:
        Matplotlib figure or None
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib not installed")
        return None

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(dates, values, 'b-', linewidth=1.5, marker='o', markersize=4)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Deformation (mm)', fontsize=12)
    ax.set_title(f'Deformation Time Series - {segment_id}', fontsize=14)

    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        logger.info(f"Saved plot to: {output_path}")

    return fig


def plot_risk_distribution(
    scores: List[int],
    output_path: Optional[str] = None
) -> Any:
    """
    Plot distribution of risk scores.

    Args:
        scores: List of risk scores (0-100)
        output_path: Optional save path

    Returns:
        Matplotlib figure or None
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib not installed")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    colors = []
    for s in scores:
        if s <= 25:
            colors.append(RISK_COLORS['green'])
        elif s <= 50:
            colors.append(RISK_COLORS['yellow'])
        elif s <= 75:
            colors.append(RISK_COLORS['orange'])
        else:
            colors.append(RISK_COLORS['red'])

    ax1.hist(scores, bins=20, range=(0, 100), edgecolor='black', alpha=0.7)
    ax1.axvline(x=25, color='green', linestyle='--', label='Green/Yellow')
    ax1.axvline(x=50, color='orange', linestyle='--', label='Yellow/Orange')
    ax1.axvline(x=75, color='red', linestyle='--', label='Orange/Red')
    ax1.set_xlabel('Risk Score')
    ax1.set_ylabel('Count')
    ax1.set_title('Risk Score Distribution')
    ax1.legend()

    # Pie chart by level
    level_counts = {
        'Green': sum(1 for s in scores if s <= 25),
        'Yellow': sum(1 for s in scores if 25 < s <= 50),
        'Orange': sum(1 for s in scores if 50 < s <= 75),
        'Red': sum(1 for s in scores if s > 75)
    }

    pie_colors = [
        RISK_COLORS['green'],
        RISK_COLORS['yellow'],
        RISK_COLORS['orange'],
        RISK_COLORS['red']
    ]

    ax2.pie(
        level_counts.values(),
        labels=level_counts.keys(),
        colors=pie_colors,
        autopct='%1.1f%%',
        startangle=90
    )
    ax2.set_title('Segments by Risk Level')

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        logger.info(f"Saved plot to: {output_path}")

    return fig


def create_plotly_dashboard(
    segments: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> Any:
    """
    Create interactive Plotly dashboard.

    Args:
        segments: List of segment data dicts
        output_path: Optional path to save HTML

    Returns:
        Plotly figure or None
    """
    if not HAS_PLOTLY:
        logger.error("plotly not installed")
        return None

    import pandas as pd

    df = pd.DataFrame(segments)

    # Map view
    fig = px.scatter_mapbox(
        df,
        lat='lat',
        lon='lon',
        color='score',
        size='score',
        hover_name='segment_id',
        hover_data=['score', 'level'],
        color_continuous_scale=[
            (0, RISK_COLORS['green']),
            (0.25, RISK_COLORS['green']),
            (0.5, RISK_COLORS['yellow']),
            (0.75, RISK_COLORS['orange']),
            (1.0, RISK_COLORS['red'])
        ],
        mapbox_style='carto-positron',
        zoom=9,
        title='HighwayLens Risk Map'
    )

    fig.update_layout(
        height=600,
        margin={"r": 0, "t": 50, "l": 0, "b": 0}
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        logger.info(f"Saved dashboard to: {output_path}")

    return fig


def main():
    """Test visualization functions"""
    # Sample data
    sample_segments = [
        {'segment_id': 'SEG_001', 'lat': 35.70, 'lon': 138.60, 'score': 25, 'level': 'green', 'message': 'Low risk'},
        {'segment_id': 'SEG_002', 'lat': 35.72, 'lon': 138.62, 'score': 45, 'level': 'yellow', 'message': 'Moderate risk'},
        {'segment_id': 'SEG_003', 'lat': 35.74, 'lon': 138.64, 'score': 68, 'level': 'orange', 'message': 'Elevated risk'},
        {'segment_id': 'SEG_004', 'lat': 35.76, 'lon': 138.66, 'score': 82, 'level': 'red', 'message': 'High risk'},
        {'segment_id': 'SEG_005', 'lat': 35.71, 'lon': 138.58, 'score': 15, 'level': 'green', 'message': 'Low risk'},
    ]

    print("Creating visualizations...")

    # Create risk map
    if HAS_FOLIUM:
        m = create_risk_map(sample_segments, output_path='output/test_risk_map.html')
        print("Created risk map: output/test_risk_map.html")

    # Create distribution plot
    if HAS_MATPLOTLIB:
        scores = [s['score'] for s in sample_segments]
        plot_risk_distribution(scores, output_path='output/test_distribution.png')
        print("Created distribution plot: output/test_distribution.png")


if __name__ == "__main__":
    main()
