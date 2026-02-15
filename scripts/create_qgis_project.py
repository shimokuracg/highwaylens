"""
Create a QGIS project with satellite basemap and expressway shapefiles.
Run with QGIS's bundled Python.
"""
import sys
import os

sys.path.insert(0, "/Applications/QGIS-LTR.app/Contents/Resources/python")
sys.path.insert(0, "/Applications/QGIS-LTR.app/Contents/Resources/python/plugins")
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["PROJ_LIB"] = "/Applications/QGIS-LTR.app/Contents/Resources/proj"

from qgis.core import (
    QgsApplication,
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsRectangle,
)

# Initialise headless QGIS
QgsApplication.setPrefixPath("/Applications/QGIS-LTR.app/Contents/MacOS", True)
app = QgsApplication([], False)
app.initQgis()

project = QgsProject.instance()
crs_4326 = QgsCoordinateReferenceSystem("EPSG:4326")
crs_3857 = QgsCoordinateReferenceSystem("EPSG:3857")
project.setCrs(crs_3857)  # Web Mercator to match XYZ tiles

# --- Satellite basemap (ESRI World Imagery — no API key needed) ---
satellite_uri = (
    "type=xyz"
    "&url=https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    "&zmin=0&zmax=19"
)
satellite = QgsRasterLayer(satellite_uri, "ESRI Satellite", "wms")
if satellite.isValid():
    project.addMapLayer(satellite)
    print("Added: ESRI Satellite basemap")
else:
    print("WARNING: satellite layer failed to load")

# --- Expressway shapefiles ---
data_dir = os.path.join(
    os.path.dirname(__file__), os.pardir, "data", "external"
)
data_dir = os.path.abspath(data_dir)

layers_config = [
    # OSM-based layers (solid lines)
    ("tomei_expressway.shp", "Tomei (E1) - OSM", "255,80,80", 1.0, None),
    ("shintomei_expressway.shp", "Shin-Tomei (E1A) - OSM", "80,80,255", 1.0, None),
    # MLIT N06 layers (dashed lines for easy visual distinction)
    ("tomei_mlit.shp", "Tomei (E1) - MLIT 2024", "255,200,0", 1.2, "dash"),
    ("shintomei_mlit.shp", "Shin-Tomei (E1A) - MLIT 2024", "0,255,150", 1.2, "dash"),
]

from qgis.core import QgsSimpleLineSymbolLayer, QgsSingleSymbolRenderer
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt

for shp_name, display_name, color, width, style in layers_config:
    shp_path = os.path.join(data_dir, shp_name)
    layer = QgsVectorLayer(shp_path, display_name, "ogr")
    if layer.isValid():
        r, g, b = [int(x) for x in color.split(",")]
        symbol = layer.renderer().symbol()
        symbol.setColor(QColor(r, g, b))
        symbol.setWidth(width)
        if style == "dash":
            symbol.symbolLayer(0).setPenStyle(Qt.DashLine)

        project.addMapLayer(layer)
        print(f"Added: {display_name}  ({shp_path})")
    else:
        print(f"WARNING: failed to load {shp_path}")

# --- Set view extent to cover both expressways (in EPSG:3857) ---
transform = QgsCoordinateTransform(crs_4326, crs_3857, project)
# Bounding box covering both expressways in WGS84
extent_4326 = QgsRectangle(136.8, 34.7, 139.7, 35.7)
extent_3857 = transform.transformBoundingBox(extent_4326)

# Store extent in project metadata so QGIS opens to this view
from qgis.core import QgsReferencedRectangle
view_extent = QgsReferencedRectangle(extent_3857, crs_3857)
project.viewSettings().setDefaultViewExtent(view_extent)

# --- Save project ---
project_path = os.path.join(data_dir, "expressways.qgz")
project.write(project_path)
print(f"\nProject saved: {project_path}")

app.exitQgis()
