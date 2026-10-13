"""
Builds LafayetteBuildings.aprx with two maps and a letter-size layout:
  * "City"  : AI buildings coloured by FEMA flood zone over the city
  * "Detail": a ~400 m inset on the 2023 NAIP image, AI vs USA Structures
Exports outputs/figures/layout.png and layout.pdf.
"""
import glob
import json
import os
import shutil
import sys

import arcpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GDB = os.path.join(ROOT, "data", "work.gdb")
FIG = os.path.join(ROOT, "outputs", "figures")
APRX = os.path.join(ROOT, "LafayetteBuildings.aprx")
BLANK = (r"C:\Program Files\ArcGIS\Pro\Resources\ArcToolBox\Services"
         r"\routingservices\data\Blank.aprx")
S = json.load(open(os.path.join(ROOT, "outputs", "summary.json")))
T = S["best_threshold"]
arcpy.env.overwriteOutput = True

# Buildings tagged with their flood zone, for symbology and the web map.
bld = os.path.join(GDB, "ai_buildings_flood")
pts = os.path.join(arcpy.env.scratchGDB, "aprx_pts")
arcpy.management.FeatureToPoint(os.path.join(GDB, "eval_det_kept"), pts, "INSIDE")
sj = os.path.join(arcpy.env.scratchGDB, "aprx_sj")
arcpy.analysis.SpatialJoin(pts, os.path.join(GDB, "flood_zones"), sj,
                           "JOIN_ONE_TO_ONE", "KEEP_ALL")
zone = {}
for uid, sfha, sub in arcpy.da.SearchCursor(sj, ["uid", "SFHA_TF", "ZONE_SUBTY"]):
    zone[uid] = ("1% annual chance (100-yr)" if sfha == "T" else
                 "0.2% annual chance (500-yr)" if sub and "0.2 PCT" in sub.upper()
                 else "Minimal / unmapped")
arcpy.management.CopyFeatures(os.path.join(GDB, "eval_det_kept"), bld)
arcpy.management.AddField(bld, "flood_zone", "TEXT", field_length=40)
with arcpy.da.UpdateCursor(bld, ["uid", "flood_zone"]) as cur:
    for uid, _ in cur:
        cur.updateRow([uid, zone.get(uid, "Minimal / unmapped")])

shutil.copy(BLANK, APRX)
aprx = arcpy.mp.ArcGISProject(APRX)
aprx.defaultGeodatabase = GDB
aprx.homeFolder = ROOT
for m in aprx.listMaps():
    aprx.deleteItem(m)


def outline(lyr, rgb, width):
    s = lyr.symbology
    s.renderer.symbol.color = {"RGB": [0, 0, 0, 0]}
    s.renderer.symbol.outlineColor = {"RGB": rgb + [100]}
    s.renderer.symbol.outlineWidth = width
    lyr.symbology = s


# City map ------------------------------------------------------------------------
city = aprx.createMap("City")
city.spatialReference = arcpy.SpatialReference(26915)
try:
    city.addBasemap("Light Gray Canvas")
except Exception:
    pass
fz = city.addDataFromPath(os.path.join(GDB, "flood_zones"))
fz.name = "FEMA flood zones"
s = fz.symbology
s.updateRenderer("UniqueValueRenderer")
s.renderer.fields = ["SFHA_TF"]
for grp in s.renderer.groups:
    for it in grp.items:
        if it.values[0][0] == "T":
            it.label = "1% annual chance (100-yr)"
            it.symbol.color = {"RGB": [78, 168, 222, 70]}
        else:
            it.label = "Outside 1% zone"
            it.symbol.color = {"RGB": [0, 0, 0, 0]}
        it.symbol.outlineColor = {"RGB": [0, 0, 0, 0]}
fz.symbology = s
b = city.addDataFromPath(bld)
b.name = "AI-detected buildings"
s = b.symbology
s.updateRenderer("UniqueValueRenderer")
s.renderer.fields = ["flood_zone"]
colors = {"1% annual chance (100-yr)": [239, 71, 111, 100],
          "0.2% annual chance (500-yr)": [255, 166, 0, 100],
          "Minimal / unmapped": [150, 155, 160, 100]}
for grp in s.renderer.groups:
    for it in grp.items:
        it.symbol.color = {"RGB": colors.get(it.values[0][0], [150, 155, 160, 100])}
        it.label = "Building: " + it.values[0][0].split(" (")[0].replace(
            "Minimal / unmapped", "outside mapped flood zones").replace(
            "annual chance", "flood zone")
        it.symbol.outlineColor = {"RGB": [0, 0, 0, 0]}
b.symbology = s
cl = city.addDataFromPath(os.path.join(GDB, "city"))
cl.name = "City of Lafayette"
outline(cl, [31, 35, 40], 1.2)

# Detail map ------------------------------------------------------------------------
detail = aprx.createMap("Detail")
detail.spatialReference = arcpy.SpatialReference(26915)
for tif in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "naip", "*_city.tif"))):
    lyr = detail.addDataFromPath(tif)
    lyr.name = "NAIP 2023 (0.3 m)"
ref = detail.addDataFromPath(os.path.join(GDB, "eval_ref"))
ref.name = "USA Structures (FEMA/ORNL)"
outline(ref, [255, 214, 10], 1.5)
ai = detail.addDataFromPath(os.path.join(GDB, "eval_det_kept"))
ai.name = "AI detection (Esri Mask R-CNN)"
outline(ai, [0, 180, 216], 1.5)

# Layout ------------------------------------------------------------------------------
lyt = aprx.createLayout(11, 8.5, "INCH", "Lafayette buildings layout")
main = lyt.createMapFrame(arcpy.Extent(0.3, 0.55, 6.9, 7.55), city, "City frame")
main.camera.setExtent(arcpy.Describe(os.path.join(GDB, "city")).extent)
inset = lyt.createMapFrame(arcpy.Extent(7.1, 3.35, 10.7, 6.95), detail, "Detail frame")
# Inset: a residential block near the city centre, 400 m across.
cx, cy = 596151, 3337924          # a matched residential block (gallery example)
inset.camera.setExtent(arcpy.Extent(cx - 200, cy - 200, cx + 200, cy + 200))


def text(s, x, y, size):
    return aprx.createTextElement(lyt, arcpy.Point(x, y), "POINT", s, size)


f = S["flood"]["AI (NAIP 2023)"]
text("GeoAI Building Footprints & Flood Exposure", 0.3, 8.0, 20)
text("City of Lafayette, Louisiana  |  Esri pretrained Mask R-CNN on 2023 NAIP imagery",
     0.3, 7.72, 10)
text("Detail: cyan = AI, yellow = USA Structures", 7.1, 7.05, 9)
leg = lyt.createMapSurroundElement(
    arcpy.Point(7.1, 3.15), "LEGEND", main,
    aprx.listStyleItems("ArcGIS 2D", "LEGEND", "Title and Medium Text Legend")[0],
    "Legend")
leg.showTitle = False
leg.fittingStrategy = "AdjustFontSize"
leg.elementWidth, leg.elementHeight = 3.6, 1.3
leg.elementPositionX, leg.elementPositionY = 7.1, 3.25
ab = S["at_best"]
text(f"{f['buildings']:,} buildings detected\n"
     f"{f['in_1pct_floodplain']:,} ({f['share_1pct']:.1%}) in the 1% floodplain\n"
     f"Precision {ab['precision']:.0%}, recall {ab['recall']:.0%} vs USA Structures\n"
     "Blue shading: FEMA 1% annual-chance flood zone",
     7.1, 1.25, 10)
bar = lyt.createMapSurroundElement(
    arcpy.Point(0.4, 0.75), "SCALE_BAR", main,
    aprx.listStyleItems("ArcGIS 2D", "SCALE_BAR", "Alternating Scale Bar 1 Metric")[0])
bar.elementWidth = 2.8
cim = bar.getDefinition("V3")
cim.divisions, cim.subdivisions, cim.divisionsBeforeZero = 2, 0, 0
bar.setDefinition(cim)
arrow = lyt.createMapSurroundElement(
    arcpy.Point(6.45, 6.75), "NORTH_ARROW", main,
    aprx.listStyleItems("ArcGIS 2D", "NORTH_ARROW", "ArcGIS North 1")[0])
arrow.elementHeight = 0.55
text("Data: USDA NAIP 2023 (Microsoft Planetary Computer); FEMA NFHL; FEMA/ORNL USA "
     "Structures; Census 2023. NAD83 / UTM 15N. Author: Shristi Karki",
     0.3, 0.25, 7)

aprx.save()
lyt.exportToPNG(os.path.join(FIG, "layout.png"), resolution=150)
lyt.exportToPDF(os.path.join(FIG, "layout.pdf"), resolution=200)
print("saved", APRX)
sys.stdout.flush()
os._exit(0)
