"""
Publishes the AI building footprints (tagged with FEMA flood zone) and the
flood zones to ArcGIS Online as a public web map. Uses the account ArcGIS Pro
is signed in to; no credentials in code. Re-running replaces the old items.
Run make_aprx.py first (it creates ai_buildings_flood).
"""
import json
import os
import sys
import zipfile

import arcpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GDB = os.path.join(ROOT, "data", "work.gdb")
WEB = os.path.join(ROOT, "data", "web.gdb")
TITLE = "Lafayette LA: AI Building Footprints and Flood Exposure"
REPO = "https://github.com/shristikarkiii/lafayette-geoai-buildings"
S = json.load(open(os.path.join(ROOT, "outputs", "summary.json")))
arcpy.env.overwriteOutput = True

if arcpy.Exists(WEB):
    arcpy.management.Delete(WEB)
arcpy.management.CreateFileGDB(os.path.dirname(WEB), "web.gdb")
b = os.path.join(WEB, "ai_buildings")
arcpy.management.CopyFeatures(os.path.join(GDB, "ai_buildings_flood"), b)
keep = {"uid", "Confidence", "flood_zone"}
drop = [f.name for f in arcpy.ListFields(b)
        if f.name not in keep and not f.required]
arcpy.management.DeleteField(b, drop)
arcpy.management.AddField(b, "area_m2", "DOUBLE")
arcpy.management.CalculateField(b, "area_m2", "round(!SHAPE.AREA!, 1)", "PYTHON3")
arcpy.management.CopyFeatures(os.path.join(GDB, "flood_zones"),
                              os.path.join(WEB, "flood_zones"))
arcpy.management.CopyFeatures(os.path.join(GDB, "city"), os.path.join(WEB, "city"))

from arcgis.gis import GIS  # noqa: E402
gis = GIS("pro")
base = os.path.join(ROOT, "data", "lafayette_web")
arcpy.management.ClearWorkspaceCache()
with zipfile.ZipFile(base + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
    for fn in os.listdir(WEB):
        if not fn.endswith(".lock"):            # live schema locks can't be read
            z.write(os.path.join(WEB, fn), os.path.join("web.gdb", fn))
for old in gis.content.search(f'title:"{TITLE}" AND owner:{gis.users.me.username}'):
    old.delete()
folder = gis.content.folders.get()
tags = "GeoAI,deep learning,building footprints,flood,Lafayette,Louisiana,NAIP"
src = folder.add({"title": TITLE, "type": "File Geodatabase", "tags": tags},
                 file=base + ".zip").result()
fl = src.publish({"name": "lafayette_ai_buildings_flood"})
names = {lyr.properties.name: i for i, lyr in enumerate(fl.layers)}


def sfs(rgb, a=255, outline=None):
    return {"type": "esriSFS", "style": "esriSFSSolid", "color": rgb + [a],
            "outline": outline or {"type": "esriSLS", "style": "esriSLSNull",
                                   "color": [0, 0, 0, 0], "width": 0}}


bld_r = {"type": "uniqueValue", "field1": "flood_zone", "uniqueValueInfos": [
    {"value": v, "label": v, "symbol": sfs(c)} for v, c in (
        ("1% annual chance (100-yr)", [239, 71, 111]),
        ("0.2% annual chance (500-yr)", [255, 166, 0]),
        ("Minimal / unmapped", [134, 142, 150]))]}
fz_r = {"type": "uniqueValue", "field1": "SFHA_TF", "uniqueValueInfos": [
    {"value": "T", "label": "FEMA 1% annual chance flood zone",
     "symbol": sfs([78, 168, 222], 90)}],
    "defaultSymbol": sfs([0, 0, 0], 0), "defaultLabel": "Other"}
city_r = {"type": "simple", "symbol": {
    "type": "esriSFS", "style": "esriSFSNull", "color": [0, 0, 0, 0],
    "outline": {"type": "esriSLS", "style": "esriSLSSolid",
                "color": [31, 35, 40, 255], "width": 1.5}}}


def op(name, title, renderer, popup, min_scale=0):
    return {"id": name, "title": title, "url": f"{fl.url}/{names[name]}",
            "layerType": "ArcGISFeatureLayer", "itemId": fl.id,
            "visibility": True, "opacity": 1, "minScale": min_scale,
            "layerDefinition": {"drawingInfo": {"renderer": renderer}},
            "popupInfo": popup}


e = arcpy.Describe(os.path.join(WEB, "city")).extent.projectAs(arcpy.SpatialReference(4326))
webmap = {
    "operationalLayers": [
        op("flood_zones", "FEMA flood zones", fz_r,
           {"title": "Zone {FLD_ZONE}", "fieldInfos": [
               {"fieldName": "ZONE_SUBTY", "label": "Subtype", "visible": True}]}),
        op("ai_buildings", "AI-detected buildings (NAIP 2023)", bld_r,
           {"title": "{flood_zone}", "fieldInfos": [
               {"fieldName": "Confidence", "label": "Model confidence", "visible": True,
                "format": {"places": 2}},
               {"fieldName": "area_m2", "label": "Footprint area (m²)", "visible": True,
                "format": {"places": 0, "digitSeparator": True}}]}),
        op("city", "City of Lafayette", city_r, None)],
    "baseMap": {"title": "Imagery Hybrid", "baseMapLayers": [
        {"id": "img", "title": "World Imagery", "layerType": "ArcGISTiledMapServiceLayer",
         "url": "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer",
         "visibility": True, "opacity": 1}]},
    "spatialReference": {"wkid": 102100, "latestWkid": 3857},
    "version": "2.31", "authoringApp": "ArcGIS API for Python"}
f = S["flood"]["AI (NAIP 2023)"]
wm = folder.add({
    "title": TITLE, "type": "Web Map", "text": json.dumps(webmap), "tags": tags,
    "extent": f"{e.XMin},{e.YMin},{e.XMax},{e.YMax}",
    "snippet": f"{f['buildings']:,} buildings found by deep learning on 2023 aerial "
               f"imagery; {f['in_1pct_floodplain']:,} sit in FEMA's 1% floodplain.",
    "description": f"Method, validation and code: {REPO}"}).result()
for item in (fl, wm):
    item.sharing.sharing_level = "EVERYONE"
src.delete()
print("web map: https://www.arcgis.com/apps/mapviewer/index.html?webmap=" + wm.id)
sys.stdout.flush()
os._exit(0)
