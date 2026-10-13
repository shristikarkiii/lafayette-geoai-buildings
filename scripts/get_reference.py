"""
Downloads the comparison layers for the City of Lafayette into data/work.gdb:

  usa_structures  FEMA / ORNL USA Structures footprints (Esri Living Atlas)
  flood_zones     FEMA National Flood Hazard Layer, flood hazard zones

Both come from public ArcGIS REST services, paged 2,000 features at a time.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import arcpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GDB = os.path.join(ROOT, "data", "work.gdb")
TMP = os.path.join(ROOT, "data", "raw", "rest")
UTM15 = arcpy.SpatialReference(26915)
SERVICES = {
    "usa_structures": "https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/"
                      "rest/services/USA_Structures_View/FeatureServer/0",
    "flood_zones": "https://hazards.fema.gov/arcgis/rest/services/public/"
                   "NFHL/MapServer/28",
}
FIELDS = {
    "usa_structures": "BUILD_ID,OCC_CLS,PRIM_OCC,SQMETERS,IMAGE_DATE,SOURCE",
    "flood_zones": "FLD_ZONE,ZONE_SUBTY,SFHA_TF,STATIC_BFE",
}
arcpy.env.overwriteOutput = True


def fetch(url, params, tries=5):
    data = urllib.parse.urlencode(params).encode()
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data), timeout=120) as r:
                return json.load(r)
        except Exception as e:                       # flaky public servers
            if k == tries - 1:
                raise
            print("  retry:", e, flush=True)
            time.sleep(5 * (k + 1))


def download(name, envelope):
    os.makedirs(TMP, exist_ok=True)
    url = SERVICES[name] + "/query"
    base = {"where": "1=1", "geometry": json.dumps(envelope),
            "geometryType": "esriGeometryEnvelope", "inSR": 4269,
            "spatialRel": "esriSpatialRelIntersects", "outFields": FIELDS[name],
            "outSR": 26915, "f": "json", "returnGeometry": "true"}
    ids = fetch(url, {**base, "returnIdsOnly": "true"})["objectIds"] or []
    ids.sort()
    print(f"{name}: {len(ids)} features", flush=True)
    parts = []
    for i in range(0, len(ids), 1000):
        chunk = ids[i:i + 1000]
        d = fetch(url, {**base, "where": "", "objectIds": ",".join(map(str, chunk))})
        if "error" in d:
            raise RuntimeError(d["error"])
        path = os.path.join(TMP, f"{name}_{i // 1000:04d}.json")
        with open(path, "w") as fh:
            json.dump(d, fh)
        parts.append(path)
        print(f"  {min(i + 1000, len(ids))}/{len(ids)}", flush=True)
    fcs = []
    for p in parts:
        fc = os.path.join(arcpy.env.scratchGDB, os.path.basename(p)[:-5])
        arcpy.conversion.JSONToFeatures(p, fc)
        fcs.append(fc)
    merged = os.path.join(arcpy.env.scratchGDB, name + "_all")
    arcpy.management.Merge(fcs, merged)
    return merged


def main():
    city = os.path.join(GDB, "city")
    e = arcpy.Describe(os.path.join(GDB, "city_wgs")).extent
    env = {"xmin": e.XMin, "ymin": e.YMin, "xmax": e.XMax, "ymax": e.YMax}
    for name in SERVICES:
        merged = download(name, env)
        out = os.path.join(GDB, name)
        if name == "usa_structures":
            # A building belongs to the city if its centre is inside it,
            # the same rule applied to the AI detections.
            lyr = arcpy.management.MakeFeatureLayer(merged, "ref_lyr")
            arcpy.management.SelectLayerByLocation(lyr, "HAVE_THEIR_CENTER_IN", city)
            arcpy.management.CopyFeatures(lyr, out)
            arcpy.management.Delete("ref_lyr")
        else:
            arcpy.analysis.PairwiseClip(merged, city, out)
        print(f"{name} in city: {arcpy.management.GetCount(out)[0]}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
