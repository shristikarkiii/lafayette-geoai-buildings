"""
Streams 2023 NAIP (0.3 m, 4-band) for the City of Lafayette from Microsoft
Planetary Computer. Each tile is read as a Cloud-Optimized GeoTIFF and only the
window overlapping the city is saved (RGB bands, which is what the model uses).
Already-downloaded windows are skipped, so the script can be re-run safely.

Run with ArcGIS Pro's Python (it ships GDAL):
  propy scripts/download_naip.py
"""
import json
import os
import sys
import urllib.request

from osgeo import gdal, ogr, osr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(RAW, "naip")
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
gdal.UseExceptions()
gdal.SetConfigOption("GDAL_HTTP_MAX_RETRY", "5")
gdal.SetConfigOption("GDAL_HTTP_RETRY_DELAY", "3")
gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")


def token():
    url = f"{STAC.replace('/stac/v1', '')}/sas/v1/token/naip"
    return json.load(urllib.request.urlopen(url))["token"]


def city_geometry_utm15():
    shp = os.path.join(RAW, "places", "cb_2023_22_place_500k.shp")
    ds = ogr.Open(shp)
    lyr = ds.GetLayer()
    lyr.SetAttributeFilter("NAME = 'Lafayette'")
    feat = lyr.GetNextFeature()        # keep a reference: GDAL frees it otherwise
    geom = feat.GetGeometryRef().Clone()
    src = lyr.GetSpatialRef()
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(26915)
    for s in (src, dst):
        s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    geom.Transform(osr.CoordinateTransformation(src, dst))
    return geom.Buffer(100)            # small margin so edge buildings are whole


def main():
    os.makedirs(OUT, exist_ok=True)
    ids = json.load(open(os.path.join(RAW, "city_tiles.json")))
    items = {f["id"]: f for f in json.load(open(os.path.join(RAW, "naip_items_2023.json")))}
    city = city_geometry_utm15()
    tok = token()
    for i, tid in enumerate(ids, 1):
        out = os.path.join(OUT, f"{tid}_city.tif")
        if os.path.exists(out):
            print(f"[{i}/{len(ids)}] {tid}: exists, skipped", flush=True)
            continue
        href = items[tid]["assets"]["image"]["href"]
        src = gdal.Open(f"/vsicurl/{href}?{tok}")
        gt = src.GetGeoTransform()
        tx0, ty1 = gt[0], gt[3]
        tx1, ty0 = tx0 + gt[1] * src.RasterXSize, ty1 + gt[5] * src.RasterYSize
        tile = ogr.CreateGeometryFromWkt(
            f"POLYGON(({tx0} {ty0},{tx1} {ty0},{tx1} {ty1},{tx0} {ty1},{tx0} {ty0}))")
        overlap = tile.Intersection(city)
        if overlap is None or overlap.IsEmpty() or overlap.GetArea() < 1e4:
            print(f"[{i}/{len(ids)}] {tid}: <0.01 km2 overlap, skipped", flush=True)
            continue
        x0, x1, y0, y1 = overlap.GetEnvelope()
        print(f"[{i}/{len(ids)}] {tid}: {overlap.GetArea() / 1e6:.1f} km2 of city, "
              f"window {(x1 - x0) / 1000:.1f} x {(y1 - y0) / 1000:.1f} km", flush=True)
        tmp = out + ".part"
        gdal.Translate(tmp, src, format="GTiff", projWin=[x0, y1, x1, y0], bandList=[1, 2, 3],
                       creationOptions=["COMPRESS=DEFLATE", "TILED=YES",
                                        "BIGTIFF=IF_SAFER", "NUM_THREADS=ALL_CPUS"])
        os.replace(tmp, out)
        tok = token()                  # tokens expire; refresh per tile
    print("done", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
