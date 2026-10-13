"""
Runs Esri's pretrained "Building Footprint Extraction - USA" Mask R-CNN on every
NAIP window with Image Analyst's Detect Objects Using Deep Learning, on the GPU.

* Threshold is deliberately low (0.3) and the confidence of each detection is
  kept, so evaluate.py can sweep the cut-off and draw a precision-recall curve.
* One feature class per tile; finished tiles are skipped, so an interrupted run
  resumes.
* NAIP tiles overlap by a few hundred metres, so the merged result goes through
  Non Maximum Suppression to drop the duplicate of every seam building.
"""
import ctypes
import glob
import os
import sys
import time

import arcpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GDB = os.path.join(ROOT, "data", "work.gdb")
NAIP = os.path.join(ROOT, "data", "raw", "naip")
MODEL = os.path.join(ROOT, "data", "raw", "usa_building_footprints.dlpk")
ARGS = "padding 64;batch_size 4;threshold 0.3;return_bboxes False;tile_size 512"

# Keep Windows awake for the length of the run (released when the process exits).
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("ImageAnalyst")

tiles = sorted(glob.glob(os.path.join(NAIP, "*_city.tif")))
done_fcs = []
t_all = time.time()
for i, tif in enumerate(tiles, 1):
    name = "det_" + os.path.basename(tif).split("_15_")[0].replace("la_m_", "")
    fc = os.path.join(GDB, name)
    done_fcs.append(fc)
    if arcpy.Exists(fc):
        print(f"[{i}/{len(tiles)}] {name}: exists, skipped", flush=True)
        continue
    r = arcpy.Raster(tif)
    km2 = r.width * r.height * r.meanCellWidth ** 2 / 1e6
    t = time.time()
    # processorType has to be set through EnvManager: a plain arcpy.env
    # assignment was silently ignored and the tool fell back to the CPU.
    with arcpy.EnvManager(processorType="GPU", gpuId="0"):
        arcpy.ia.DetectObjectsUsingDeepLearning(
            tif, fc + "_tmp", MODEL, ARGS, "NO_NMS", "Confidence", "Class", 0,
            "PROCESS_AS_MOSAICKED_IMAGE")
    arcpy.management.Rename(fc + "_tmp", fc)
    n = int(arcpy.management.GetCount(fc)[0])
    print(f"[{i}/{len(tiles)}] {name}: {km2:.1f} km2, {n} detections, "
          f"{(time.time() - t) / 60:.1f} min", flush=True)

print("Merging tiles and removing seam duplicates...", flush=True)
merged = os.path.join(arcpy.env.scratchGDB, "det_merged")
arcpy.management.Merge(done_fcs, merged)
nms = os.path.join(arcpy.env.scratchGDB, "det_nms")
arcpy.ia.NonMaximumSuppression(merged, "Confidence", nms, None, 0.5)  # IoU

# Same membership rule as the reference: centre inside the city boundary.
lyr = arcpy.management.MakeFeatureLayer(nms, "det_lyr")
arcpy.management.SelectLayerByLocation(lyr, "HAVE_THEIR_CENTER_IN",
                                       os.path.join(GDB, "city"))
out = os.path.join(GDB, "ai_buildings")
arcpy.management.CopyFeatures(lyr, out)
print(f"ai_buildings: {arcpy.management.GetCount(merged)[0]} raw -> "
      f"{arcpy.management.GetCount(nms)[0]} after NMS -> "
      f"{arcpy.management.GetCount(out)[0]} in city | "
      f"{(time.time() - t_all) / 60:.0f} min total", flush=True)
sys.stdout.flush()
os._exit(0)
