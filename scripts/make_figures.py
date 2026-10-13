"""
README figures, from the evaluate.py outputs and the feature classes:

  outputs/figures/pr_curve.png        precision-recall, raw and offset-corrected
  outputs/figures/recall_by_size.png  which buildings the model misses
  outputs/figures/gallery.png         hits, false alarms and misses on the imagery
  outputs/figures/flood_map.png       detected buildings in FEMA flood zones
"""
import csv
import glob
import json
import os
import random
import sys

import arcpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GDB = os.path.join(ROOT, "data", "work.gdb")
OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(OUT, "figures")
NAIP = glob.glob(os.path.join(ROOT, "data", "raw", "naip", "*_city.tif"))
os.makedirs(FIG, exist_ok=True)
arcpy.env.overwriteOutput = True

INK, MUTED, GRID = "#1f2328", "#59636e", "#d0d7de"
AI_C, REF_C, BAD_C = "#00b4d8", "#ffd60a", "#ef476f"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
S = json.load(open(os.path.join(OUT, "summary.json")))
T = S["best_threshold"]
T_SQL = T * S.get("confidence_field_scale", 1.0)    # field is a percentage


def read_csv(name):
    with open(os.path.join(OUT, name), newline="") as fh:
        return list(csv.DictReader(fh))


# 1. Precision-recall -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.8), dpi=160)
for name, label, color in [
        ("pr_curve.csv", f"As published (AP {S['average_precision_iou50']:.2f})", MUTED),
        ("pr_curve_offset_corrected.csv",
         f"Reference shifted {S['reference_offset_m']['median_distance']:.1f} m "
         f"to match imagery (AP {S['after_offset_correction']['average_precision_iou50']:.2f})",
         AI_C)]:
    if not os.path.exists(os.path.join(OUT, name)):
        continue
    rows = read_csv(name)
    r = [float(x["recall"]) for x in rows]
    p = [float(x["precision"]) for x in rows]
    ax.plot(r, p, "-o", ms=3, color=color, lw=1.8, label=label)
    for x in rows:
        if float(x["threshold"]) in (0.3, 0.5, 0.7, 0.9):
            ax.annotate(x["threshold"], (float(x["recall"]), float(x["precision"])),
                        textcoords="offset points", xytext=(4, 4), fontsize=7,
                        color=color)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("Recall (share of USA Structures buildings found)")
ax.set_ylabel("Precision (share of detections that are real)")
ax.set_title("Lafayette, LA: AI footprints vs FEMA/ORNL USA Structures (IoU ≥ 0.5)",
             loc="left", fontsize=9.5, color=INK)
ax.grid(color=GRID, lw=0.5)
ax.legend(fontsize=7.5, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "pr_curve.png")); plt.close(fig)

# 2. Recall by size --------------------------------------------------------------
rows = read_csv("recall_by_size.csv")
fig, ax = plt.subplots(figsize=(6, 3.6), dpi=160)
vals = [float(r["recall"]) * 100 for r in rows]
ax.bar([r["group"] for r in rows], vals, color=AI_C, width=0.6)
for i, (v, r) in enumerate(zip(vals, rows)):
    ax.text(i, v + 1.5, f"{v:.0f}%\nn={int(r['n_reference']):,}", ha="center",
            fontsize=7.5, color=INK)
ax.set_ylim(0, 105)
ax.set_ylabel(f"Recall at threshold {T}")
ax.set_title("Small buildings (sheds, garages) are where the model misses most",
             loc="left", fontsize=9.5, color=INK)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "recall_by_size.png")); plt.close(fig)

# 3. Gallery -----------------------------------------------------------------------
mosaic = os.path.join(arcpy.env.scratchFolder, "naip_mosaic.vrt")
if not os.path.exists(mosaic):
    from osgeo import gdal
    gdal.BuildVRT(mosaic, NAIP)
img = arcpy.Raster(mosaic)
work = arcpy.env.scratchGDB
det, ref = os.path.join(GDB, "eval_det"), os.path.join(GDB, "eval_ref")


def outlines(fc, ext, where=None):
    shapes = []
    for (g,) in arcpy.da.SearchCursor(fc, ["SHAPE@"], where, spatial_filter=ext):
        for part in g:
            shapes.append(np.array([(p.X, p.Y) for p in part if p]))
    return shapes


def chip(ax, x, y, title, half=40):
    ext = arcpy.Extent(x - half, y - half, x + half, y + half)
    n = int(2 * half / 0.3)
    a = arcpy.RasterToNumPyArray(img, arcpy.Point(x - half, y - half), n, n)
    ax.imshow(np.moveaxis(a[:3], 0, -1), extent=(ext.XMin, ext.XMax, ext.YMin, ext.YMax))
    for s in outlines(ref, ext):
        ax.plot(s[:, 0], s[:, 1], color=REF_C, lw=1.3)
    for s in outlines(det, ext, f"Confidence >= {T_SQL}"):
        ax.plot(s[:, 0], s[:, 1], color=AI_C, lw=1.3)
    ax.set_xlim(ext.XMin, ext.XMax); ax.set_ylim(ext.YMin, ext.YMax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8, color=INK)


random.seed(599)
groups = json.load(open(os.path.join(OUT, "gallery_ids.json")))
fig, axes = plt.subplots(3, 4, figsize=(11, 8.6), dpi=140)
for row, (key, label) in zip(axes, [("tp", "Match (IoU ≥ 0.5)"),
                                     ("fp", "Clear false alarm (no reference)"),
                                     ("fn", "Clear miss (no AI outline)")]):
    picks = random.sample(groups[key], min(4, len(groups[key])))
    for ax, (x, y) in zip(row, picks):
        chip(ax, x, y, label)
fig.suptitle("Cyan = AI detection (2023 NAIP)   Yellow = USA Structures   "
             "80 m chips centred on a random example", fontsize=10, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "gallery.png")); plt.close(fig)

# 4. Flood map ----------------------------------------------------------------------
city = os.path.join(GDB, "city")
fz = os.path.join(GDB, "flood_zones")
fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
for (g, sfha, sub) in arcpy.da.SearchCursor(fz, ["SHAPE@", "SFHA_TF", "ZONE_SUBTY"]):
    col = "#4ea8de" if sfha == "T" else ("#bde0fe" if sub and "0.2 PCT" in sub.upper() else None)
    if col:
        for part in g:
            xy = np.array([(p.X, p.Y) if p else (np.nan, np.nan) for p in part])
            ax.fill(xy[:, 0], xy[:, 1], color=col, lw=0)
pts = os.path.join(work, "fig_pts")
arcpy.management.FeatureToPoint(os.path.join(GDB, "eval_det_kept"), pts, "INSIDE")
sj = os.path.join(work, "fig_sj")
arcpy.analysis.SpatialJoin(pts, fz, sj, "JOIN_ONE_TO_ONE", "KEEP_ALL")
xy = np.array([(p[0], p[1], s == "T") for (p, s) in
               arcpy.da.SearchCursor(sj, ["SHAPE@XY", "SFHA_TF"])], dtype=object)
dry, wet = xy[~xy[:, 2].astype(bool)], xy[xy[:, 2].astype(bool)]
ax.scatter(dry[:, 0], dry[:, 1], s=0.15, color="#adb5bd", linewidths=0)
ax.scatter(wet[:, 0], wet[:, 1], s=0.6, color=BAD_C, linewidths=0)
for (g,) in arcpy.da.SearchCursor(city, ["SHAPE@"]):
    for part in g:
        p = np.array([(q.X, q.Y) if q else (np.nan, np.nan) for q in part])
        ax.plot(p[:, 0], p[:, 1], color=INK, lw=0.8)
f = S["flood"]["AI (NAIP 2023)"]
ax.set_title(f"{f['in_1pct_floodplain']:,} of {f['buildings']:,} AI-detected buildings "
             f"({f['share_1pct']:.1%}) sit in FEMA's 1% annual-chance floodplain",
             loc="left", fontsize=10, color=INK)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ax.legend(handles=[Patch(color="#4ea8de", label="1% annual chance (A/AE)"),
                   Patch(color="#bde0fe", label="0.2% annual chance"),
                   Line2D([], [], marker="o", ls="", color=BAD_C, label="Building in 1% zone"),
                   Line2D([], [], marker="o", ls="", color="#adb5bd", label="Other building")],
          loc="lower left", fontsize=8, frameon=True)
fig.text(0.01, 0.01, "Buildings: Esri pretrained Mask R-CNN on NAIP 2023. Flood zones: "
         "FEMA NFHL. City of Lafayette, LA. NAD83 / UTM 15N.", fontsize=7, color=MUTED)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "flood_map.png")); plt.close(fig)
print("figures written")
sys.stdout.flush()
os._exit(0)
