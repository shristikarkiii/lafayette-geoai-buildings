"""
Scores the AI footprints against FEMA/ORNL USA Structures and counts buildings
in FEMA flood zones.

  propy scripts/evaluate.py [detections_fc] [reference_fc] [area_fc]

Defaults are the full-city layers in data/work.gdb. Outputs (outputs/):
  pr_curve.csv            precision / recall / F1 for thresholds 0.30-0.95
  summary.json            headline numbers at the best-F1 threshold
  recall_by_size.csv      recall by reference footprint area
  recall_by_occupancy.csv recall by USA Structures occupancy class
  flood_exposure.csv      buildings in each FEMA zone, AI vs reference
"""
import csv
import json
import os
import sys

import arcpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from geoai_metrics import average_precision, greedy_match, pr_curve  # noqa: E402

GDB = os.path.join(ROOT, "data", "work.gdb")
OUT = os.path.join(ROOT, "outputs")
MIN_M2 = 41.8          # USA Structures only maps buildings >= 450 sq ft
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(14)]
SIZE_BINS = [(41.8, 100), (100, 200), (200, 500), (500, 2000), (2000, 1e12)]

det = sys.argv[1] if len(sys.argv) > 1 else os.path.join(GDB, "ai_buildings")
ref = sys.argv[2] if len(sys.argv) > 2 else os.path.join(GDB, "usa_structures")
area = sys.argv[3] if len(sys.argv) > 3 else os.path.join(GDB, "city")
tag = sys.argv[4] if len(sys.argv) > 4 else ""
arcpy.env.overwriteOutput = True
work = arcpy.env.scratchGDB
os.makedirs(OUT, exist_ok=True)


def out(name):
    return os.path.join(OUT, tag + name)


# 1. Comparable sets: same area rule, same minimum size ------------------------
def subset(fc, name):
    lyr = arcpy.management.MakeFeatureLayer(fc, name + "_lyr",
                                            f"Shape_Area >= {MIN_M2}")
    arcpy.management.SelectLayerByLocation(lyr, "HAVE_THEIR_CENTER_IN", area)
    dst = os.path.join(GDB, tag + name)          # kept for make_figures.py
    arcpy.management.CopyFeatures(lyr, dst)
    arcpy.management.Delete(lyr)
    oid = arcpy.Describe(dst).OIDFieldName
    arcpy.management.CalculateField(dst, "uid", f"!{oid}!", "PYTHON3",
                                    field_type="LONG")
    return dst


D = subset(det, "eval_det")
R = subset(ref, "eval_ref")
conf = {u: c for u, c in arcpy.da.SearchCursor(D, ["uid", "Confidence"])}
# Detect Objects Using Deep Learning writes confidence as a percentage
# (30-100); work in 0-1 and convert back only for SQL on the field.
SCALE = 100.0 if max(conf.values()) > 1 else 1.0
conf = {u: c / SCALE for u, c in conf.items()}
d_xy = {u: xy for u, xy in arcpy.da.SearchCursor(D, ["uid", "SHAPE@TRUECENTROID"])}
r_xy = {u: xy for u, xy in arcpy.da.SearchCursor(R, ["uid", "SHAPE@TRUECENTROID"])}
d_area = {u: a for u, a in arcpy.da.SearchCursor(D, ["uid", "SHAPE@AREA"])}
ref_rows = {u: (a, occ) for u, a, occ in
            arcpy.da.SearchCursor(R, ["uid", "SHAPE@AREA", "OCC_CLS"])}
print(f"detections >= {MIN_M2} m2: {len(conf)} | reference: {len(ref_rows)}")

# 2. IoU of every overlapping pair --------------------------------------------
def iou_pairs(ref_fc):
    inter = os.path.join(work, "eval_inter")
    arcpy.analysis.PairwiseIntersect([D, ref_fc], inter, "ONLY_FID")
    names = [f.name for f in arcpy.ListFields(inter)]
    fid_d = [n for n in names if n.startswith("FID_") and "eval_det" in n][0]
    fid_r = [n for n in names if n.startswith("FID_") and "eval_ref" in n][0]
    # FIDs in the intersect output are the input OIDs, which equal "uid".
    acc = {}
    for d, r, a in arcpy.da.SearchCursor(inter, [fid_d, fid_r, "SHAPE@AREA"]):
        acc[(d, r)] = acc.get((d, r), 0.0) + a
    return {(d, r): a / (d_area[d] + ref_rows[r][0] - a) for (d, r), a in acc.items()}


pairs = iou_pairs(R)

matches = greedy_match(pairs, conf, 0.5)
curve = pr_curve(conf, matches, len(ref_rows), THRESHOLDS)
ap = average_precision(conf, matches, len(ref_rows))
best = max(curve, key=lambda row: row["f1"])
with open(out("pr_curve.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(curve[0]))
    w.writeheader()
    w.writerows({k: round(v, 4) if isinstance(v, float) else v
                 for k, v in row.items()} for row in curve)

# Looser view: a reference building is "found" if any kept detection covers it
# by IoU >= 0.3 (outline quality aside, did the model see the building?).
loose = greedy_match(pairs, {d: c for d, c in conf.items()
                             if c >= best["threshold"]}, 0.3)
t = best["threshold"]
kept = {d for d, c in conf.items() if c >= t}
matched_refs = {matches[d][0] for d in kept if d in matches}
ious = [matches[d][1] for d in kept if d in matches]
summary = {
    "reference": "FEMA/ORNL USA Structures", "min_area_m2": MIN_M2,
    "n_reference": len(ref_rows), "n_detections_all": len(conf),
    "confidence_field_scale": SCALE,
    "average_precision_iou50": round(ap, 4), "best_threshold": t,
    "at_best": {k: round(v, 4) if isinstance(v, float) else v
                for k, v in best.items()},
    "recall_iou30_at_best": round(len(loose) / len(ref_rows), 4),
    "median_iou_of_matches": round(sorted(ious)[len(ious) // 2], 3) if ious else None,
}

# Systematic shift between the datasets: reference minus AI centroid for
# matched buildings (IoU >= 0.3, so shifted pairs are still counted).
def median(v):
    v = sorted(v)
    return v[len(v) // 2]


dx = [r_xy[r][0] - d_xy[d][0] for d, (r, _) in loose.items()]
dy = [r_xy[r][1] - d_xy[d][1] for d, (r, _) in loose.items()]
if dx:
    mdx, mdy = median(dx), median(dy)
    summary["reference_offset_m"] = {
        "median_dx_east": round(mdx, 2), "median_dy_north": round(mdy, 2),
        "median_distance": round(median([(a * a + b * b) ** 0.5
                                         for a, b in zip(dx, dy)]), 2)}

# Re-score after moving the reference by the median offset, which separates
# registration error between the two datasets from errors of the model.
if dx and (mdx * mdx + mdy * mdy) ** 0.5 > 1.0:
    shifted = os.path.join(work, "eval_ref_shifted")
    arcpy.management.CopyFeatures(R, shifted)
    with arcpy.da.UpdateCursor(shifted, ["SHAPE@"]) as cur:
        for (g,) in cur:
            parts = arcpy.Array()
            for part in g:
                ring = arcpy.Array()
                for p in part:
                    ring.add(arcpy.Point(p.X - mdx, p.Y - mdy) if p else None)
                parts.add(ring)
            cur.updateRow([arcpy.Polygon(parts, g.spatialReference)])
    s_pairs = iou_pairs(shifted)
    s_matches = greedy_match(s_pairs, conf, 0.5)
    s_curve = pr_curve(conf, s_matches, len(ref_rows), THRESHOLDS)
    s_best = max(s_curve, key=lambda row: row["f1"])
    summary["after_offset_correction"] = {
        "average_precision_iou50": round(average_precision(conf, s_matches, len(ref_rows)), 4),
        "best_threshold": s_best["threshold"],
        "precision": round(s_best["precision"], 4),
        "recall": round(s_best["recall"], 4), "f1": round(s_best["f1"], 4),
        "median_iou_of_matches": round(median([v[1] for v in s_matches.values()]), 3)}
    with open(out("pr_curve_offset_corrected.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(s_curve[0]))
        w.writeheader()
        w.writerows({k: round(v, 4) if isinstance(v, float) else v
                     for k, v in row.items()} for row in s_curve)

# 3. Where does it miss? Recall by size and by occupancy ------------------------
def recall_table(key_fn, labels, path):
    rows = []
    for lab in labels:
        ids = [u for u, v in ref_rows.items() if key_fn(v) == lab]
        hit = sum(1 for u in ids if u in matched_refs)
        rows.append({"group": lab, "n_reference": len(ids), "found": hit,
                     "recall": round(hit / len(ids), 4) if ids else None})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def size_label(v):
    a = v[0]
    for lo, hi in SIZE_BINS:
        if lo <= a < hi:
            return f"{lo:.0f}-{hi:.0f} m2" if hi < 1e11 else f">{lo:.0f} m2"


summary["recall_by_size"] = recall_table(
    size_label, [size_label((lo + 1e-6, None)) for lo, _ in SIZE_BINS],
    out("recall_by_size.csv"))
occs = sorted({v[1] or "Unknown" for v in ref_rows.values()})
summary["recall_by_occupancy"] = recall_table(
    lambda v: v[1] or "Unknown", occs, out("recall_by_occupancy.csv"))

# 4. Flood exposure ------------------------------------------------------------
fz = os.path.join(GDB, "flood_zones")
if arcpy.Exists(fz) and not tag:
    kept_fc = os.path.join(GDB, "eval_det_kept")
    arcpy.analysis.Select(D, kept_fc, f"Confidence >= {t * SCALE}")
    rows = []
    for label, fc in (("AI (NAIP 2023)", kept_fc), ("USA Structures", R)):
        pts = os.path.join(work, "eval_pts")
        arcpy.management.FeatureToPoint(fc, pts, "INSIDE")
        sj = os.path.join(work, "eval_sj")
        arcpy.analysis.SpatialJoin(pts, fz, sj, "JOIN_ONE_TO_ONE", "KEEP_ALL",
                                   match_option="INTERSECT")
        counts = {}
        for zone, sub, sfha in arcpy.da.SearchCursor(sj, ["FLD_ZONE", "ZONE_SUBTY", "SFHA_TF"]):
            if sfha == "T":
                k = f"{zone} (1% annual chance)"
            elif sub and "0.2 PCT" in sub.upper():
                k = "X shaded (0.2% annual chance)"
            else:
                k = "Minimal hazard / unmapped"
            counts[k] = counts.get(k, 0) + 1
        total = sum(counts.values())
        for k, n in sorted(counts.items()):
            rows.append({"dataset": label, "zone": k, "buildings": n,
                         "share": round(n / total, 4)})
        sfha_n = sum(n for k, n in counts.items() if "1% annual" in k)
        summary.setdefault("flood", {})[label] = {
            "buildings": total, "in_1pct_floodplain": sfha_n,
            "share_1pct": round(sfha_n / total, 4)}
    with open(out("flood_exposure.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

# Split the errors: an unmatched detection that still overlaps a reference
# building (IoU >= 0.1) is an outline disagreement; one that overlaps nothing
# is a clear false alarm. Same for unmatched reference buildings.
touch_d = {d for (d, r), v in pairs.items() if v >= 0.1}
touch_r = {r for (d, r), v in pairs.items() if v >= 0.1 and d in kept}
fp_all = [d for d in kept if d not in matches]
fn_all = [r for r in ref_rows if r not in matched_refs]
fp_ids = [d for d in fp_all if d not in touch_d]
fn_ids = [r for r in fn_all if r not in touch_r]
summary["error_breakdown_at_best"] = {
    "false_alarms": len(fp_all), "clear_false_alarms_no_overlap": len(fp_ids),
    "misses": len(fn_all), "clear_misses_no_overlap": len(fn_ids)}
gallery = {"tp": [list(d_xy[d]) for d in kept if d in matches][:5000],
           "fp": [list(d_xy[d]) for d in fp_ids][:5000],
           "fn": [list(r_xy[r]) for r in fn_ids][:5000]}
with open(out("gallery_ids.json"), "w") as fh:
    json.dump(gallery, fh)

with open(out("summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)
print(json.dumps(summary, indent=2))
sys.stdout.flush()
os._exit(0)
