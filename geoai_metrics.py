"""Matching and scoring for building detections. Pure Python, so it can be
unit-tested without ArcGIS."""


def greedy_match(pairs, confidence, min_iou=0.5):
    """COCO-style one-to-one matching.

    pairs       {(det_id, ref_id): iou} for every overlapping pair
    confidence  {det_id: score}
    Detections are visited from most to least confident; each takes the
    unmatched reference with the highest IoU >= min_iou.
    Returns {det_id: (ref_id, iou)} for true positives.
    """
    by_det = {}
    for (d, r), iou in pairs.items():
        if iou >= min_iou:
            by_det.setdefault(d, []).append((iou, r))
    taken, matches = set(), {}
    for d in sorted(confidence, key=lambda k: (-confidence[k], k)):
        for iou, r in sorted(by_det.get(d, []), reverse=True):
            if r not in taken:
                taken.add(r)
                matches[d] = (r, iou)
                break
    return matches


def pr_curve(confidence, matches, n_ref, thresholds):
    """Precision, recall and F1 when only detections with score >= t are kept.
    Because matching is done most-confident-first, raising the threshold never
    changes which reference a surviving detection is matched to."""
    rows = []
    for t in thresholds:
        kept = [d for d, c in confidence.items() if c >= t]
        tp = sum(1 for d in kept if d in matches)
        p = tp / len(kept) if kept else 1.0
        r = tp / n_ref if n_ref else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({"threshold": t, "detections": len(kept), "tp": tp,
                     "fp": len(kept) - tp, "fn": n_ref - tp,
                     "precision": p, "recall": r, "f1": f1})
    return rows


def average_precision(confidence, matches, n_ref):
    """Area under the precision-recall curve using every detection as a
    threshold, with the usual monotone (interpolated) precision envelope."""
    order = sorted(confidence, key=lambda k: -confidence[k])
    tp = 0
    pts = []
    for i, d in enumerate(order, 1):
        tp += d in matches
        pts.append((tp / n_ref, tp / i))
    ap, best = 0.0, 0.0
    prev_r = pts[-1][0] if pts else 0.0
    for r, p in reversed(pts):
        best = max(best, p)
        ap += (prev_r - r) * best if prev_r > r else 0.0
        prev_r = r
    ap += prev_r * best                # from recall 0 to the first point
    return ap
