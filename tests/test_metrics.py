import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geoai_metrics import average_precision, greedy_match, pr_curve  # noqa: E402


def test_one_to_one_most_confident_wins():
    pairs = {("a", "R1"): 0.8, ("b", "R1"): 0.9}
    conf = {"a": 0.95, "b": 0.6}
    # "a" is more confident, so it claims R1 even though "b" overlaps more.
    assert greedy_match(pairs, conf) == {"a": ("R1", 0.8)}


def test_iou_below_threshold_is_not_a_match():
    assert greedy_match({("a", "R1"): 0.49}, {"a": 0.9}) == {}


def test_pr_curve_counts():
    conf = {"a": 0.9, "b": 0.7, "c": 0.4}
    matches = {"a": ("R1", 0.7), "c": ("R2", 0.6)}
    lo, hi = pr_curve(conf, matches, n_ref=4, thresholds=[0.3, 0.8])
    assert (lo["tp"], lo["fp"], lo["fn"]) == (2, 1, 2)
    assert abs(lo["precision"] - 2 / 3) < 1e-9 and lo["recall"] == 0.5
    assert (hi["tp"], hi["fp"], hi["precision"], hi["recall"]) == (1, 0, 1.0, 0.25)


def test_average_precision_perfect_and_partial():
    conf = {"a": 0.9, "b": 0.8}
    assert average_precision(conf, {"a": 1, "b": 1}, n_ref=2) == 1.0
    # one hit at rank 1, a miss at rank 2, half the references found -> 0.5
    assert average_precision(conf, {"a": 1}, n_ref=2) == 0.5
