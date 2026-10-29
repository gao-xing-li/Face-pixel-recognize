"""Strict binary pixel evaluation shared by B's validation and C's testing.

Positive means the agreed skin label. Undefined metrics are represented by None.
Image means exclude undefined values and report a separate denominator per metric.
"""

from collections.abc import Iterable, Mapping
from numbers import Integral

import numpy as np

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "iou")
COUNT_NAMES = ("tp", "fp", "tn", "fn", "n_valid")


def _binary_mask(value, name: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError(f"{name} must be a nonempty 2-D mask")
    if mask.dtype.kind not in "bui" or not np.isin(mask, (0, 1)).all():
        raise ValueError(f"{name} must contain only boolean/integer 0 or 1")
    return mask.astype(bool, copy=False)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    n_valid = tp + fp + tn + fn
    if n_valid == 0:
        raise ValueError("at least one valid evaluation pixel is required")
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_valid": n_valid,
        "accuracy": _ratio(tp + tn, n_valid),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "iou": _ratio(tp, tp + fp + fn),
    }


def evaluate(pred_mask, gt_mask, valid_mask=None) -> dict:
    """Return counts and metrics for one pair of same-size 0/1 masks.

    The argument order is prediction first, ground truth second. valid_mask=1
    selects pixels to score; None scores every pixel. Unknown labels, automatic
    thresholding/resizing and an entirely ignored image are rejected.
    """
    pred = _binary_mask(pred_mask, "pred_mask")
    gt = _binary_mask(gt_mask, "gt_mask")
    if pred.shape != gt.shape:
        raise ValueError("pred_mask and gt_mask must have the same shape")
    valid = np.ones(gt.shape, dtype=bool)
    if valid_mask is not None:
        valid = _binary_mask(valid_mask, "valid_mask")
        if valid.shape != gt.shape:
            raise ValueError("valid_mask must have the same shape as gt_mask")
    tp = int(np.count_nonzero(pred & gt & valid))
    fp = int(np.count_nonzero(pred & ~gt & valid))
    tn = int(np.count_nonzero(~pred & ~gt & valid))
    fn = int(np.count_nonzero(~pred & gt & valid))
    return _from_counts(tp, fp, tn, fn)


def summarize(per_image_metrics: Iterable[Mapping]) -> dict:
    """Pool pixel counts and separately compute an unweighted mean per image.

    Metrics are recalculated from counts, never averaged to obtain global scores.
    """
    rows = list(per_image_metrics)
    if not rows:
        raise ValueError("cannot summarize an empty image set")
    checked = []
    for row in rows:
        for name in COUNT_NAMES:
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        current = _from_counts(*(int(row[name]) for name in ("tp", "fp", "tn", "fn")))
        if current["n_valid"] != row["n_valid"]:
            raise ValueError("n_valid must equal tp + fp + tn + fn")
        checked.append(current)
    pooled = _from_counts(
        *(sum(row[name] for row in checked) for name in ("tp", "fp", "tn", "fn"))
    )
    image_mean, n_defined = {}, {}
    for name in METRIC_NAMES:
        values = [row[name] for row in checked if row[name] is not None]
        image_mean[name] = sum(values) / len(values) if values else None
        n_defined[name] = len(values)
    return {
        "n_images": len(checked),
        "global": pooled,
        "image_mean": image_mean,
        "image_mean_n_defined": n_defined,
    }
