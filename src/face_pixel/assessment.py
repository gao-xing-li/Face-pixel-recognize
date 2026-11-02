"""Supplementary boundary and image-cluster uncertainty measures.

These describe fixed saved predictions; they do not select a model or threshold.
"""

import math

import numpy as np

from .evaluation import _binary_mask


def boundary_iou(pred_mask, gt_mask, valid_mask=None, *, dilation_ratio=0.02) -> dict:
    """IoU of interior boundary bands (Cheng et al., CVPR 2021).

    Erode a zero-padded foreground with a 3x3 square for d iterations, where
    d=max(1, round(ratio*image_diagonal)). Remove the eroded interior. Ignore
    pixels and their d-neighborhood are excluded; physical image borders remain.
    Empty union returns None. Test images here contain no ignored regions.
    """
    import cv2

    if isinstance(dilation_ratio, bool) or not isinstance(dilation_ratio, (int, float)) or not math.isfinite(dilation_ratio) or not 0 < dilation_ratio <= 0.5:
        raise ValueError("dilation_ratio must be finite and in (0, 0.5]")
    pred, gt = _binary_mask(pred_mask, "pred_mask"), _binary_mask(gt_mask, "gt_mask")
    if pred.shape != gt.shape:
        raise ValueError("prediction and truth must have the same shape")
    valid = np.ones(gt.shape, dtype=bool) if valid_mask is None else _binary_mask(valid_mask, "valid_mask")
    if valid.shape != gt.shape or not valid.any():
        raise ValueError("valid mask must match the image and contain valid pixels")
    radius = max(1, int(round(dilation_ratio * math.hypot(*gt.shape))))
    kernel = np.ones((3, 3), dtype=np.uint8)
    safe = cv2.erode(valid.astype(np.uint8), kernel, iterations=radius, borderType=cv2.BORDER_CONSTANT, borderValue=1).astype(bool)

    def band(mask):
        source = (mask & valid).astype(np.uint8)
        padded = np.pad(source, 1, constant_values=0)
        interior = cv2.erode(padded, kernel, iterations=radius, borderType=cv2.BORDER_CONSTANT, borderValue=0)[1:-1, 1:-1]
        return (source != interior) & safe

    p, g = band(pred), band(gt)
    intersection, union = int(np.count_nonzero(p & g)), int(np.count_nonzero(p | g))
    return {"boundary_intersection": intersection, "boundary_union": union, "boundary_iou": intersection / union if union else None, "band_radius_pixels": radius, "eligible_pixels": int(safe.sum())}


def paired_bootstrap_f1(baseline_counts, final_counts, *, n_resamples=10000, seed=42, confidence=0.95) -> dict:
    """Percentile intervals, resampling whole images with shared paired indices.

    Counts columns are TP, FP, TN, FN. Each resample pools counts before F1;
    individual pixels are never treated as independent observations.
    """
    base, final = np.asarray(baseline_counts), np.asarray(final_counts)
    if base.ndim != 2 or base.shape[1] != 4 or base.shape[0] < 2 or base.shape != final.shape:
        raise ValueError("provide matching N-by-4 count matrices with at least two images")
    if base.dtype.kind not in "ui" or final.dtype.kind not in "ui" or (base < 0).any() or (final < 0).any():
        raise ValueError("counts must be nonnegative integers")
    if not np.array_equal(base[:, 0] + base[:, 3], final[:, 0] + final[:, 3]) or not np.array_equal(base[:, 1] + base[:, 2], final[:, 1] + final[:, 2]):
        raise ValueError("paired counts must refer to the same true positive/negative pixels")
    if type(n_resamples) is not int or n_resamples < 100 or not 0 < confidence < 1:
        raise ValueError("use at least 100 resamples and confidence in (0, 1)")
    base, final = base.astype(np.float64), final.astype(np.float64)

    def f1(counts):
        denominator = 2 * counts[..., 0] + counts[..., 1] + counts[..., 3]
        return np.divide(2 * counts[..., 0], denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0)

    estimates = (float(f1(base.sum(axis=0))), float(f1(final.sum(axis=0))))
    if not all(math.isfinite(value) for value in estimates):
        raise ValueError("global F1 must be defined for both schemes")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(base), size=(n_resamples, len(base)))
    b, f = f1(base[indices].sum(axis=1)), f1(final[indices].sum(axis=1))
    defined = np.isfinite(b) & np.isfinite(f)
    if not defined.any():
        raise ValueError("no resample has defined paired F1 values")
    low, high = (1 - confidence) / 2, 1 - (1 - confidence) / 2

    def interval(values, estimate):
        return {"estimate": estimate, "low": float(np.quantile(values[defined], low)), "high": float(np.quantile(values[defined], high))}

    return {"method": "paired image bootstrap; percentile interval", "confidence": confidence, "n_images": len(base), "n_resamples": n_resamples, "n_defined_resamples": int(defined.sum()), "seed": seed, "baseline_global_f1": interval(b, estimates[0]), "final_global_f1": interval(f, estimates[1]), "final_minus_baseline_global_f1": interval(f - b, estimates[1] - estimates[0])}


def paired_mean_interval(baseline_values, final_values, *, n_resamples=10000, seed=42, confidence=0.95) -> dict:
    """Paired percentile interval for an image-level mean difference."""
    b, f = np.asarray(baseline_values, dtype=float), np.asarray(final_values, dtype=float)
    if b.ndim != 1 or b.shape != f.shape or len(b) < 2 or not np.isfinite(b).all() or not np.isfinite(f).all():
        raise ValueError("provide matching finite image-level arrays of length >= 2")
    if type(n_resamples) is not int or n_resamples < 100 or not 0 < confidence < 1:
        raise ValueError("invalid resampling settings")
    differences = f - b
    rng = np.random.default_rng(seed)
    sampled = differences[rng.integers(0, len(b), size=(n_resamples, len(b)))].mean(axis=1)
    tail = (1 - confidence) / 2
    return {"estimate": float(differences.mean()), "low": float(np.quantile(sampled, tail)), "high": float(np.quantile(sampled, 1 - tail)), "n_images": len(b), "n_resamples": n_resamples, "confidence": confidence, "seed": seed, "method": "paired image bootstrap; percentile interval"}
