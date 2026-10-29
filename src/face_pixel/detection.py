"""B detector: Cb/Cr thresholds and image-only binary postprocessing."""

import math

import cv2
import numpy as np

METHODS = ("none", "fill", "largest_fill", "close_largest_fill")


def validate_model(params):
    expected = {"schema_version": 1, "fit_split": "train", "skin_label_ids": [1, 2],
                "color_space": "OpenCV YCrCb", "input_color_order": "BGR", "input_dtype": "uint8",
                "conversion": "cv2.COLOR_BGR2YCrCb", "conversion_channel_order": ["Y", "Cr", "Cb"],
                "model_channel_indices": {"Cr": 1, "Cb": 2}, "pixel_value_range": [0, 255],
                "preprocessing": "none; no resizing or color enhancement"}
    for key, value in expected.items():
        if params.get(key) != value:
            raise ValueError(f"unsupported model metadata: {key}")
    for channel in ("cb", "cr"):
        for kind in ("mean", "std"):
            value = params.get(f"{kind}_{channel}")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 255:
                raise ValueError(f"invalid {kind}_{channel}")


def fill_holes(mask):
    """Fill background not 4-connected to the image exterior, with padding."""
    padded = np.pad(mask.astype(np.uint8), 1)
    cv2.floodFill(padded, None, (0, 0), 2, flags=4)
    return (padded[1:-1, 1:-1] != 2).astype(np.uint8)


def postprocess(mask, method, kernel=5):
    if method not in METHODS:
        raise ValueError(f"unknown postprocess: {method}")
    if type(kernel) is not int or kernel < 1 or kernel % 2 != 1:
        raise ValueError("closing kernel must be an odd positive integer")
    mask = mask.copy()
    if method == "close_largest_fill":
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element, iterations=1,
                              borderType=cv2.BORDER_CONSTANT, borderValue=0)
    if method in ("largest_fill", "close_largest_fill"):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count > 1:
            # Ties follow OpenCV's component labels; version is recorded.
            component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = (labels == component).astype(np.uint8)
    return fill_holes(mask) if method != "none" else mask


def detect(image_bgr, params, config):
    """Image/parameters/config only: no labels or ignore-mask argument."""
    if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or not image_bgr.size:
        raise ValueError("expected nonempty uint8 BGR image")
    threshold = config["threshold"]
    if threshold.get("rule") != "cbcr_box" or threshold.get("channels") != ["Cb", "Cr"]:
        raise ValueError("unsupported threshold rule/channels")
    if config.get("preprocess") != "none" or config.get("evaluation_size") != "original; no resizing":
        raise ValueError("unsupported preprocessing or resizing")
    k = threshold["k"]
    if isinstance(k, bool) or not isinstance(k, (float, int)) or not math.isfinite(k) or k <= 0:
        raise ValueError("k must be finite and positive")
    converted = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    cr, cb = converted[:, :, 1], converted[:, :, 2]
    mask = ((np.abs(cr - params["mean_cr"]) <= k * params["std_cr"]) &
            (np.abs(cb - params["mean_cb"]) <= k * params["std_cb"])).astype(np.uint8)
    return postprocess(mask, config["postprocess"]["method"], config["postprocess"]["kernel"])
