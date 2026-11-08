"""Image-only selection and morphology inside nested eye-derived ellipses."""

import cv2
import numpy as np


def choose_eye_pair(strict_pairs, relaxed_pairs):
    """Use relaxed geometry only as a fallback, never to replace a strict pair."""
    if strict_pairs:
        return strict_pairs[0], "standard_pair"
    if relaxed_pairs:
        return relaxed_pairs[0], "recovered_pair"
    return None, "no_pair"


def shifted_geometry(base, eye_centers, shift_fraction):
    """Shift an ellipse along the eye line by a fraction of eye distance."""
    eyes = np.asarray(sorted(eye_centers, key=lambda point: point[0]), dtype=float)
    vector = eyes[1] - eyes[0]
    distance = float(np.linalg.norm(vector))
    if distance <= 1e-6:
        raise ValueError("eye centers must be distinct")
    along = vector / distance
    geometry = {
        "center": (np.asarray(base["center"], dtype=float) + shift_fraction * distance * along).tolist(),
        "axes": list(map(float, base["axes"])),
        "angle_degrees": float(base["angle_degrees"]),
    }
    return geometry


def ellipse_candidates(base, eye_centers, shift_fraction):
    """Return centered, image-left and image-right pose candidates."""
    if not 0 < shift_fraction < .5:
        raise ValueError("shift_fraction must be between 0 and 0.5")
    return [
        {"name": "center", "shift_fraction": 0.0, "geometry": shifted_geometry(base, eye_centers, 0.0)},
        {"name": "image_left", "shift_fraction": -shift_fraction,
         "geometry": shifted_geometry(base, eye_centers, -shift_fraction)},
        {"name": "image_right", "shift_fraction": shift_fraction,
         "geometry": shifted_geometry(base, eye_centers, shift_fraction)},
    ]


def morphology(mask, operation, kernel_size):
    if type(kernel_size) is not int or kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("kernel size must be a positive odd integer")
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    padded = np.pad(mask.astype(np.uint8), kernel_size)
    changed = cv2.morphologyEx(
        padded, operation, element, iterations=1,
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return changed[kernel_size:-kernel_size, kernel_size:-kernel_size]


def fill_holes(mask):
    padded = np.pad(mask.astype(np.uint8), 1)
    cv2.floodFill(padded, None, (0, 0), 2, flags=4)
    return (padded[1:-1, 1:-1] != 2).astype(np.uint8)


def _seed_components(local, inner, minimum_pixels):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(local, connectivity=8)
    retained = []
    for component in range(1, count):
        overlap = int(np.count_nonzero((labels == component) & inner.astype(bool)))
        if overlap >= minimum_pixels:
            retained.append(component)
    return np.isin(labels, retained).astype(np.uint8), retained, stats


def reconstruct_low_support(relaxed, seed, inner, kernel_size, minimum_seed_pixels):
    """Reconnect relaxed skin evidence to trusted core pixels inside the inner ellipse."""
    relaxed = np.asarray(relaxed, dtype=np.uint8)
    seed = np.asarray(seed, dtype=np.uint8)
    inner = np.asarray(inner, dtype=np.uint8)
    if relaxed.shape != seed.shape or relaxed.shape != inner.shape:
        raise ValueError("relaxed, seed and inner masks must have the same shape")
    allowed = morphology(relaxed & inner, cv2.MORPH_CLOSE, kernel_size) & inner
    allowed |= seed & inner
    count, labels = cv2.connectedComponents(allowed, connectivity=8)
    retained = []
    for component in range(1, count):
        overlap = int(np.count_nonzero((labels == component) & seed.astype(bool)))
        if overlap >= minimum_seed_pixels:
            retained.append(component)
    connected = np.isin(labels, retained).astype(np.uint8)
    return fill_holes(connected) & inner


def refine(raw, inner, outer, config, relaxed=None):
    """Apply strong core and weak transition morphology, bounded by outer ellipse."""
    raw = np.asarray(raw, dtype=np.uint8)
    inner = np.asarray(inner, dtype=np.uint8)
    outer = np.asarray(outer, dtype=np.uint8)
    if raw.shape != inner.shape or raw.shape != outer.shape:
        raise ValueError("raw, inner and outer masks must have the same shape")
    if np.any(inner > outer):
        raise ValueError("inner ellipse must be contained in outer ellipse")
    local = raw & outer
    selected, retained, _ = _seed_components(local, inner, config["minimum_core_pixels"])
    stages = {"local": local, "selected_raw": selected, "seed": selected & inner}
    if not retained:
        stages["final"] = np.zeros_like(raw)
        return "no_core_component", stages, {"retained_components": 0, "score": None}

    weak = morphology(selected, cv2.MORPH_CLOSE, config["weak_kernel"]) & outer
    strong = fill_holes(morphology(selected, cv2.MORPH_CLOSE, config["strong_kernel"])) & inner
    inner_support = float(selected[inner.astype(bool)].mean()) if inner.any() else 0.0
    reconstruction = np.zeros_like(raw)
    low_support = relaxed is not None and inner_support < config.get("low_support_threshold", 0.0)
    if low_support:
        reconstruction = reconstruct_low_support(
            relaxed, stages["seed"], inner,
            config["reconstruction_close_kernel"], config["minimum_reconstruction_seed_pixels"],
        )
        strong |= reconstruction
    proposal = strong | (weak & (outer & (1 - inner)))
    count, labels = cv2.connectedComponents(proposal, connectivity=8)
    anchor_labels = np.unique(labels[stages["seed"].astype(bool)])
    anchor_labels = anchor_labels[anchor_labels > 0]
    connected = np.isin(labels, anchor_labels).astype(np.uint8)
    final = fill_holes(connected) & outer

    boundary = outer - cv2.erode(outer, np.ones((3, 3), np.uint8))
    outer_support = float(selected[outer.astype(bool)].mean()) if outer.any() else 0.0
    retained_fraction = float(selected.sum() / local.sum()) if local.any() else 0.0
    boundary_occupancy = float(final[boundary.astype(bool)].mean()) if boundary.any() else 0.0
    score = (
        config["score_weights"]["inner_support"] * inner_support
        + config["score_weights"]["outer_support"] * outer_support
        + config["score_weights"]["retained_fraction"] * retained_fraction
        - config["score_weights"]["boundary_occupancy"] * boundary_occupancy
    )
    descriptors = {
        "retained_components": len(retained),
        "inner_support": inner_support,
        "outer_support": outer_support,
        "retained_fraction": retained_fraction,
        "boundary_occupancy": boundary_occupancy,
        "low_support_triggered": bool(low_support),
        "reconstruction_pixels": int(reconstruction.sum()),
        "score": float(score),
    }
    stages.update(weak=weak, strong=strong, reconstruction=reconstruction, proposal=proposal, final=final)
    return "ok", stages, descriptors


def select_candidate(candidates, center_penalty):
    """Choose only from image-derived scores; favor center when scores are close."""
    eligible = [candidate for candidate in candidates if candidate["status"] == "ok"]
    if not eligible:
        return None
    for candidate in eligible:
        penalty = center_penalty if candidate["name"] != "center" else 0.0
        candidate["selection_score"] = candidate["descriptors"]["score"] - penalty
    return max(eligible, key=lambda candidate: (candidate["selection_score"], candidate["name"] == "center"))
