"""Candidate generation and ellipse fitting using image information only."""

from itertools import combinations
import math

import cv2
import numpy as np


def box_gap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(bx - (ax + aw - 1), ax - (bx + bw - 1), 0)
    dy = max(by - (ay + ah - 1), ay - (by + bh - 1), 0)
    return math.hypot(dx, dy)


def extract_components(raw, image_bgr, config):
    count, labels, stats, centers = cv2.connectedComponentsWithStats(raw, connectivity=8)
    chroma = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 1:3]
    components = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        item = {"id": label, "area": area, "box": stats[label, :4].tolist(),
                "center": centers[label].tolist(), "retained": area >= config["min_component_area"]}
        if item["retained"]:
            mask = (labels == label).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            points = []
            for contour in contours:
                contour = contour.reshape(-1, 2)
                size = min(len(contour), config["max_points_per_contour"])
                indices = np.linspace(0, len(contour) - 1, size, dtype=int)
                points.extend(contour[indices].tolist())
            item["points"] = points
            item["mean_cr_cb"] = chroma[mask.astype(bool)].mean(axis=0).tolist()
        components.append(item)
    return labels, components


def candidate_groups(components, config):
    kept = sorted((c for c in components if c["retained"]), key=lambda c: (-c["area"], c["id"]))
    singles = [(c["id"],) for c in kept]
    groups = set()
    for anchor in kept:
        neighbors = []
        for other in kept:
            if anchor["id"] == other["id"]:
                continue
            gap = box_gap(anchor["box"], other["box"])
            color_distance = float(np.linalg.norm(np.subtract(anchor["mean_cr_cb"], other["mean_cr_cb"])))
            if gap <= config["max_box_gap"] and color_distance <= config["max_chroma_distance"]:
                neighbors.append((gap, color_distance, other["id"]))
        neighbors.sort()
        nearest = [n[2] for n in neighbors[:config["neighbors_per_component"]]]
        for size in (1, 2):
            for subset in combinations(nearest, size):
                groups.add(tuple(sorted((anchor["id"], *subset))))
    return singles + sorted(groups, key=lambda ids: (len(ids), ids))


def ellipse_region(shape, geometry, scale=1.0):
    yy, xx = np.indices(shape)
    angle = math.radians(geometry["angle_degrees"])
    dx = xx - geometry["center"][0]
    dy = yy - geometry["center"][1]
    u = dx * math.cos(angle) + dy * math.sin(angle)
    v = -dx * math.sin(angle) + dy * math.cos(angle)
    a, b = np.asarray(geometry["axes"]) * scale / 2
    return ((u / a) ** 2 + (v / b) ** 2 <= 1).astype(np.uint8)


def fit_candidate(points, mask, raw, inner_scale):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) < 5:
        return {"status": "too_few_points"}, None
    if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        return {"status": "collinear_points"}, None
    try:
        center, axes, angle = cv2.fitEllipse(points.reshape(-1, 1, 2))
    except cv2.error:
        return {"status": "fit_failed"}, None
    if not np.isfinite([*center, *axes, angle]).all() or min(axes) <= 0:
        return {"status": "invalid_geometry"}, None
    major, minor = axes
    if major < minor:
        major, minor = minor, major
        angle += 90
    geometry = {"center": list(center), "axes": [major, minor], "angle_degrees": angle % 180}
    region = ellipse_region(mask.shape, geometry)
    if not region.any():
        return {"status": "outside_image", "geometry": geometry}, None
    inner = ellipse_region(mask.shape, geometry, inner_scale).astype(bool)
    region_bool = region.astype(bool)
    outer = region_bool & ~inner
    support = lambda where: float(raw[where].mean()) if where.any() else None
    h, w = mask.shape
    result = {
        "status": "ok", "geometry": geometry, "axis_ratio": major / minor,
        "ellipse_pixels": int(region.sum()), "candidate_pixels": int(mask.sum()),
        "raw_support": support(region_bool), "inner_support": support(inner),
        "outer_support": support(outer),
        "candidate_coverage": float(region[mask.astype(bool)].mean()),
        "center_outside_image": not (0 <= center[0] < w and 0 <= center[1] < h),
        "touches_image_edge": bool(region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any()),
    }
    return result, region
