"""Estimate a face ellipse from two eye centers and training-only geometry."""

import math

import cv2
import numpy as np


def eye_centers(label):
    """Return label 4/5 centroids ordered from image-left to image-right."""
    points = []
    for label_id in (4, 5):
        yy, xx = np.where(label == label_id)
        if not len(xx):
            return None
        points.append([float(xx.mean()), float(yy.mean())])
    return sorted(points, key=lambda point: point[0])


def fit_face_ellipse(face_mask):
    """Fit an ellipse to the external contour of the largest face component."""
    face = np.asarray(face_mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(face, connectivity=8)
    if count <= 1:
        return None
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    main = (labels == component).astype(np.uint8)
    contours, _ = cv2.findContours(main, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return None
    center, axes, angle = cv2.fitEllipse(contour)
    major, minor = map(float, axes)
    if major < minor:
        major, minor = minor, major
        angle += 90.0
    values = [*center, major, minor, angle]
    if not np.isfinite(values).all() or minor <= 0:
        return None
    return {
        "center": [float(center[0]), float(center[1])],
        "axes": [major, minor],
        "angle_degrees": float(angle % 180.0),
    }


def eye_frame(centers):
    """Return midpoint, eye-line unit vector, downward normal and eye distance."""
    points = np.asarray(sorted(centers, key=lambda point: point[0]), dtype=float)
    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError("centers must contain two finite 2-D points")
    vector = points[1] - points[0]
    distance = float(np.linalg.norm(vector))
    if distance <= 1e-6:
        raise ValueError("eye centers must be distinct")
    along = vector / distance
    down = np.array([-along[1], along[0]])
    if down[1] < 0:
        down = -down
    return (points[0] + points[1]) / 2.0, along, down, distance


def geometry_sample(centers, face_geometry):
    """Express a fitted face ellipse in an eye-centered coordinate system."""
    midpoint, along, down, distance = eye_frame(centers)
    offset = np.asarray(face_geometry["center"], dtype=float) - midpoint
    normal_angle = math.degrees(math.atan2(down[1], down[0]))
    relative_angle = ((face_geometry["angle_degrees"] - normal_angle + 90.0) % 180.0) - 90.0
    return {
        "center_along_per_eye_distance": float(np.dot(offset, along) / distance),
        "center_down_per_eye_distance": float(np.dot(offset, down) / distance),
        "major_axis_per_eye_distance": float(face_geometry["axes"][0] / distance),
        "minor_axis_per_eye_distance": float(face_geometry["axes"][1] / distance),
        "major_angle_from_down_degrees": float(relative_angle),
    }


def _distribution(values):
    values = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(values)),
        "q10": float(np.quantile(values, 0.10)),
        "q90": float(np.quantile(values, 0.90)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def fit_geometry_prior(samples, minimum_samples=5):
    """Aggregate normalized training samples with robust component-wise medians."""
    samples = list(samples)
    if len(samples) < minimum_samples:
        raise ValueError(f"need at least {minimum_samples} valid training samples")
    names = (
        "center_along_per_eye_distance",
        "center_down_per_eye_distance",
        "major_axis_per_eye_distance",
        "minor_axis_per_eye_distance",
        "major_angle_from_down_degrees",
    )
    distributions = {name: _distribution([sample[name] for sample in samples]) for name in names}
    return {
        "training_images": len(samples),
        "aggregation": "component-wise median",
        "coordinate_system": "origin=eye midpoint; x=eye line; y=downward normal; lengths divided by eye distance",
        "model": {name: distributions[name]["median"] for name in names},
        "distributions": distributions,
    }


def ellipse_from_eyes(centers, geometry_prior):
    """Map two detected eyes to a face ellipse using the training geometry prior."""
    midpoint, along, down, distance = eye_frame(centers)
    model = geometry_prior.get("model", geometry_prior)
    center = midpoint + distance * (
        model["center_along_per_eye_distance"] * along
        + model["center_down_per_eye_distance"] * down
    )
    normal_angle = math.degrees(math.atan2(down[1], down[0]))
    major = distance * model["major_axis_per_eye_distance"]
    minor = distance * model["minor_axis_per_eye_distance"]
    if not np.isfinite([*center, major, minor]).all() or major <= 0 or minor <= 0:
        raise ValueError("geometry prior produced an invalid ellipse")
    return {
        "center": center.tolist(),
        "axes": [float(major), float(minor)],
        "angle_degrees": float((normal_angle + model["major_angle_from_down_degrees"]) % 180.0),
    }


def ellipse_region(shape, geometry):
    """Rasterize a geometry whose axes store full diameters."""
    height, width = shape
    canvas = np.zeros((height, width), dtype=np.uint8)
    center = tuple(np.rint(geometry["center"]).astype(int))
    axes = tuple(np.maximum(1, np.rint(np.asarray(geometry["axes"]) / 2.0)).astype(int))
    cv2.ellipse(canvas, center, axes, geometry["angle_degrees"], 0, 360, 1, -1, cv2.LINE_8)
    return canvas
