"""Handcrafted eye proposals: grayscale black-hat, blobs, then eye-pair geometry.

No per-image annotations or learned eye/face detector enter this module.
Midpoint or individual-eye priors are estimated separately from training images.
"""

from itertools import combinations
import math

import cv2
import numpy as np


def odd_size(value):
    return max(3, int(round(value)) | 1)


def rank_pairs(regions, score_key, limit):
    """Rank the same proposals under either visual-only or prior-assisted scores."""
    proposals = [p for r in regions for p in sorted(r["pairs"], key=lambda p: -p[score_key])[:limit]]
    ranked = []
    for pair in sorted(proposals, key=lambda p: -p[score_key]):
        if not any(np.max(np.linalg.norm(np.subtract(pair["centers"], p["centers"]), axis=1)) < 4 for p in ranked):
            ranked.append(pair)
    return ranked[:limit]


def locate_in_eye_windows(bgr, skin, config, prior):
    """Find actual dark blobs separately near the two training-derived eye centers."""
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    h, w = gray.shape
    yy, xx = np.indices((h, w))
    scale = float(np.linalg.norm((np.array(prior["eyes"][1]["mean"])-prior["eyes"][0]["mean"])*[w, h])*2.5)
    kw, kh = [odd_size(scale*f) for f in config["blackhat_kernel_fraction"]]
    response = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh)))
    region = {"id": "eyes", "kind": "eye_windows", "status": "ok", "box": [0, 0, w, h],
              "kernel": [kw, kh], "candidates": [], "pairs": [], "windows": [], "response": response}
    binary = np.zeros_like(gray)
    for eye in prior["eyes"]:
        d2 = ((xx/w-eye["mean"][0])/eye["sigma"][0])**2 + ((yy/h-eye["mean"][1])/eye["sigma"][1])**2
        window = d2 <= prior["window_radius_sigma"]**2
        otsu, _ = cv2.threshold(response[window].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        threshold = max(config["minimum_response"], otsu)
        local = ((response > threshold) & window).astype(np.uint8)
        local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8))
        local[~window] = 0
        binary |= local
        region["windows"].append({"side": eye["side"], "mean": eye["mean"], "sigma": eye["sigma"],
                                  "radius_sigma": prior["window_radius_sigma"], "threshold": float(threshold)})
        count, labels, stats, centers = cv2.connectedComponentsWithStats(local, connectivity=8)
        for label in range(1, count):
            x, y, bw, bh, area = map(int, stats[label])
            if not (config["eye_width_fraction"][0]*scale <= bw <= config["eye_width_fraction"][1]*scale
                    and config["eye_height_fraction"][0]*scale <= bh <= config["eye_height_fraction"][1]*scale
                    and config["eye_area_fraction"][0]*scale**2 <= area <= config["eye_area_fraction"][1]*scale**2
                    and config["eye_aspect_ratio"][0] <= bw/bh <= config["eye_aspect_ratio"][1]):
                continue
            mask = (labels == label).astype(np.uint8)
            radius = odd_size(.05*scale)
            ring = (cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))) > 0) & (mask == 0)
            offset = (centers[label]/[w, h]-eye["mean"])/eye["sigma"]
            distance2 = float(np.dot(offset, offset))
            if distance2 > prior["window_radius_sigma"]**2:
                continue
            region["candidates"].append({"id": f"eyes-{'L' if eye['side']=='image_left' else 'R'}{label}",
                "side": eye["side"], "center": centers[label].tolist(), "box": [x, y, bw, bh], "area": area,
                "response": float(response[mask > 0].mean()), "skin_ring": float(skin[ring].mean()) if ring.any() else 0.,
                "prior_distance_squared": distance2})
    region["binary"] = binary
    region["threshold"] = float(np.mean([window["threshold"] for window in region["windows"]]))
    left = [c for c in region["candidates"] if c["side"] == "image_left"]
    right = [c for c in region["candidates"] if c["side"] == "image_right"]
    for a in left:
        for b in right:
            dx, dy = np.subtract(b["center"], a["center"])
            distance, angle = math.hypot(dx, dy), math.degrees(math.atan2(dy, dx))
            ratio = min(a["area"], b["area"])/max(a["area"], b["area"])
            if not (config["pair_distance_fraction"][0]*scale <= distance <= config["pair_distance_fraction"][1]*scale
                    and abs(angle) <= config["max_pair_angle_degrees"] and ratio >= config["min_pair_area_ratio"]):
                continue
            alignment = 1-abs(angle)/config["max_pair_angle_degrees"]
            spacing = max(0, 1-abs(distance/scale-.36)/.30)
            strength = min(1, (a["response"]+b["response"])/160)
            support = (a["skin_ring"]+b["skin_ring"])/2
            base = .25*alignment+.15*ratio+.15*spacing+.25*strength+.20*support
            position = math.exp(-.25*(a["prior_distance_squared"]+b["prior_distance_squared"]))
            score = (1-prior["weight"])*base+prior["weight"]*position
            region["pairs"].append({"roi_id": "eyes", "eyes": [a["id"], b["id"]], "centers": [a["center"], b["center"]],
                "score": score, "base_score": base, "position_score": position,
                "angle_degrees": angle, "distance": distance,
                "score_terms": {"alignment": alignment, "size_similarity": ratio, "spacing": spacing,
                                "response": strength, "skin_ring": support}})
    region["pair_count"] = len(region["pairs"])
    return gray, [region], rank_pairs([region], "score", config["display_pairs"])


def locate_eyes(bgr, skin, config, position_prior=None):
    if position_prior is not None and position_prior.get("mode") == "paired_eyes":
        return locate_in_eye_windows(bgr, skin, config, position_prior)
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    height, width = gray.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
    component_ids = sorted(range(1, count), key=lambda i: (-stats[i, cv2.CC_STAT_AREA], i))
    regions = []
    for rank, component_id in enumerate(component_ids[:config["max_components"]], 1):
        x, y, w, h, area = map(int, stats[component_id])
        padding = config["roi_padding_fraction"]
        x0, y0 = max(0, int(x - padding*w)), max(0, int(y - padding*h))
        x1, y1 = min(width, math.ceil(x + w*(1+padding))), min(height, math.ceil(y + h*(1+padding)))
        rw, rh = x1-x0, y1-y0
        region = {"id": f"r{rank}", "component_id": component_id, "component_area": area,
                  "box": [x0, y0, rw, rh], "candidates": [], "pairs": []}
        regions.append(region)
        # Tiny fragments provide no plausible scale for a pair of eyes.
        if rw < width*config["min_roi_width_fraction"] or rh < height*0.15:
            region["status"] = "roi_too_small"
            continue
        sy0 = y0 + int(rh*config["search_y_fraction"][0])
        sy1 = y0 + int(rh*config["search_y_fraction"][1])
        search_box = [x0, sy0, rw, sy1-sy0]
        region["search_box"] = search_box
        kw, kh = [odd_size(rw*f) for f in config["blackhat_kernel_fraction"]]
        response = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh)))
        crop = response[sy0:sy1, x0:x1]
        otsu, _ = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = max(config["minimum_response"], otsu)
        binary = np.zeros_like(gray)
        binary[sy0:sy1, x0:x1] = (crop > threshold).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8))
        n, labels, blob_stats, centers = cv2.connectedComponentsWithStats(binary, connectivity=8)
        region.update(status="ok", kernel=[kw, kh], threshold=float(threshold),
                      response=response, binary=binary)
        for label in range(1, n):
            bx, by, bw, bh, ba = map(int, blob_stats[label])
            if not (config["eye_width_fraction"][0]*rw <= bw <= config["eye_width_fraction"][1]*rw
                    and config["eye_height_fraction"][0]*rw <= bh <= config["eye_height_fraction"][1]*rw
                    and config["eye_area_fraction"][0]*rw*rh <= ba <= config["eye_area_fraction"][1]*rw*rh
                    and config["eye_aspect_ratio"][0] <= bw/bh <= config["eye_aspect_ratio"][1]):
                continue
            mask = (labels == label).astype(np.uint8)
            radius = odd_size(0.05*rw)
            ring = (cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))) > 0) & (mask == 0)
            region["candidates"].append({"id": f"r{rank}-e{len(region['candidates'])+1}",
                "center": centers[label].tolist(), "box": [bx, by, bw, bh], "area": ba,
                "response": float(response[mask > 0].mean()),
                "skin_ring": float(skin[ring].mean()) if ring.any() else 0.0})
        for a, b in combinations(region["candidates"], 2):
            left, right = sorted([a, b], key=lambda c: c["center"][0])
            dx, dy = np.subtract(right["center"], left["center"])
            distance = math.hypot(dx, dy)
            angle = math.degrees(math.atan2(dy, dx))
            ratio = min(a["area"], b["area"])/max(a["area"], b["area"])
            if not (config["pair_distance_fraction"][0]*rw <= distance <= config["pair_distance_fraction"][1]*rw
                    and abs(angle) <= config["max_pair_angle_degrees"] and ratio >= config["min_pair_area_ratio"]):
                continue
            alignment = 1 - abs(angle)/config["max_pair_angle_degrees"]
            spacing = max(0, 1 - abs(distance/rw - 0.36)/0.30)
            strength = min(1, (a["response"]+b["response"])/160)
            support = (a["skin_ring"]+b["skin_ring"])/2
            base_score = .25*alignment + .15*ratio + .15*spacing + .25*strength + .20*support
            position_score, weight = None, 0.0
            midpoint = (np.asarray(left["center"]) + right["center"])/2
            if position_prior is not None:
                normalized_midpoint = midpoint / [width, height]
                offset = (normalized_midpoint - position_prior["mean"]) / position_prior["sigma"]
                position_score = float(np.exp(-0.5*np.dot(offset, offset)))
                weight = position_prior["weight"]
            score = (1-weight)*base_score + weight*(position_score or 0)
            region["pairs"].append({"roi_id": region["id"], "eyes": [left["id"], right["id"]],
                "centers": [left["center"], right["center"]], "score": float(score),
                "base_score": float(base_score), "position_score": position_score,
                "midpoint": midpoint.tolist(),
                "angle_degrees": angle, "distance": distance,
                "score_terms": {"alignment": alignment, "size_similarity": ratio,
                                "spacing": spacing, "response": strength, "skin_ring": support}})
        region["pair_count"] = len(region["pairs"])
    return gray, regions, rank_pairs(regions, "score", config["display_pairs"])
