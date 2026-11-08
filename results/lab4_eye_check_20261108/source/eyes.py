"""Handcrafted eye proposals: grayscale black-hat, blobs, then eye-pair geometry.

No annotations or learned eye/face detector enter this module.
"""

from itertools import combinations
import math

import cv2
import numpy as np


def odd_size(value):
    return max(3, int(round(value)) | 1)


def locate_eyes(bgr, skin, config):
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    height, width = gray.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
    component_ids = sorted(range(1, count), key=lambda i: (-stats[i, cv2.CC_STAT_AREA], i))
    regions, all_pairs = [], []
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
            score = .25*alignment + .15*ratio + .15*spacing + .25*strength + .20*support
            region["pairs"].append({"roi_id": region["id"], "eyes": [left["id"], right["id"]],
                "centers": [left["center"], right["center"]], "score": float(score),
                "angle_degrees": angle, "distance": distance,
                "score_terms": {"alignment": alignment, "size_similarity": ratio,
                                "spacing": spacing, "response": strength, "skin_ring": support}})
        region["pairs"].sort(key=lambda p: -p["score"])
        region["pair_count"] = len(region["pairs"])
        region["pairs"] = region["pairs"][:config["display_pairs"]]
        all_pairs.extend(region["pairs"])
    # Overlapping search boxes can propose the same physical pair.
    ranked = []
    for pair in sorted(all_pairs, key=lambda p: -p["score"]):
        if not any(np.max(np.linalg.norm(np.subtract(pair["centers"], p["centers"]), axis=1)) < 4 for p in ranked):
            ranked.append(pair)
    return gray, regions, ranked[:config["display_pairs"]]
