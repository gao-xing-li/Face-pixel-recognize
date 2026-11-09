"""One round of weak morphology, ellipse estimation and local refinement."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE.parent / "ellipse_check"))
from ellipse import extract_components, candidate_groups, fit_candidate, ellipse_region
from build_gallery import component_overview
from face_pixel.data_io import PROJECT, _read_csv, read_bgr, read_json, write_json
from face_pixel.dataset import binary_labels
from face_pixel.detection import detect, fill_holes, postprocess
from face_pixel.evaluation import evaluate, summarize
from face_pixel.visualization import _font_path

VARIANTS = {"raw": "原始", "close": "轻量闭运算", "open": "轻量开运算"}
STAGES = {"raw": "原始阈值", "lab3": "Lab3 最大块填孔", "local": "仅限制外椭圆", "final": "一轮完整修整"}


def morphology(mask, operation, size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    padded = np.pad(mask, size)
    changed = cv2.morphologyEx(padded, operation, kernel, iterations=1,
                               borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return changed[size:-size, size:-size]


def refine(raw, geometry, config):
    inner = ellipse_region(raw.shape, geometry, config["refine_inner_scale"])
    outer = ellipse_region(raw.shape, geometry, config["refine_outer_scale"])
    local = raw & outer
    seed = raw & inner
    stages = {"inner": inner, "outer": outer, "local": local, "seed": seed}
    if not seed.any():
        return "no_inner_seed", stages
    repair = fill_holes(morphology(local, cv2.MORPH_CLOSE, config["strong_kernel"])) & outer
    proposal = (repair & inner) | (local & (outer & (1 - inner)))
    _, labels = cv2.connectedComponents(proposal, connectivity=8)
    seed_labels = np.unique(labels[seed.astype(bool)])
    seed_labels = seed_labels[seed_labels > 0]
    final = np.isin(labels, seed_labels).astype(np.uint8)
    stages.update(repair=repair, proposal=proposal, final=final)
    return "ok", stages


def support_descriptors(raw, gradient, region, geometry):
    footprint = np.ones((5, 5), dtype=np.uint8)
    padded = np.pad(region, 4)
    dilated = cv2.dilate(padded, footprint)[4:-4, 4:-4]
    eroded = cv2.erode(padded, footprint)[4:-4, 4:-4]
    farther = cv2.dilate(np.pad(region, 8), np.ones((9, 9), np.uint8))[8:-8, 8:-8]
    band = (dilated > eroded)
    adjacent = (farther > dilated)
    mean = lambda selected: float(gradient[selected].mean()) if selected.any() else None
    yy, xx = np.indices(raw.shape)
    cx, cy = geometry["center"]
    quadrants = []
    for selected in ((xx < cx) & (yy < cy), (xx >= cx) & (yy < cy),
                     (xx < cx) & (yy >= cy), (xx >= cx) & (yy >= cy)):
        selected &= region.astype(bool)
        quadrants.append(float(raw[selected].mean()) if selected.any() else None)
    return {"gradient_band_mean": mean(band), "gradient_adjacent_mean": mean(adjacent),
            "quadrant_support": quadrants}, band


def pixel_changes(before, after, gt, valid):
    added = (after > before) & valid.astype(bool)
    removed = (before > after) & valid.astype(bool)
    face = gt.astype(bool)
    return {"added_tp": int((added & face).sum()), "added_fp": int((added & ~face).sum()),
            "removed_tp": int((removed & face).sum()), "removed_fp": int((removed & ~face).sum())}


def stage_board(path, candidate, stages, raw, gt, valid, font):
    tile, gap, caption = 256, 12, 30
    board = Image.new("RGB", (4 * tile + 5 * gap, 2 * (tile + caption + gap) + gap), "white")
    draw = ImageDraw.Draw(board)
    changes = []
    for changed in (stages["final"] > raw, raw > stages["final"]):
        rgb = np.zeros((*raw.shape, 3), np.uint8)
        rgb[changed & (gt > 0) & (valid > 0)] = (244, 135, 49)
        rgb[changed & (gt == 0) & (valid > 0)] = (45, 191, 203)
        rgb[changed & (valid == 0)] = (160, 160, 160)
        changes.append(Image.fromarray(rgb))
    panels = [
        ("候选区域", candidate), ("仅限外椭圆 Mlocal", stages["local"]),
        ("内部种子 S", stages["seed"]), ("闭运算与填孔建议 P", stages["repair"]),
        ("内部接受修补", stages["proposal"]), ("种子关联后最终结果", stages["final"]),
        ("相对阈值新增", changes[0]), ("相对阈值删除", changes[1]),
    ]
    for i, (label, picture) in enumerate(panels):
        x = gap + (i % 4) * (tile + gap)
        y = gap + (i // 4) * (tile + caption + gap)
        draw.text((x, y), label, font=font, fill="#173f3b")
        if isinstance(picture, np.ndarray):
            picture = Image.fromarray(picture * 255).convert("RGB")
        board.paste(picture.resize((tile, tile), Image.Resampling.NEAREST), (x, y + caption))
    board.save(path)


def run(output, config_path, splits):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    config = read_json(config_path)
    for key in ("weak_kernel", "strong_kernel"):
        if config[key] < 1 or config[key] % 2 == 0:
            raise ValueError("Use positive odd kernel sizes")
    if not 0 < config["refine_inner_scale"] < 1 < config["refine_outer_scale"]:
        raise ValueError("Use inner scale < 1 and outer scale > 1")
    params = read_json(PROJECT / "configs/skin_params.json")
    detector = read_json(PROJECT / "configs/original_config.json")
    detector["threshold"]["k"] = config["k"]
    detector["postprocess"]["method"] = "none"
    rows = _read_csv(PROJECT / "data/split_manifest.csv", {"sample_id", "split", "image_path", "label_path"})
    rows = [r for r in rows if ("validation" if r["split"] == "val" else r["split"]) in splits]
    if not rows:
        raise ValueError("No images in requested splits")
    font = ImageFont.truetype(_font_path(), 18)
    records, flat_rows = [], []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".estimate-refine-", dir=output.parent) as temporary:
        stage = Path(temporary) / "gallery"
        (stage / "assets").mkdir(parents=True)
        for index, row in enumerate(rows):
            sid = row["sample_id"]
            if Path(sid).name != sid or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in sid):
                raise ValueError("Invalid sample ID")
            asset = stage / "assets" / sid
            asset.mkdir()
            split = "validation" if row["split"] == "val" else row["split"]
            bgr = read_bgr(PROJECT / row["image_path"])
            raw = detect(bgr, params, detector)
            lab3 = postprocess(raw, "largest_fill")
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            weak = {"raw": raw, "close": morphology(raw, cv2.MORPH_CLOSE, config["weak_kernel"]),
                    "open": morphology(raw, cv2.MORPH_OPEN, config["weak_kernel"])}
            record = {"id": sid, "split": split, "width": raw.shape[1], "height": raw.shape[0], "variants": {}}
            computations = []
            # All candidate fitting and refinement is computed before ground truth is read.
            for variant, mask in weak.items():
                labels, components = extract_components(mask, bgr, config)
                by_id = {c["id"]: c for c in components}
                candidates = []
                Image.fromarray(mask * 255).save(asset / f"{variant}-weak.png")
                component_overview(bgr, labels, components).save(asset / f"{variant}-components.jpg", quality=90)
                for number, ids in enumerate(candidate_groups(components, config), start=1):
                    name = f"{variant}-c{number:03d}"
                    candidate_mask = np.isin(labels, ids).astype(np.uint8)
                    points = [point for label in ids for point in by_id[label]["points"]]
                    fit, region = fit_candidate(points, candidate_mask, raw, config["inner_scale"])
                    candidate = {"id": name, "components": list(ids), "kind": "single" if len(ids) == 1 else "group",
                                 "area": int(candidate_mask.sum()), **fit}
                    if region is not None:
                        descriptors, band = support_descriptors(raw, gradient, region, candidate["geometry"])
                        candidate.update(descriptors)
                        status, stages = refine(raw, candidate["geometry"], config)
                        candidate["refine_status"] = status
                        edge_map = np.repeat(gradient[:, :, None], 3, axis=2)
                        edge_map[band] = (20, 220, 220)
                        Image.fromarray(edge_map).save(asset / f"{name}-edges.png")
                        computations.append((candidate, candidate_mask, stages))
                    else:
                        candidate["refine_status"] = "ellipse_unavailable"
                    candidates.append(candidate)
                record["variants"][variant] = {"components": components, "candidates": candidates}
            with Image.open(PROJECT / row["label_path"]) as label:
                gt, valid = binary_labels(np.asarray(label))
            record["baseline_metrics"] = {"raw": evaluate(raw, gt, valid), "lab3": evaluate(lab3, gt, valid)}
            for candidate, mask, stages in computations:
                candidate["metrics"] = {"local": evaluate(stages["local"], gt, valid)}
                if candidate["refine_status"] == "ok":
                    candidate["metrics"]["final"] = evaluate(stages["final"], gt, valid)
                    candidate["changes_from_raw"] = pixel_changes(raw, stages["final"], gt, valid)
                    candidate["changes_from_local"] = pixel_changes(stages["local"], stages["final"], gt, valid)
                    stage_board(asset / f"{candidate['id']}-stages.png", mask, stages, raw, gt, valid, font)
                    Image.fromarray(stages["final"] * 255).save(asset / f"{candidate['id']}-final.png")
                else:
                    Image.fromarray(stages["local"] * 255).save(asset / f"{candidate['id']}-local.png")
            for variant, info in record["variants"].items():
                for candidate in info["candidates"]:
                    flat = {"sample_id": sid, "split": split, "variant": variant, "candidate": candidate["id"],
                            "components": "+".join(map(str, candidate["components"])), "status": candidate["refine_status"]}
                    for name in ("raw_support", "inner_support", "gradient_band_mean", "gradient_adjacent_mean"):
                        flat[name] = candidate.get(name)
                    for stage_name in ("local", "final"):
                        metrics = candidate.get("metrics", {}).get(stage_name, {})
                        for name in ("precision", "recall", "f1", "iou", "tp", "fp", "fn"):
                            flat[f"{stage_name}_{name}"] = metrics.get(name)
                    flat_rows.append(flat)
            Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(asset / "original.jpg", quality=93)
            for name, mask in (("raw", raw), ("lab3", lab3), ("truth", gt)):
                Image.fromarray(mask * 255).save(asset / f"{name}.png")
            Image.fromarray(gradient).save(asset / "gradient.png")
            records.append(record)
            if (index + 1) % 4 == 0 or index + 1 == len(rows):
                print(f"Processed {index + 1}/{len(rows)} images, {len(flat_rows)} candidates", flush=True)
        summary = {}
        for split in splits:
            subset = [r for r in records if r["split"] == split]
            if not subset:
                continue
            summary[split] = {"images": len(subset), "variants": {}}
            for variant in VARIANTS:
                eligible = [r for r in subset if r["variants"][variant]["candidates"] and
                            r["variants"][variant]["candidates"][0]["refine_status"] == "ok"]
                scores = {}
                for name in STAGES:
                    values = []
                    for record in eligible:
                        if name in ("raw", "lab3"):
                            values.append(record["baseline_metrics"][name])
                        else:
                            values.append(record["variants"][variant]["candidates"][0]["metrics"][name])
                    scores[name] = summarize(values) if values else None
                all_candidates = [c for r in subset for c in r["variants"][variant]["candidates"]]
                summary[split]["variants"][variant] = {"paired_images": len(eligible), "scores": scores,
                    "candidate_statuses": dict(Counter(c["refine_status"] for c in all_candidates))}
        payload = {"records": records, "summary": summary, "config": config, "variants": VARIANTS}
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
        template = (HERE / "gallery.html").read_text(encoding="utf-8")
        (stage / "index.html").write_text(template.replace("__DATA__", encoded), encoding="utf-8")
        write_json(stage / "records.json", records)
        write_json(stage / "summary.json", summary)
        write_json(stage / "settings.json", {"experiment": "lab4/estimate_refine", "config": config,
                   "skin_params": params, "threshold": detector, "splits": splits,
                   "reference": "first/largest retained single candidate per variant; paired valid image subset"})
        with (stage / "per_candidate.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]) if flat_rows else ["sample_id"])
            writer.writeheader()
            writer.writerows(flat_rows)
        shutil.copyfile(HERE / "README.md", stage / "README.md")
        (stage / "source").mkdir()
        for path in [HERE / "run.py", HERE / "gallery.html", HERE / "config.json",
                     HERE.parent / "ellipse_check/ellipse.py"]:
            shutil.copyfile(path, stage / "source" / path.name)
        stage.rename(output)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--splits", nargs="+", choices=("validation", "test", "train"), default=["validation"])
    args = parser.parse_args()
    run(args.output, args.config, list(dict.fromkeys(args.splits)))
