"""Build the standalone Lab4 ellipse inspection gallery."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from ellipse import extract_components, candidate_groups, fit_candidate
from face_pixel.data_io import PROJECT, _read_csv, read_bgr, read_json, write_json
from face_pixel.dataset import binary_labels
from face_pixel.detection import detect, validate_model
from face_pixel.evaluation import evaluate, summarize


def component_overview(bgr, labels, components):
    canvas = (bgr * 0.35).astype(np.uint8)
    for c in components:
        if not c["retained"]:
            continue
        color = np.array([(c["id"] * 67) % 175 + 80, (c["id"] * 103) % 175 + 80,
                          (c["id"] * 139) % 175 + 80])
        selected = labels == c["id"]
        canvas[selected] = (bgr[selected] * 0.4 + color * 0.6).astype(np.uint8)
        x, y, _, _ = c["box"]
        cv2.putText(canvas, str(c["id"]), (x, max(y + 15, 15)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, str(c["id"]), (x, max(y + 15, 15)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def build(output, config_path, splits, preprocess="raw"):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    config = read_json(config_path)
    config["preprocess"] = preprocess
    if (config["min_component_area"] < 1 or config["max_box_gap"] < 0 or
            config["max_chroma_distance"] < 0 or config["neighbors_per_component"] < 0 or
            config["max_points_per_contour"] < 5 or not 0 < config["inner_scale"] < 1):
        raise ValueError("Invalid candidate settings")
    params = read_json(PROJECT / "configs/skin_params.json")
    validate_model(params)
    detector = read_json(PROJECT / "configs/original_config.json")
    detector["threshold"]["k"] = config["k"]
    detector["postprocess"]["method"] = "none"
    rows = _read_csv(PROJECT / "data/split_manifest.csv", {"sample_id", "split", "image_path", "label_path"})
    rows = [r for r in rows if ("validation" if r["split"] == "val" else r["split"]) in splits]
    if not rows:
        raise ValueError("No images in requested splits")
    records, flat_records = [], []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ellipse-check-", dir=output.parent) as tmp:
        stage = Path(tmp) / "gallery"
        (stage / "assets").mkdir(parents=True)
        for index, row in enumerate(rows):
            sid = row["sample_id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", sid):
                raise ValueError(f"Invalid image ID: {sid}")
            split = "validation" if row["split"] == "val" else row["split"]
            asset = stage / "assets" / sid
            asset.mkdir()
            bgr = read_bgr(PROJECT / row["image_path"])
            raw = detect(bgr, params, detector)
            processed = raw
            if preprocess != "raw":
                size = config["weak_kernel"]
                element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
                operation = cv2.MORPH_CLOSE if preprocess == "close" else cv2.MORPH_OPEN
                padded = cv2.morphologyEx(np.pad(raw, size), operation, element,
                                         borderType=cv2.BORDER_CONSTANT, borderValue=0)
                processed = padded[size:-size, size:-size]
            labels, components = extract_components(processed, bgr, config)
            by_id = {c["id"]: c for c in components}
            candidates = []
            # Candidate construction and fitting receive image information only.
            for number, ids in enumerate(candidate_groups(components, config), start=1):
                name = f"c{number:03d}"
                mask = np.isin(labels, ids).astype(np.uint8)
                points = [p for label in ids for p in by_id[label]["points"]]
                fit, region = fit_candidate(points, mask, raw, config["inner_scale"])
                candidates.append({"id": name, "components": list(ids), "kind": "single" if len(ids) == 1 else "group",
                                   "area": int(mask.sum()), "point_count": len(points), **fit})
                Image.fromarray(mask * 255).save(asset / f"{name}.png")
                if region is not None:
                    Image.fromarray(region * 255).save(asset / f"{name}-ellipse.png")
            # Ground truth enters only after candidates and their ellipses are fixed.
            with Image.open(PROJECT / row["label_path"]) as label:
                gt, valid = binary_labels(np.asarray(label))
            for candidate in candidates:
                if candidate["status"] == "ok":
                    region_path = asset / f"{candidate['id']}-ellipse.png"
                    with Image.open(region_path) as image:
                        region = (np.asarray(image) > 0).astype(np.uint8)
                    candidate["ellipse_metrics"] = evaluate(region, gt, valid)
                flat = {"sample_id": sid, "split": split, "candidate_id": candidate["id"],
                        "components": "+".join(map(str, candidate["components"])), "kind": candidate["kind"],
                        "status": candidate["status"], "area": candidate["area"], "point_count": candidate["point_count"]}
                for key in ("raw_support", "inner_support", "outer_support", "candidate_coverage", "axis_ratio"):
                    flat[key] = candidate.get(key)
                flat.update({f"ellipse_{key}": candidate.get("ellipse_metrics", {}).get(key)
                             for key in ("precision", "recall", "f1", "iou", "tp", "fp", "fn")})
                flat_records.append(flat)
            original = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            original.save(asset / "original.jpg", quality=93)
            Image.fromarray(raw * 255).save(asset / "raw.png")
            Image.fromarray(processed * 255).save(asset / "processed.png")
            component_overview(bgr, labels, components).save(asset / "components.jpg", quality=93)
            truth_rgb = np.repeat((gt * 255)[:, :, None], 3, axis=2)
            truth_rgb[valid == 0] = (255, 195, 0)
            Image.fromarray(truth_rgb).save(asset / "truth.png")
            records.append({"id": sid, "split": split, "width": bgr.shape[1], "height": bgr.shape[0],
                            "components": components, "candidates": candidates})
            if (index + 1) % 20 == 0 or index + 1 == len(rows):
                print(f"Processed {index + 1}/{len(rows)} images, {len(flat_records)} candidates", flush=True)
        summaries = {}
        for split in splits:
            subset = [r for r in records if r["split"] == split]
            if not subset:
                continue
            all_candidates = [c for r in subset for c in r["candidates"]]
            # Largest single component is a reproducible browsing reference, not an IoU-selected winner.
            reference_metrics = [r["candidates"][0]["ellipse_metrics"] for r in subset
                                 if r["candidates"] and r["candidates"][0]["status"] == "ok"]
            summaries[split] = {"images": len(subset), "candidates": len(all_candidates),
                                "status_counts": dict(Counter(c["status"] for c in all_candidates)),
                                "no_candidate_images": sum(not r["candidates"] for r in subset),
                                "largest_single_reference": summarize(reference_metrics) if reference_metrics else None}
        payload = {"records": records, "summary": summaries, "config": config}
        template = (HERE / "gallery.html").read_text(encoding="utf-8")
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
        (stage / "index.html").write_text(template.replace("__DATA__", encoded), encoding="utf-8")
        write_json(stage / "candidates.json", records)
        write_json(stage / "summary.json", summaries)
        git = ["git", "-c", f"safe.directory={PROJECT.as_posix()}", "-C", str(PROJECT)]
        head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        write_json(stage / "settings.json", {"experiment": "lab4/ellipse_check", "source_head": head,
                   "candidate_settings": config, "threshold_config": detector, "skin_params": params,
                   "opencv": cv2.__version__, "splits": splits,
                   "reference": "largest retained single component; no ground-truth-based ranking"})
        with (stage / "per_candidate.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            fields = ["sample_id", "split", "candidate_id", "components", "kind", "status", "area", "point_count",
                      "raw_support", "inner_support", "outer_support", "candidate_coverage", "axis_ratio",
                      "ellipse_precision", "ellipse_recall", "ellipse_f1", "ellipse_iou", "ellipse_tp", "ellipse_fp", "ellipse_fn"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat_records)
        shutil.copyfile(HERE / "README.md", stage / "README.md")
        # Keep the exact experiment source alongside a portable result folder.
        (stage / "source").mkdir()
        for name in ("build_gallery.py", "ellipse.py", "gallery.html", "config.json"):
            shutil.copyfile(HERE / name, stage / "source" / name)
        stage.rename(output)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--splits", nargs="+", choices=("validation", "test", "train"), default=["validation"])
    parser.add_argument("--preprocess", choices=("raw", "close", "open", "all"), default="all")
    args = parser.parse_args()
    splits = list(dict.fromkeys(args.splits))
    if args.preprocess != "all":
        build(args.output, args.config, splits, args.preprocess)
    else:
        root = Path(args.output).resolve()
        if root.exists():
            parser.error("Output already exists; choose a new directory")
        for method in ("raw", "close", "open"):
            build(root / method, args.config, splits, method)
        (root / "index.html").write_text('''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lab4 前三连通块椭圆对照</title>
<style>body{font:17px/1.8 "Microsoft YaHei",sans-serif;max-width:850px;margin:60px auto;padding:20px;background:#f2f6f3;color:#173f3b}a{color:#127453}li{margin:20px 0}</style>
<h1>前三连通块 · 画椭圆对照</h1><p>每张图按面积保留前三个连通块，分别画椭圆，页面可切换第 1、2、3 块。</p>
<ul><li><a href="raw/index.html">原始肤色阈值</a></li><li><a href="close/index.html">5×5 轻量闭运算后</a></li><li><a href="open/index.html">5×5 轻量开运算后</a></li></ul>
<p>本轮范围到椭圆可视化。三种输入分别取各自的前三个块，块编号不跨输入对应。参数与图像划分见各页面的运行记录。</p></html>'''.replace("5×5", "{0}×{0}".format(read_json(args.config)["weak_kernel"])), encoding="utf-8")
        print(root / "index.html")
