"""Build a validation gallery for the detected-eyes-to-face-ellipse experiment."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE.parent / "eye_check"))

from build_gallery import fit_position_prior
from eyes import locate_eyes
from eye_ellipse import (ellipse_from_eyes, ellipse_region, eye_centers,
                         fit_face_ellipse, fit_geometry_prior, geometry_sample)
from face_pixel.data_io import PROJECT, _read_csv, read_bgr, read_json
from face_pixel.dataset import binary_labels
from face_pixel.detection import detect
from face_pixel.evaluation import evaluate, summarize
from face_pixel.visualization import _font_path


def write_text_lf(path, value, encoding="utf-8"):
    with Path(path).open("w", encoding=encoding, newline="\n") as stream:
        stream.write(value)


def write_json_lf(path, value):
    write_text_lf(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _geometry_prior(rows, minimum_samples):
    samples, excluded = [], []
    for row in rows:
        if row["split"] != "train":
            continue
        label = np.asarray(Image.open(PROJECT / row["label_path"]))
        centers = eye_centers(label)
        if centers is None:
            excluded.append({"id": row["sample_id"], "reason": "missing_eye_label"})
            continue
        face, _ = binary_labels(label)
        geometry = fit_face_ellipse(face)
        if geometry is None:
            excluded.append({"id": row["sample_id"], "reason": "face_ellipse_unavailable"})
            continue
        sample = geometry_sample(centers, geometry)
        sample["id"] = row["sample_id"]
        samples.append(sample)
    prior = fit_geometry_prior(samples, minimum_samples)
    prior["training_ids"] = [sample["id"] for sample in samples]
    prior["excluded"] = excluded
    return prior


def _eye_error(centers, label):
    truth = eye_centers(label)
    if centers is None or truth is None:
        return None
    distance = float(np.linalg.norm(np.subtract(truth[1], truth[0])))
    if distance <= 0:
        return None
    errors = np.linalg.norm(np.subtract(centers, truth), axis=1) / distance
    return float(errors.max())


def _draw_geometry(bgr, geometry, color, eyes=None, label=None):
    canvas = bgr.copy()
    if geometry is not None:
        center = tuple(np.rint(geometry["center"]).astype(int))
        axes = tuple(np.maximum(1, np.rint(np.asarray(geometry["axes"]) / 2)).astype(int))
        cv2.ellipse(canvas, center, axes, geometry["angle_degrees"], 0, 360, color, 3, cv2.LINE_AA)
        cv2.drawMarker(canvas, center, color, cv2.MARKER_CROSS, 14, 2)
    if eyes is not None:
        points = [tuple(np.rint(point).astype(int)) for point in eyes]
        cv2.line(canvas, points[0], points[1], (0, 75, 255), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(canvas, point, 7, (0, 75, 255), 2, cv2.LINE_AA)
    if label:
        cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .68, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _mask_panel(mask, color=(230, 230, 230)):
    panel = np.full((*mask.shape, 3), 24, dtype=np.uint8)
    panel[mask.astype(bool)] = color
    return panel


def _truth_panel(bgr, gt, geometry):
    canvas = bgr.astype(float) * .55
    canvas[gt.astype(bool)] = bgr[gt.astype(bool)] * .35 + np.array([70, 180, 70]) * .65
    return _draw_geometry(canvas.clip(0, 255).astype(np.uint8), geometry, (40, 240, 40))


def _error_panel(bgr, pred, gt, valid):
    pred, gt, valid = (value.astype(bool) for value in (pred, gt, valid))
    canvas = bgr.astype(float) * .32
    for condition, color in (
        (pred & gt & valid, (65, 185, 65)),
        (pred & ~gt & valid, (45, 60, 235)),
        (~pred & gt & valid, (235, 105, 45)),
        (~valid, (0, 195, 255)),
    ):
        canvas[condition] = bgr[condition] * .25 + np.array(color) * .75
    return canvas.clip(0, 255).astype(np.uint8)


def _save_bgr(path, bgr):
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(path, quality=91)


def _board(path, panels):
    tile, heading = 384, 36
    canvas = Image.new("RGB", (tile * 3, (tile + heading) * 2), "#edf1f5")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(_font_path(), 17)
    for index, (title, bgr) in enumerate(panels):
        x = (index % 3) * tile
        y = (index // 3) * (tile + heading)
        draw.text((x + 8, y + 7), title, font=font, fill="#183046")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        canvas.paste(Image.fromarray(rgb).resize((tile, tile), Image.Resampling.LANCZOS), (x, y + heading))
    canvas.save(path, quality=91)


def _metric_subset(records, key, require_detection=False):
    values = []
    for record in records:
        if require_detection and record["status"] != "ok":
            continue
        metrics = record.get(key)
        if metrics is not None:
            values.append(metrics)
    return summarize(values) if values else None


def build(output, config_path):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Use a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    config = read_json(config_path)
    skin_params = read_json(PROJECT / "configs/skin_params.json")
    detector = read_json(PROJECT / "configs/original_config.json")
    detector["threshold"]["k"] = config["k"]
    detector["postprocess"]["method"] = "none"
    manifest = _read_csv(PROJECT / "data/split_manifest.csv", {"sample_id", "split", "image_path", "label_path"})
    position_prior = fit_position_prior(manifest, config["position_prior"])
    geometry_prior = _geometry_prior(manifest, config["minimum_geometry_samples"])
    rows = [row for row in manifest if row["split"] in ("val", "validation")]
    source_head = None
    if (PROJECT / ".git").exists():
        try:
            source_head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

    with tempfile.TemporaryDirectory(prefix=".eye-ellipse-", dir=output.parent) as temporary:
        stage = Path(temporary) / output.name
        assets = stage / "assets"
        assets.mkdir(parents=True)
        records = []
        for index, row in enumerate(rows, 1):
            sample_id = row["sample_id"]
            asset = assets / sample_id
            asset.mkdir()
            bgr = read_bgr(PROJECT / row["image_path"])
            skin = detect(bgr, skin_params, detector)
            _, _, pairs = locate_eyes(bgr, skin, config, position_prior)
            selected = pairs[0] if pairs else None

            # Prediction is complete before this validation annotation is opened.
            detected_geometry = ellipse_from_eyes(selected["centers"], geometry_prior) if selected else None
            detected_mask = ellipse_region(bgr.shape[:2], detected_geometry) if selected else np.zeros(bgr.shape[:2], np.uint8)
            fixed_eyes = [(np.asarray(eye["mean"]) * [bgr.shape[1], bgr.shape[0]]).tolist()
                          for eye in position_prior["eyes"]]
            prior_geometry = ellipse_from_eyes(fixed_eyes, geometry_prior)
            prior_mask = ellipse_region(bgr.shape[:2], prior_geometry)

            label = np.asarray(Image.open(PROJECT / row["label_path"]))
            gt, valid = binary_labels(label)
            oracle_geometry = fit_face_ellipse(gt)
            detected_metrics = evaluate(detected_mask, gt, valid) if selected else None
            prior_metrics = evaluate(prior_mask, gt, valid)
            eye_error = _eye_error(selected["centers"], label) if selected else None

            original_eyes = _draw_geometry(bgr, None, (0, 0, 0), selected["centers"] if selected else None,
                                           None if selected else "NO EYE PAIR")
            detected_overlay = _draw_geometry(bgr, detected_geometry, (0, 165, 255),
                                               selected["centers"] if selected else None,
                                               None if selected else "NO ELLIPSE")
            prior_overlay = _draw_geometry(bgr, prior_geometry, (210, 80, 210), fixed_eyes)
            truth_overlay = _truth_panel(bgr, gt, oracle_geometry)
            errors = _error_panel(bgr, detected_mask, gt, valid)
            _save_bgr(asset / "original_eyes.jpg", original_eyes)
            _save_bgr(asset / "detected_overlay.jpg", detected_overlay)
            Image.fromarray(detected_mask * 255).save(asset / "detected_mask.png")
            _save_bgr(asset / "prior_overlay.jpg", prior_overlay)
            _save_bgr(asset / "truth_overlay.jpg", truth_overlay)
            _save_bgr(asset / "errors.jpg", errors)
            _board(asset / "overview.jpg", [
                ("1  图像检测到的双眼", original_eyes),
                ("2  双眼推算的椭圆", detected_overlay),
                ("3  椭圆二值区域", _mask_panel(detected_mask)),
                ("4  固定平均眼位对照", prior_overlay),
                ("5  人工真值与事后拟合参考", truth_overlay),
                ("6  椭圆误差：绿TP 红FP 蓝FN", errors),
            ])
            record = {
                "id": sample_id,
                "status": "ok" if selected else "no_eye_pair",
                "eye_pair": selected,
                "eye_normalized_error": eye_error,
                "detected_geometry": detected_geometry,
                "detected_metrics": detected_metrics,
                "prior_only_geometry": prior_geometry,
                "prior_only_metrics": prior_metrics,
                "oracle_geometry_display_only": oracle_geometry,
            }
            records.append(record)
            f1 = detected_metrics["f1"] if detected_metrics else None
            print(f"{index}/{len(rows)} {sample_id}: {record['status']}, F1={f1}", flush=True)

        detected = [record for record in records if record["status"] == "ok"]
        summary = {
            "images": len(records),
            "images_with_detected_eye_pair": len(detected),
            "images_without_detected_eye_pair": len(records) - len(detected),
            "coverage_rate": len(detected) / len(records) if records else None,
            "detected_eye_ellipse_on_detected_images": _metric_subset(records, "detected_metrics", True),
            "prior_only_on_same_detected_images": _metric_subset(
                [{**record, "comparison": record["prior_only_metrics"]} for record in records], "comparison", True),
            "prior_only_on_all_images": _metric_subset(records, "prior_only_metrics"),
            "eye_localization": {
                "median_normalized_error_when_detected": float(np.median([
                    record["eye_normalized_error"] for record in detected
                    if record["eye_normalized_error"] is not None
                ])) if detected else None,
                "hits_at_0.15": sum(record["eye_normalized_error"] is not None and record["eye_normalized_error"] <= .15
                                    for record in detected),
                "hits_at_0.25": sum(record["eye_normalized_error"] is not None and record["eye_normalized_error"] <= .25
                                    for record in detected),
            },
            "failed_ids": [record["id"] for record in records if record["status"] != "ok"],
        }
        write_json_lf(stage / "records.json", records)
        write_json_lf(stage / "summary.json", summary)
        write_json_lf(stage / "settings.json", {
            "experiment": "lab4/eye_ellipse",
            "source_head": source_head,
            "split": "validation",
            "config": config,
            "detector": detector,
            "position_prior": position_prior,
            "eye_to_face_geometry_prior": geometry_prior,
            "validation_label_policy": "opened only after eye selection and ellipse prediction; evaluation/display only",
            "opencv": cv2.__version__,
        })
        with (stage / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            fields = ["id", "status", "eye_normalized_error", "precision", "recall", "f1", "iou",
                      "prior_only_precision", "prior_only_recall", "prior_only_f1", "prior_only_iou"]
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for record in records:
                metrics = record["detected_metrics"] or {}
                prior_metrics = record["prior_only_metrics"]
                writer.writerow({
                    "id": record["id"], "status": record["status"],
                    "eye_normalized_error": record["eye_normalized_error"],
                    **{name: metrics.get(name) for name in ("precision", "recall", "f1", "iou")},
                    **{f"prior_only_{name}": prior_metrics.get(name) for name in ("precision", "recall", "f1", "iou")},
                })
        payload = json.dumps({"records": records, "summary": summary, "geometry_prior": geometry_prior},
                             ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
        html = (HERE / "gallery.html").read_text(encoding="utf-8").replace("__DATA__", payload)
        write_text_lf(stage / "index.html", html)
        write_text_lf(stage / "README.md",
            "# 双眼生成脸部椭圆：验证结果\n\n打开 `index.html` 查看图册。"
            "本目录只评估椭圆估计，没有对肤色二值图进行形态学修整。\n",
        )
        source = stage / "source"
        source.mkdir()
        for name in ("build_gallery.py", "eye_ellipse.py", "gallery.html", "config.json", "README.md"):
            shutil.copyfile(HERE / name, source / name)
        stage.rename(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(output / "index.html")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(HERE / "config.json"))
    arguments = parser.parse_args()
    build(arguments.output, arguments.config)
