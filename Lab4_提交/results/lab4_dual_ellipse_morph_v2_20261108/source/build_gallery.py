"""Build the nested-ellipse constrained morphology validation gallery."""

import argparse
from collections import Counter
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
sys.path.insert(0, str(HERE.parent / "eye_ellipse"))

from dual_ellipse import choose_eye_pair, ellipse_candidates, refine, select_candidate
from eyes import locate_eyes
from eye_ellipse import (ellipse_from_eyes, ellipse_region, eye_centers,
                         fit_face_ellipse, fit_geometry_prior, geometry_sample)
from face_pixel.data_io import PROJECT, _read_csv, read_bgr, read_json
from face_pixel.dataset import binary_labels
from face_pixel.detection import detect, postprocess
from face_pixel.evaluation import evaluate, summarize
from face_pixel.visualization import _font_path


def write_text_lf(path, value, encoding="utf-8"):
    with Path(path).open("w", encoding=encoding, newline="\n") as stream:
        stream.write(value)


def write_json_lf(path, value):
    write_text_lf(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def fit_position_prior(rows, settings):
    points, sample_ids = [], []
    for row in rows:
        if row["split"] != "train":
            continue
        label = np.asarray(Image.open(PROJECT / row["label_path"]))
        centers = eye_centers(label)
        if centers is not None:
            points.append(np.asarray(centers) / [label.shape[1], label.shape[0]])
            sample_ids.append(row["sample_id"])
    if not points:
        raise ValueError("No training images contain both eye labels")
    means = np.mean(points, axis=0)
    measured = np.std(points, axis=0, ddof=0)
    return {
        "mode": "paired_eyes",
        "weight": settings["weight"],
        "window_radius_sigma": settings["window_radius_sigma"],
        "training_images": len(points),
        "training_ids": sample_ids,
        "normalization": "x/image_width, y/image_height; label eye centroids sorted by image x",
        "eyes": [
            {
                "side": side,
                "mean": means[index].tolist(),
                "measured_sigma": measured[index].tolist(),
                "sigma": np.maximum(measured[index], settings["minimum_sigma_fraction"]).tolist(),
            }
            for index, side in enumerate(("image_left", "image_right"))
        ],
    }


def fit_eye_face_prior(rows, minimum_samples):
    samples, excluded = [], []
    for row in rows:
        if row["split"] != "train":
            continue
        label = np.asarray(Image.open(PROJECT / row["label_path"]))
        centers = eye_centers(label)
        face, _ = binary_labels(label)
        geometry = fit_face_ellipse(face)
        if centers is None or geometry is None:
            excluded.append(row["sample_id"])
            continue
        sample = geometry_sample(centers, geometry)
        sample["id"] = row["sample_id"]
        samples.append(sample)
    prior = fit_geometry_prior(samples, minimum_samples)
    prior["training_ids"] = [sample["id"] for sample in samples]
    prior["excluded_ids"] = excluded
    return prior


def scaled_region(shape, geometry, scale):
    scaled = {**geometry, "axes": (np.asarray(geometry["axes"]) * scale).tolist()}
    return ellipse_region(shape, scaled)


def draw_nested(bgr, geometry, inner_scale, outer_scale, eyes=None, label=None):
    canvas = bgr.copy()
    if geometry is not None:
        center = tuple(np.rint(geometry["center"]).astype(int))
        angle = geometry["angle_degrees"]
        for scale, color, width in ((outer_scale, (255, 190, 20), 3), (inner_scale, (40, 220, 80), 3)):
            axes = tuple(np.maximum(1, np.rint(np.asarray(geometry["axes"]) * scale / 2)).astype(int))
            cv2.ellipse(canvas, center, axes, angle, 0, 360, color, width, cv2.LINE_AA)
    if eyes is not None:
        points = [tuple(np.rint(point).astype(int)) for point in eyes]
        cv2.line(canvas, points[0], points[1], (0, 70, 255), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(canvas, point, 7, (0, 70, 255), 2, cv2.LINE_AA)
    if label:
        cv2.putText(canvas, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, .68, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def mask_panel(mask, color=(225, 225, 225)):
    panel = np.full((*mask.shape, 3), 25, np.uint8)
    panel[mask.astype(bool)] = color
    return panel


def truth_panel(bgr, gt):
    canvas = bgr.astype(float) * .42
    selected = gt.astype(bool)
    canvas[selected] = bgr[selected] * .30 + np.array([60, 190, 60]) * .70
    return canvas.clip(0, 255).astype(np.uint8)


def error_panel(bgr, prediction, gt, valid):
    prediction, gt, valid = (mask.astype(bool) for mask in (prediction, gt, valid))
    canvas = bgr.astype(float) * .30
    for selected, color in (
        (prediction & gt & valid, (65, 185, 65)),
        (prediction & ~gt & valid, (45, 60, 235)),
        (~prediction & gt & valid, (235, 105, 45)),
        (~valid, (0, 195, 255)),
    ):
        canvas[selected] = bgr[selected] * .25 + np.asarray(color) * .75
    return canvas.clip(0, 255).astype(np.uint8)


def save_bgr(path, bgr):
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(path, quality=91)


def board(path, panels):
    tile, heading = 320, 34
    canvas = Image.new("RGB", (tile * 4, (tile + heading) * 2), "#edf2f6")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(_font_path(), 16)
    for index, (title, bgr) in enumerate(panels):
        x, y = (index % 4) * tile, (index // 4) * (tile + heading)
        draw.text((x + 7, y + 6), title, font=font, fill="#17364c")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        canvas.paste(Image.fromarray(rgb).resize((tile, tile), Image.Resampling.LANCZOS), (x, y + heading))
    canvas.save(path, quality=91)


def changes(before, after, gt, valid):
    before, after, gt, valid = (mask.astype(bool) for mask in (before, after, gt, valid))
    added = after & ~before & valid
    removed = before & ~after & valid
    return {
        "added_tp": int(np.count_nonzero(added & gt)),
        "added_fp": int(np.count_nonzero(added & ~gt)),
        "removed_tp": int(np.count_nonzero(removed & gt)),
        "removed_fp": int(np.count_nonzero(removed & ~gt)),
    }


def build(output, config_path):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Use a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    config = read_json(config_path)
    if not 0 < config["inner_scale"] < 1 < config["outer_scale"]:
        raise ValueError("Expected inner_scale < 1 < outer_scale")
    skin_params = read_json(PROJECT / "configs/skin_params.json")
    detector = read_json(PROJECT / "configs/original_config.json")
    detector["threshold"]["k"] = config["k"]
    detector["postprocess"]["method"] = "none"
    relaxed_detector = {
        **detector,
        "threshold": {**detector["threshold"], "k": config["reconstruction_k"]},
        "postprocess": {**detector["postprocess"], "method": "none"},
    }
    manifest = _read_csv(PROJECT / "data/split_manifest.csv", {"sample_id", "split", "image_path", "label_path"})
    position_prior = fit_position_prior(manifest, config["position_prior"])
    geometry_prior = fit_eye_face_prior(manifest, config["minimum_geometry_samples"])
    validation = [row for row in manifest if row["split"] in ("val", "validation")]
    source_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()

    with tempfile.TemporaryDirectory(prefix=".dual-ellipse-", dir=output.parent) as temporary:
        stage = Path(temporary) / output.name
        assets = stage / "assets"
        assets.mkdir(parents=True)
        records, metrics = [], {name: [] for name in ("raw", "lab3", "center", "final", "oracle")}
        for index, row in enumerate(validation, 1):
            sample_id = row["sample_id"]
            asset = assets / sample_id
            asset.mkdir()
            bgr = read_bgr(PROJECT / row["image_path"])
            raw = detect(bgr, skin_params, detector)
            relaxed_raw = detect(bgr, skin_params, relaxed_detector)
            lab3 = postprocess(raw, "largest_fill")
            _, regions, relaxed_pairs = locate_eyes(bgr, raw, config, position_prior)
            strict_config = {**config, "min_pair_area_ratio": .30}
            _, _, strict_pairs = locate_eyes(bgr, raw, strict_config, position_prior)
            selected_pair, pair_status = choose_eye_pair(strict_pairs, relaxed_pairs)
            candidate_records, selected, center_candidate = [], None, None
            final = lab3.copy()
            selected_stages = None

            # Everything through final prediction uses image data and training priors only.
            if selected_pair:
                base = ellipse_from_eyes(selected_pair["centers"], geometry_prior)
                internal = []
                for candidate in ellipse_candidates(base, selected_pair["centers"], config["pose_shift_fraction"]):
                    inner = scaled_region(raw.shape, candidate["geometry"], config["inner_scale"])
                    outer = scaled_region(raw.shape, candidate["geometry"], config["outer_scale"])
                    status, stages, descriptors = refine(lab3, inner, outer, config, relaxed_raw)
                    item = {**candidate, "status": status, "descriptors": descriptors}
                    internal.append((item, stages))
                    if candidate["name"] == "center":
                        center_candidate = (item, stages)
                selected_item = select_candidate([item for item, _ in internal], config["center_candidate_penalty"])
                if selected_item is not None:
                    selected, selected_stages = next((item, stages) for item, stages in internal if item is selected_item)
                    final = selected_stages["final"]
                for item, stages in internal:
                    Image.fromarray(stages["final"] * 255).save(asset / f"candidate-{item['name']}.png")
                    candidate_records.append(item)
            else:
                base = None

            # Validation labels enter only after the final mask and candidate choice are fixed.
            label = np.asarray(Image.open(PROJECT / row["label_path"]))
            gt, valid = binary_labels(label)
            row_metrics = {
                "raw": evaluate(raw, gt, valid),
                "lab3": evaluate(lab3, gt, valid),
                "final": evaluate(final, gt, valid),
            }
            if center_candidate and center_candidate[0]["status"] == "ok":
                row_metrics["center"] = evaluate(center_candidate[1]["final"], gt, valid)
            else:
                row_metrics["center"] = row_metrics["lab3"]
            for item in candidate_records:
                mask = np.asarray(Image.open(asset / f"candidate-{item['name']}.png"))
                item["metrics"] = evaluate((mask > 0).astype(np.uint8), gt, valid)
            oracle = max(candidate_records, key=lambda item: item["metrics"]["f1"]) if candidate_records else None
            row_metrics["oracle"] = oracle["metrics"] if oracle else row_metrics["lab3"]
            for name in metrics:
                metrics[name].append(row_metrics[name])

            selected_name = selected["name"] if selected else "lab3_fallback"
            record = {
                "id": sample_id,
                "pair_status": pair_status,
                "eye_pair": selected_pair,
                "base_geometry": base,
                "selected_candidate": selected_name,
                "candidates": candidate_records,
                "oracle_candidate_display_only": oracle["name"] if oracle else "lab3_fallback",
                "low_support_triggered": bool(selected and selected["descriptors"].get("low_support_triggered", False)),
                "reconstruction_pixels": int(selected["descriptors"].get("reconstruction_pixels", 0)) if selected else 0,
                "metrics": row_metrics,
                "changes_from_lab3": changes(lab3, final, gt, valid),
            }
            records.append(record)

            overlay = draw_nested(
                bgr, selected["geometry"] if selected else None,
                config["inner_scale"], config["outer_scale"],
                selected_pair["centers"] if selected_pair else None,
                "LAB3 FALLBACK" if not selected_pair else None,
            )
            selected_raw = selected_stages["selected_raw"] if selected_stages else lab3
            reconstruction = selected_stages.get("reconstruction", np.zeros_like(raw)) if selected_stages else np.zeros_like(raw)
            stage_five = reconstruction if reconstruction.any() else selected_raw
            stage_five_title = "5  低支持肤色重建" if reconstruction.any() else "5  与核心区相交的 Lab3 区域"
            for name, picture in (
                ("nested.jpg", overlay),
                ("errors.jpg", error_panel(bgr, final, gt, valid)),
            ):
                save_bgr(asset / name, picture)
            for name, mask in (("raw", raw), ("relaxed", relaxed_raw), ("lab3", lab3),
                               ("selected_raw", selected_raw), ("reconstruction", reconstruction),
                               ("final", final), ("truth", gt)):
                Image.fromarray(mask * 255).save(asset / f"{name}.png")
            board(asset / "overview.jpg", [
                ("1  双眼与候选方向", draw_nested(bgr, base, 1.0, 1.0,
                                            selected_pair["centers"] if selected_pair else None,
                                            "NO EYE PAIR" if not selected_pair else None)),
                ("2  绿色内椭圆 / 蓝色外椭圆", overlay),
                ("3  原始肤色阈值", mask_panel(raw)),
                ("4  Lab3 最大块填孔", mask_panel(lab3)),
                (stage_five_title, mask_panel(stage_five)),
                ("6  双椭圆形态学结果", mask_panel(final)),
                ("7  人工真值（事后显示）", truth_panel(bgr, gt)),
                ("8  误差：绿TP 红FP 蓝FN", error_panel(bgr, final, gt, valid)),
            ])
            delta = row_metrics["final"]["f1"] - row_metrics["lab3"]["f1"]
            print(f"{index}/{len(validation)} {sample_id}: {pair_status}, {selected_name}, ΔF1={delta:+.4f}", flush=True)

        summary = {
            "images": len(records),
            "pair_statuses": dict(Counter(record["pair_status"] for record in records)),
            "selected_candidates": dict(Counter(record["selected_candidate"] for record in records)),
            "low_support_reconstructions": sum(record["low_support_triggered"] for record in records),
            "metrics": {name: summarize(values) for name, values in metrics.items()},
            "image_comparison": {
                "improved_over_lab3": sum(record["metrics"]["final"]["f1"] > record["metrics"]["lab3"]["f1"] for record in records),
                "declined_from_lab3": sum(record["metrics"]["final"]["f1"] < record["metrics"]["lab3"]["f1"] for record in records),
                "equal_to_lab3": sum(record["metrics"]["final"]["f1"] == record["metrics"]["lab3"]["f1"] for record in records),
                "mean_image_f1_delta": float(np.mean([
                    record["metrics"]["final"]["f1"] - record["metrics"]["lab3"]["f1"] for record in records
                ])),
            },
            "oracle_note": "Per-image best of center/left/right, selected only after labels; diagnostic upper bound, not a valid predictor.",
        }
        write_json_lf(stage / "records.json", records)
        write_json_lf(stage / "summary.json", summary)
        write_json_lf(stage / "settings.json", {
            "experiment": "lab4/dual_ellipse_morph",
            "source_head": source_head,
            "split": "validation",
            "config": config,
            "detector": detector,
            "position_prior": position_prior,
            "eye_to_face_geometry_prior": geometry_prior,
            "selection_policy": "candidate selection uses image-only descriptors; validation labels are evaluation/display only",
            "opencv": cv2.__version__,
        })
        with (stage / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            fields = ["id", "pair_status", "selected_candidate", "oracle_candidate",
                      "low_support_triggered", "reconstruction_pixels", "lab3_f1", "final_f1",
                      "delta_f1", "lab3_precision", "final_precision", "lab3_recall", "final_recall",
                      "added_tp", "added_fp", "removed_tp", "removed_fp"]
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for record in records:
                before, after = record["metrics"]["lab3"], record["metrics"]["final"]
                writer.writerow({
                    "id": record["id"], "pair_status": record["pair_status"],
                    "selected_candidate": record["selected_candidate"],
                    "oracle_candidate": record["oracle_candidate_display_only"],
                    "low_support_triggered": record["low_support_triggered"],
                    "reconstruction_pixels": record["reconstruction_pixels"],
                    "lab3_f1": before["f1"], "final_f1": after["f1"],
                    "delta_f1": after["f1"] - before["f1"],
                    "lab3_precision": before["precision"], "final_precision": after["precision"],
                    "lab3_recall": before["recall"], "final_recall": after["recall"],
                    **record["changes_from_lab3"],
                })
        payload = json.dumps({"records": records, "summary": summary, "config": config},
                             ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
        html = (HERE / "gallery.html").read_text(encoding="utf-8").replace("__DATA__", payload)
        write_text_lf(stage / "index.html", html)
        write_text_lf(stage / "README.md",
            "# 双椭圆约束形态学：验证结果\n\n打开 `index.html` 查看图册。验证标签只用于事后评价。\n",
        )
        source = stage / "source"
        source.mkdir()
        for name in ("build_gallery.py", "dual_ellipse.py", "gallery.html", "config.json", "README.md"):
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
