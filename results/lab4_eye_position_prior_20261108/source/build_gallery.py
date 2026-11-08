"""Run the eye-localization experiment and build an offline validation gallery."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from eyes import locate_eyes, rank_pairs
from face_pixel.data_io import PROJECT, _read_csv, read_bgr, read_json, write_json
from face_pixel.detection import detect
from face_pixel.visualization import _font_path


def save_image(path, bgr):
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(path)


def overlay(bgr, region=None, pair=None, truth=None, candidates=False):
    canvas = bgr.copy()
    if region:
        x, y, w, h = region.get("search_box", region["box"])
        cv2.rectangle(canvas, (x, y), (x+w-1, y+h-1), (0, 210, 255), 2)
        if candidates:
            for c in region["candidates"]:
                x, y, w, h = c["box"]
                cv2.rectangle(canvas, (x, y), (x+w-1, y+h-1), (255, 220, 0), 1)
                cv2.putText(canvas, c["id"].split("-")[-1], (x, max(12, y-3)),
                            cv2.FONT_HERSHEY_SIMPLEX, .36, (255, 220, 0), 1, cv2.LINE_AA)
    if truth:
        for point in truth:
            if point is not None:
                cv2.drawMarker(canvas, tuple(np.rint(point).astype(int)), (50, 240, 50), cv2.MARKER_CROSS, 16, 2)
    if pair:
        points = [tuple(np.rint(p).astype(int)) for p in pair["centers"]]
        cv2.line(canvas, *points, (0, 80, 255), 2, cv2.LINE_AA)
        for p in points:
            cv2.circle(canvas, p, 7, (0, 80, 255), 2, cv2.LINE_AA)
    return canvas


def board(path, panels):
    size = 384
    canvas = Image.new("RGB", (size*3, (size+34)*2), "#f0f3f6")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(_font_path(), 17)
    for i, (title, bgr) in enumerate(panels):
        x, y = (i % 3)*size, (i // 3)*(size+34)
        draw.text((x+8, y+6), title, font=font, fill="#152e42")
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((size, size))
        canvas.paste(image, (x, y+34))
    canvas.save(path, quality=90)


def fit_position_prior(rows, settings):
    """Estimate only the global midpoint distribution, using training labels only."""
    midpoints, sample_ids = [], []
    for row in rows:
        if row["split"] != "train":
            continue
        with Image.open(PROJECT / row["label_path"]) as image:
            label = np.asarray(image)
        centers = []
        for label_id in (4, 5):
            yy, xx = np.where(label == label_id)
            if len(xx):
                centers.append([float(xx.mean()), float(yy.mean())])
        if len(centers) == 2:
            midpoints.append(np.mean(centers, axis=0) / [label.shape[1], label.shape[0]])
            sample_ids.append(row["sample_id"])
    if not midpoints:
        raise ValueError("No training images have both eye labels")
    measured_sigma = np.std(midpoints, axis=0, ddof=0)
    return {"mean": np.mean(midpoints, axis=0).tolist(), "measured_sigma": measured_sigma.tolist(),
            "sigma": np.maximum(measured_sigma, settings["minimum_sigma_fraction"]).tolist(),
            "weight": settings["weight"], "training_images": len(midpoints), "training_ids": sample_ids,
            "normalization": "x/image_width, y/image_height; midpoint of label 4 and label 5 centroids"}


def prior_heatmap(bgr, prior):
    h, w = bgr.shape[:2]
    yy, xx = np.indices((h, w))
    dx = (xx/w-prior["mean"][0])/prior["sigma"][0]
    dy = (yy/h-prior["mean"][1])/prior["sigma"][1]
    response = np.exp(-.5*(dx*dx+dy*dy))
    color = cv2.applyColorMap(np.rint(response*255).astype(np.uint8), cv2.COLORMAP_TURBO)
    canvas = cv2.addWeighted(bgr, .55, color, .45, 0)
    center = tuple(np.rint(np.array(prior["mean"])*[w, h]).astype(int))
    axes = tuple(np.rint(np.array(prior["sigma"])*[w, h]).astype(int))
    cv2.ellipse(canvas, center, axes, 0, 0, 360, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.drawMarker(canvas, center, (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
    return canvas


def evaluate_after_prediction(regions, pairs, label):
    points = []
    for label_id in (4, 5):
        yy, xx = np.where(label == label_id)
        points.append([float(xx.mean()), float(yy.mean())] if len(xx) else None)
    if any(p is None for p in points):
        return {"status": "missing_eye_label", "label_centers": points}
    points.sort(key=lambda p: p[0])  # Image-left/image-right, independent of anatomical naming.
    distance = float(np.linalg.norm(np.subtract(*points)))
    if distance <= 0:
        return {"status": "degenerate_eye_labels", "label_centers": points}
    for pair in pairs:
        errors = np.linalg.norm(np.subtract(pair["centers"], points), axis=1)/distance
        pair["normalized_error"] = float(errors.max())
    # Candidate availability diagnoses proposal failure versus ranking failure.
    candidates = [c for r in regions for c in r["candidates"]]
    nearest = [min((float(np.linalg.norm(np.subtract(c["center"], p))/distance) for c in candidates),
                   default=None) for p in points]
    return {"status": "ok", "label_centers": points, "interocular_distance": distance,
            "selected_error": pairs[0]["normalized_error"] if pairs else None,
            "top5_best_error": min((p["normalized_error"] for p in pairs), default=None),
            "nearest_candidate_errors": nearest}


def build(output, config_path):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Use a new output directory: {output}")
    config = read_json(config_path)
    params = read_json(PROJECT / "configs/skin_params.json")
    detector = read_json(PROJECT / "configs/original_config.json")
    detector["threshold"]["k"] = config["k"]
    detector["postprocess"]["method"] = "none"
    manifest = _read_csv(PROJECT / "data/split_manifest.csv", {"sample_id", "split", "image_path", "label_path"})
    prior = fit_position_prior(manifest, config["position_prior"]) if config.get("position_prior") else None
    if prior is not None:
        print(f"Training prior: {prior['training_images']} images, mean={prior['mean']}, sigma={prior['sigma']}, weight={prior['weight']}", flush=True)
    rows = [r for r in manifest if r["split"] in ("val", "validation")]
    (output / "assets").mkdir(parents=True)
    records = []
    for row in rows:
        sid = row["sample_id"]
        asset = output / "assets" / sid
        asset.mkdir()
        bgr = read_bgr(PROJECT / row["image_path"])
        skin = detect(bgr, params, detector)
        gray, regions, pairs = locate_eyes(bgr, skin, config, prior)
        baseline_pairs = rank_pairs(regions, "base_score", config["display_pairs"])
        for region in regions:
            region["pairs"] = sorted(region["pairs"], key=lambda p: -p["score"])[:config["display_pairs"]]
        # This validation image's annotations enter only after both rankings are fixed.
        label = np.asarray(Image.open(PROJECT / row["label_path"]))
        evaluation = evaluate_after_prediction(regions, pairs, label)
        baseline_evaluation = evaluate_after_prediction(regions, baseline_pairs, label)
        save_image(asset / "original.jpg", bgr)
        Image.fromarray(gray).save(asset / "gray.png")
        Image.fromarray(skin*255).save(asset / "skin.png")
        for region in regions:
            if region["status"] != "ok":
                continue
            response, binary = region.pop("response"), region.pop("binary")
            # Fixed gain for display only; thresholds and proposals use the unscaled response.
            shown = np.minimum(response.astype(np.uint16)*3, 255).astype(np.uint8)
            Image.fromarray(shown).save(asset / f"{region['id']}-blackhat.png")
            Image.fromarray(binary*255).save(asset / f"{region['id']}-binary.png")
            save_image(asset / f"{region['id']}-candidates.jpg", overlay(bgr, region, candidates=True))
        selected = pairs[0] if pairs else None
        baseline_selected = baseline_pairs[0] if baseline_pairs else None
        save_image(asset / "baseline.jpg", overlay(bgr, pair=baseline_selected, truth=evaluation["label_centers"]))
        if prior is not None:
            save_image(asset / "prior.jpg", prior_heatmap(bgr, prior))
        save_image(asset / "prediction.jpg", overlay(bgr, pair=selected))
        save_image(asset / "comparison.jpg", overlay(bgr, pair=selected, truth=evaluation["label_centers"]))
        reference = next((r for r in regions if selected and r["id"] == selected["roi_id"]),
                         next((r for r in regions if r["status"] == "ok"), None))
        if reference:
            response_image = cv2.cvtColor(np.asarray(Image.open(asset / f"{reference['id']}-blackhat.png")), cv2.COLOR_GRAY2BGR)
            candidate_image = overlay(bgr, reference, candidates=True)
        else:
            response_image = np.zeros_like(bgr)
            candidate_image = bgr
        board(asset / "overview.jpg", [
            ("加先验前（红圈预测，绿十字标签）", overlay(bgr, pair=baseline_selected, truth=evaluation["label_centers"])),
            ("加先验后（红圈预测，绿十字标签）", overlay(bgr, pair=selected, truth=evaluation["label_centers"])),
            ("双眼中点位置先验（仅辅助排序）", prior_heatmap(bgr, prior) if prior else bgr),
            ("黑帽响应（显示放大 3 倍）", response_image),
            ("搜索框与暗部候选", candidate_image), ("原图", bgr)])
        before_hit = baseline_evaluation.get("selected_error") is not None and baseline_evaluation["selected_error"] <= .25
        after_hit = evaluation.get("selected_error") is not None and evaluation["selected_error"] <= .25
        outcome = ("corrected" if after_hit else "regressed") if before_hit != after_hit else ("both_hit" if after_hit else "both_miss")
        changed = bool(selected and baseline_selected and not np.allclose(selected["centers"], baseline_selected["centers"]))
        records.append({"id": sid, "regions": regions, "pairs": pairs, "evaluation": evaluation,
                        "baseline_pairs": baseline_pairs, "baseline_evaluation": baseline_evaluation,
                        "comparison": {"outcome": outcome if evaluation["status"] == "ok" else "unlabeled", "selection_changed": changed}})
        print(f"{len(records)}/{len(rows)} {sid}: {sum(len(r['candidates']) for r in regions)} blobs, {len(pairs)} pairs", flush=True)
    eligible = [r for r in records if r["evaluation"]["status"] == "ok"]
    summary = {"images": len(records), "two_eye_label_images": len(eligible),
               "missing_eye_label_images": len(records)-len(eligible),
               "images_with_pair": sum(bool(r["pairs"]) for r in records), "localization": {}}
    for tolerance in (.15, .25):
        summary["localization"][str(tolerance)] = {
            "selected_pair_hits": sum(r["evaluation"]["selected_error"] is not None and r["evaluation"]["selected_error"] <= tolerance for r in eligible),
            "top5_contains_hit": sum(r["evaluation"]["top5_best_error"] is not None and r["evaluation"]["top5_best_error"] <= tolerance for r in eligible),
            "both_eye_candidates_available": sum(all(e is not None and e <= tolerance for e in r["evaluation"]["nearest_candidate_errors"]) for r in eligible)}
    errors = [r["evaluation"]["selected_error"] for r in eligible if r["pairs"]]
    summary["median_error_when_pair_exists"] = float(np.median(errors)) if errors else None
    summary["baseline"] = {"localization": {str(t): {
        "selected_pair_hits": sum(r["baseline_evaluation"]["selected_error"] is not None and r["baseline_evaluation"]["selected_error"] <= t for r in eligible),
        "top5_contains_hit": sum(r["baseline_evaluation"]["top5_best_error"] is not None and r["baseline_evaluation"]["top5_best_error"] <= t for r in eligible)} for t in (.15, .25)}}
    summary["comparison"] = {"selection_changed": sum(r["comparison"]["selection_changed"] for r in records),
        **{key: [r["id"] for r in records if r["comparison"]["outcome"] == key] for key in ("corrected", "regressed", "both_hit", "both_miss")}}
    write_json(output / "records.json", records)
    write_json(output / "summary.json", summary)
    write_json(output / "settings.json", {"config": config, "detector": detector, "skin_params": params,
               "position_prior": prior, "opencv": cv2.__version__, "split": "validation", "label_map": {"4": "l_eye", "5": "r_eye"}})
    payload = json.dumps({"records": records, "summary": summary, "config": config, "prior": prior}, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    (output / "index.html").write_text((HERE / "gallery.html").read_text(encoding="utf-8").replace("__DATA__", payload), encoding="utf-8")
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "candidate_count", "pair_count", "evaluation_status", "selected_error", "top5_best_error", "baseline_error", "outcome", "selection_changed"])
        writer.writeheader()
        for r in records:
            writer.writerow({"id": r["id"], "candidate_count": sum(len(g["candidates"]) for g in r["regions"]),
                             "pair_count": len(r["pairs"]), "evaluation_status": r["evaluation"]["status"],
                             "selected_error": r["evaluation"].get("selected_error"), "top5_best_error": r["evaluation"].get("top5_best_error"),
                             "baseline_error": r["baseline_evaluation"].get("selected_error"), **r["comparison"]})
    (output / "source").mkdir()
    for name in ("eyes.py", "build_gallery.py", "gallery.html", "README.md"):
        shutil.copyfile(HERE / name, output / "source" / name)
    shutil.copyfile(config_path, output / "source/config.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(output / "index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(HERE / "config.json"))
    args = parser.parse_args()
    build(args.output, args.config)
