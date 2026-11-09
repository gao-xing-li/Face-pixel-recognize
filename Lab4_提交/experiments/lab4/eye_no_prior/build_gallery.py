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
from eyes import locate_eyes
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
    rows = [r for r in _read_csv(PROJECT / "data/split_manifest.csv",
                               {"sample_id", "split", "image_path", "label_path"}) if r["split"] in ("val", "validation")]
    (output / "assets").mkdir(parents=True)
    records = []
    for row in rows:
        sid = row["sample_id"]
        asset = output / "assets" / sid
        asset.mkdir()
        bgr = read_bgr(PROJECT / row["image_path"])
        skin = detect(bgr, params, detector)
        gray, regions, pairs = locate_eyes(bgr, skin, config)
        # Annotations enter only after candidate generation AND all ranking have finished.
        label = np.asarray(Image.open(PROJECT / row["label_path"]))
        evaluation = evaluate_after_prediction(regions, pairs, label)
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
            ("原图", bgr), ("灰度图", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)),
            ("黑帽响应（显示放大 3 倍）", response_image),
            ("搜索框与暗部候选", candidate_image), ("自动首选眼睛配对", overlay(bgr, pair=selected)),
            ("红圈：预测；绿十字：标签中心", overlay(bgr, pair=selected, truth=evaluation["label_centers"]))])
        records.append({"id": sid, "regions": regions, "pairs": pairs, "evaluation": evaluation})
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
    write_json(output / "records.json", records)
    write_json(output / "summary.json", summary)
    write_json(output / "settings.json", {"config": config, "detector": detector, "skin_params": params,
               "opencv": cv2.__version__, "split": "validation", "label_map": {"4": "l_eye", "5": "r_eye"}})
    payload = json.dumps({"records": records, "summary": summary, "config": config}, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    (output / "index.html").write_text((HERE / "gallery.html").read_text(encoding="utf-8").replace("__DATA__", payload), encoding="utf-8")
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "candidate_count", "pair_count", "evaluation_status", "selected_error", "top5_best_error"])
        writer.writeheader()
        for r in records:
            writer.writerow({"id": r["id"], "candidate_count": sum(len(g["candidates"]) for g in r["regions"]),
                             "pair_count": len(r["pairs"]), "evaluation_status": r["evaluation"]["status"],
                             "selected_error": r["evaluation"].get("selected_error"), "top5_best_error": r["evaluation"].get("top5_best_error")})
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
