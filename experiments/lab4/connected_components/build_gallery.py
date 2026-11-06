"""Compare the effect of selecting the largest component before filling holes."""

import argparse
import csv
import json
from pathlib import Path
import re
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from face_pixel.data_io import PROJECT, _read_csv, read_bgr, read_json, write_json
from face_pixel.dataset import binary_labels
from face_pixel.detection import detect, fill_holes, postprocess, validate_model
from face_pixel.evaluation import evaluate, summarize
from face_pixel.visualization import _font_path


NAMES = ("raw", "largest", "fill_all", "largest_fill")
SPLITS = {"validation": "验证集", "test": "原测试集", "train": "训练集"}
FACE = (244, 135, 49)
NONFACE = (45, 191, 203)


def largest_component(raw):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    if count <= 1:
        return raw.copy()
    selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == selected).astype(np.uint8)


def difference(change, gt, valid):
    color = np.zeros((*gt.shape, 3), dtype=np.uint8)
    color[change & valid & gt] = FACE
    color[change & valid & ~gt] = NONFACE
    color[change & ~valid] = (160, 160, 160)
    return Image.fromarray(color)


def save_board(path, image, gt, valid, masks, removed, regained, font):
    tile, gap, heading, caption = 384, 14, 39, 33
    width = tile * 4 + gap * 5
    row_height = heading + caption + tile + gap
    board = Image.new("RGB", (width, 2 * row_height + gap), "#ffffff")
    draw = ImageDraw.Draw(board)
    mono = lambda mask: Image.fromarray(mask.astype(np.uint8) * 255)
    rows = (
        ("① 最大连通域删掉了什么？", (
            ("原图", image),
            ("阈值分割", mono(masks["raw"])),
            ("只保留最大连通域", mono(masks["largest"])),
            ("被删掉的像素", difference(removed, gt, valid)),
        )),
        ("② 同样填孔，跳过最大连通域会怎样？", (
            ("脸部真值", mono(gt)),
            ("保留最大连通域 → 填孔", mono(masks["largest_fill"])),
            ("跳过最大连通域 → 填孔", mono(masks["fill_all"])),
            ("跳过后多保留的像素", difference(regained, gt, valid)),
        )),
    )
    for i, (title, panels) in enumerate(rows):
        y = gap + i * row_height
        draw.text((gap, y), title, font=font, fill="#15433f")
        for j, (label, picture) in enumerate(panels):
            x = gap + j * (tile + gap)
            draw.text((x, y + heading), label, font=font, fill="#29454a")
            resample = Image.Resampling.LANCZOS if picture is image else Image.Resampling.NEAREST
            picture = picture.convert("RGB")
            picture.thumbnail((tile, tile), resample)
            board.paste(picture, (x, y + heading + caption))
    board.save(path)


def build(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    params = read_json(PROJECT / "configs/skin_params.json")
    validate_model(params)
    reference = read_json(PROJECT / "configs/original_config.json")
    if reference["threshold"]["k"] != 1.25 or reference["postprocess"]["method"] != "largest_fill":
        raise ValueError("Expected the lab3 reference: k=1.25 and largest_fill")
    config = {**reference, "postprocess": {**reference["postprocess"], "method": "none"}}
    manifest = PROJECT / "data/split_manifest.csv"
    rows = _read_csv(manifest, {"sample_id", "image_path", "label_path", "split"})
    font = ImageFont.truetype(_font_path(), 22)
    records = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".components-", dir=output.parent) as tmp:
        stage = Path(tmp) / "gallery"
        (stage / "images").mkdir(parents=True)
        for index, row in enumerate(rows):
            sid = row["sample_id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", sid):
                raise ValueError(f"Invalid sample ID: {sid}")
            split = "validation" if row["split"] == "val" else row["split"]
            if split not in SPLITS:
                raise ValueError(f"Invalid split: {split}")
            bgr = read_bgr(PROJECT / row["image_path"])
            with Image.open(PROJECT / row["label_path"]) as label:
                gt, valid = (m.astype(bool) for m in binary_labels(np.asarray(label)))
            raw = detect(bgr, params, config)
            largest = largest_component(raw)
            fill_all = postprocess(raw, "fill")
            largest_fill = postprocess(raw, "largest_fill")
            if not np.array_equal(fill_holes(largest), largest_fill) or np.any(largest_fill > fill_all):
                raise ValueError(f"Unexpected component/fill relationship: {sid}")
            removed = (raw > largest)
            regained = (fill_all > largest_fill)
            masks = dict(zip(NAMES, (raw, largest, fill_all, largest_fill)))
            metrics = {name: evaluate(mask, gt, valid) for name, mask in masks.items()}
            a, b, c, d = (metrics[n] for n in NAMES)
            records.append({
                "id": sid, "split": split, "metrics": metrics,
                "removed_tp": a["tp"] - b["tp"], "removed_fp": a["fp"] - b["fp"],
                "extra_tp": c["tp"] - d["tp"], "extra_fp": c["fp"] - d["fp"],
                "delta_f1": c["f1"] - d["f1"] if c["f1"] is not None and d["f1"] is not None else None,
            })
            save_board(stage / "images" / f"{sid}.png", Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
                       gt, valid, masks, removed, regained, font)
            if (index + 1) % 32 == 0 or index + 1 == len(rows):
                print(f"Rendered {index + 1}/{len(rows)}", flush=True)
        summary = {
            split: {name: summarize(r["metrics"][name] for r in records if r["split"] == split)
                    for name in NAMES}
            for split in SPLITS
        }
        payload = {"records": records, "summary": summary, "splits": SPLITS}
        template = Path(__file__).with_name("gallery.html").read_text(encoding="utf-8")
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
        (stage / "index.html").write_text(template.replace("__DATA__", data), encoding="utf-8")
        write_json(stage / "summary.json", summary)
        write_json(stage / "settings.json", {"threshold_config": config, "skin_params": params,
                   "data_manifest": manifest.relative_to(PROJECT).as_posix(), "opencv": cv2.__version__})
        with (stage / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            flattened = [{**{key: val for key, val in r.items() if key != "metrics"},
                          **{f"{name}_{k}": v for name, scores in r["metrics"].items() for k, v in scores.items()}}
                         for r in records]
            writer = csv.DictWriter(stream, fieldnames=list(flattened[0]))
            writer.writeheader()
            writer.writerows(flattened)
        (stage / "README.md").write_text(Path(__file__).with_name("README.md").read_text(encoding="utf-8"),
                                         encoding="utf-8")
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New directory for the offline gallery")
    build(parser.parse_args().output)
