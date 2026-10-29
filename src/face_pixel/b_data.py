"""Lossless adapter from A's approved manifest to the common C interface."""

from collections import Counter
import csv
import json
import os
from pathlib import Path
import re
import tempfile

import cv2
import numpy as np
from PIL import Image

from .part_a_common import PROJECT, digest
from .detection import validate_model

POSITIVE_IDS = [1, 2, 4, 5, 6, 7, 10, 11, 12]
IGNORE_IDS = [3]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def binary_labels(raw):
    if raw.ndim != 2 or raw.dtype.kind not in "bui" or not np.isin(raw, range(19)).all():
        raise ValueError("expected 0..18 integer raw labels")
    return np.isin(raw, POSITIVE_IDS).astype(np.uint8), (~np.isin(raw, IGNORE_IDS)).astype(np.uint8)


def read_bgr(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode {path}")
    return image


def prepare(output, root=PROJECT):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    manifest = root / "data/split_manifest.csv"
    params = read_json(root / "configs/skin_params.json")
    metadata = read_json(root / "data/part_a/metadata.json")
    validate_model(params)
    if params["manifest_sha256"] != digest(manifest):
        raise ValueError("A model/manifest hash mismatch")
    if metadata["face_positive_ids"] != POSITIVE_IDS or metadata["evaluation_ignore_ids"] != IGNORE_IDS:
        raise ValueError("A label agreement changed; review adapter")
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        original = list(csv.DictReader(stream))
    if Counter(r["split"] for r in original) != {"train": 98, "val": 32, "test": 32}:
        raise ValueError("approved split changed")
    train_ids = {r["sample_id"] for r in original if r["split"] == "train"}
    if train_ids != set(params["training_sample_ids"]) or len(train_ids) != params["training_image_count"]:
        raise ValueError("training IDs mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, checks, ids, image_hashes = [], [], set(), set()
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".prepare-") as temp:
        stage = Path(temp) / "data"
        (stage / "masks").mkdir(parents=True)
        (stage / "valid").mkdir()
        for row in original:
            sid = row["sample_id"]
            if sid in ids or not re.fullmatch(r"[A-Za-z0-9_-]+", sid):
                raise ValueError("duplicate or unsafe sample ID")
            ids.add(sid)
            paths = {}
            for key, sha in (("image_path", "image_sha256"), ("label_path", "label_sha256")):
                path = (root / row[key]).resolve()
                path.relative_to(root / "data/raw/face_segment")
                if digest(path) != row[sha]:
                    raise ValueError(f"source hash mismatch: {sid}/{key}")
                paths[key] = path
            if row["image_sha256"] in image_hashes:
                raise ValueError("duplicate image content")
            image_hashes.add(row["image_sha256"])
            with Image.open(paths["image_path"]) as im, Image.open(paths["label_path"]) as label:
                if im.mode != "RGB" or label.mode != "L" or im.size != label.size or im.size != (int(row["width"]), int(row["height"])):
                    raise ValueError(f"mode/size mismatch: {sid}")
                raw = np.array(label)
            gt, valid = binary_labels(raw)
            if not valid.any():
                raise ValueError("entirely ignored sample")
            Image.fromarray(gt * 255).save(stage / "masks" / f"{sid}.png")
            Image.fromarray(valid * 255).save(stage / "valid" / f"{sid}.png")
            rows.append({"sample_id": sid, "image_path": os.path.relpath(paths["image_path"], output),
                         "mask_path": f"masks/{sid}.png", "split": "validation" if row["split"] == "val" else row["split"],
                         "valid_mask_path": f"valid/{sid}.png"})
            checks.append({"sample_id": sid, "source_split": row["split"], "split": rows[-1]["split"],
                           "image_sha256": row["image_sha256"], "raw_label_sha256": row["label_sha256"],
                           "mask_sha256": digest(stage / rows[-1]["mask_path"]),
                           "valid_sha256": digest(stage / rows[-1]["valid_mask_path"])})
        write_csv(stage / "manifest.csv", rows)
        report = {"source_manifest_sha256": digest(manifest), "derived_manifest_sha256": digest(stage / "manifest.csv"),
                  "model_params_sha256": digest(root / "configs/skin_params.json"),
                  "positive_ids": POSITIVE_IDS, "ignore_ids": IGNORE_IDS, "encoding": "binary255",
                  "counts": dict(Counter(r["split"] for r in rows)), "samples": checks,
                  "note": "Derived view only: no sample or assignment changes; no detection scores."}
        write_json(stage / "adapter.json", report)
        stage.rename(output)
    return report


def verify_inputs(manifest, params_path):
    manifest, params_path = Path(manifest), Path(params_path)
    params = read_json(params_path)
    validate_model(params)
    adapter = read_json(manifest.parent / "adapter.json")
    if adapter["derived_manifest_sha256"] != digest(manifest) or adapter["model_params_sha256"] != digest(params_path):
        raise ValueError("derived manifest or statistics changed")
    if adapter["source_manifest_sha256"] != params["manifest_sha256"]:
        raise ValueError("source manifest mismatch")
    return params, adapter
