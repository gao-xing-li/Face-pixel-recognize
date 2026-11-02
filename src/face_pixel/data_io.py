"""Manifest and mask readers. All relative paths use the CSV's parent directory."""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]


def _read_csv(path: Path, required: set[str]) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)) or not required.issubset(headers):
            raise ValueError(f"{path}: expected unique columns including {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: manifest is empty")
    seen = set()
    for number, row in enumerate(rows, start=2):
        if None in row or any(not (row.get(key) or "").strip() for key in required):
            raise ValueError(f"{path}:{number}: missing or malformed required field")
        for key, value in row.items():
            row[key] = value.strip() if value is not None else ""
        sample_id = row["sample_id"]
        if sample_id in seen:
            raise ValueError(f"{path}: duplicate sample_id {sample_id!r}")
        seen.add(sample_id)
    return rows


def _resolve(parent: Path, value: str) -> Path:
    return (parent / value).resolve()


def read_dataset_manifest(path) -> list[dict]:
    """Read A's unique train/validation/test manifest and reject reused image paths."""
    path = Path(path).resolve()
    rows = _read_csv(path, {"sample_id", "image_path", "mask_path", "split"})
    seen_images = {}
    for row in rows:
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError(f"unknown split {row['split']!r}; use train/validation/test")
        for key in ("image_path", "mask_path", "valid_mask_path"):
            if row.get(key):
                row[key] = _resolve(path.parent, row[key])
        image_path = row["image_path"]
        if image_path in seen_images:
            raise ValueError(f"image path reused by {seen_images[image_path]!r} and {row['sample_id']!r}")
        seen_images[image_path] = row["sample_id"]
    return rows


def read_prediction_manifest(path) -> dict[str, Path]:
    path = Path(path).resolve()
    rows = _read_csv(path, {"sample_id", "pred_path"})
    return {row["sample_id"]: _resolve(path.parent, row["pred_path"]) for row in rows}


def read_binary_mask(path, encoding: str = "binary01") -> np.ndarray:
    """Decode explicitly declared 0/1 or 0/255, without color conversion or thresholding."""
    if encoding not in {"binary01", "binary255"}:
        raise ValueError("encoding must be binary01 or binary255")
    with Image.open(path) as image:
        mask = np.array(image)
    high = 1 if encoding == "binary01" else 255
    if mask.ndim != 2 or mask.dtype.kind not in "bui" or not np.isin(mask, (0, high)).all():
        raise ValueError(f"{path}: expected a 2-D {encoding} mask; normalize raw labels explicitly")
    return (mask != 0).astype(np.uint8)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_bgr(path):
    import cv2
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image
