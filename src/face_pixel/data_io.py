"""Manifest and mask readers. All relative paths use the CSV's parent directory."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image


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
