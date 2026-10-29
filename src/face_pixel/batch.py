"""Evaluate saved predictions, preserving pairing, input hashes and run settings."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw

from .data_io import read_binary_mask, read_dataset_manifest, read_prediction_manifest
from .evaluation import COUNT_NAMES, METRIC_NAMES, evaluate, summarize


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _save_case(path: Path, original: Image.Image, gt, pred, valid) -> None:
    errors = np.zeros((*gt.shape, 3), dtype=np.uint8)
    errors[(pred == 1) & (gt == 1) & valid] = (0, 180, 0)
    errors[(pred == 1) & (gt == 0) & valid] = (255, 0, 0)
    errors[(pred == 0) & (gt == 1) & valid] = (0, 80, 255)
    errors[~valid] = (255, 210, 0)
    tiles = [original, Image.fromarray(gt * 255).convert("RGB"), Image.fromarray(pred * 255).convert("RGB"), Image.fromarray(errors)]
    width = min(480, max(240, original.width))
    height = max(1, round(original.height * width / original.width))
    canvas = Image.new("RGB", (4 * width, height + 30), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (tile, title) in enumerate(zip(tiles, ("Original", "Ground truth", "Prediction", "TP green / FP red / FN blue"))):
        canvas.paste(tile.resize((width, height), Image.Resampling.NEAREST), (index * width, 30))
        draw.text((index * width + 4, 7), title, fill="black")
    canvas.save(path)


def evaluate_predictions(
    manifest_path,
    prediction_manifest_path,
    output_dir,
    *,
    split: str,
    config_id: str,
    run_config_path=None,
    gt_encoding: str = "binary01",
    pred_encoding: str = "binary01",
    valid_encoding: str = "binary01",
) -> dict:
    """Evaluate exactly one validation/test prediction set, refusing partial output.

    B supplies a prediction manifest for the selected split. C can use the same
    entry point after B freezes the detector. This function does not select k or
    run a detector; all scores describe the supplied saved masks.
    """
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if not config_id.strip():
        raise ValueError("config_id is required")
    manifest_path = Path(manifest_path).resolve()
    prediction_manifest_path = Path(prediction_manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}; choose a new run directory")
    records = read_dataset_manifest(manifest_path)
    selected = [row for row in records if row["split"] == split]
    if not selected:
        raise ValueError(f"no images in split {split!r}")
    predictions = read_prediction_manifest(prediction_manifest_path)
    expected = {row["sample_id"] for row in selected}
    if set(predictions) != expected:
        raise ValueError(f"prediction IDs mismatch; missing={sorted(expected - set(predictions))}, extra={sorted(set(predictions) - expected)}")

    # Detect exact file copies even when A's manifest uses different names.
    seen_hashes, image_hashes = {}, {}
    for row in records:
        digest = _sha256(row["image_path"])
        if digest in seen_hashes:
            raise ValueError(f"duplicate image content: {seen_hashes[digest]!r} and {row['sample_id']!r}; A must review the split")
        seen_hashes[digest] = row["sample_id"]
        image_hashes[row["sample_id"]] = digest

    config = None
    config_sha = None
    if run_config_path is not None:
        run_config_path = Path(run_config_path).resolve()
        config = json.loads(run_config_path.read_text(encoding="utf-8-sig"))
        if not isinstance(config, dict) or config.get("config_id") != config_id:
            raise ValueError("run config must be a JSON object with matching config_id")
        config_sha = _sha256(run_config_path)

    rows, input_hashes = [], []
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".evaluation-", dir=output_dir.parent) as scratch:
        stage = Path(scratch) / "run"
        (stage / "cases").mkdir(parents=True)
        for index, record in enumerate(selected):
            sample_id = record["sample_id"]
            gt = read_binary_mask(record["mask_path"], gt_encoding)
            pred = read_binary_mask(predictions[sample_id], pred_encoding)
            valid_path = record.get("valid_mask_path")
            valid = read_binary_mask(valid_path, valid_encoding) if valid_path else np.ones(gt.shape, dtype=np.uint8)
            metrics = evaluate(pred, gt, valid)
            with Image.open(record["image_path"]) as image:
                original = image.convert("RGB")
            if (original.height, original.width) != gt.shape:
                raise ValueError(f"{sample_id}: original image and masks have different dimensions")
            case_path = f"cases/{index:04d}.png"
            _save_case(stage / case_path, original, gt, pred, valid.astype(bool))
            rows.append({"sample_id": sample_id, "split": split, "config_id": config_id, **metrics, "case_path": case_path})
            input_hashes.append({
                "sample_id": sample_id,
                "image_sha256": image_hashes[sample_id],
                "gt_sha256": _sha256(record["mask_path"]),
                "pred_sha256": _sha256(predictions[sample_id]),
                "valid_sha256": _sha256(valid_path) if valid_path else None,
            })

        summary = summarize(rows)
        summary["split"] = split
        summary["config_id"] = config_id
        summary["metric_policy"] = {
            "positive_label": 1,
            "undefined": "null; excluded from image means with denominator reported",
            "global": "sum tp/fp/tn/fn over valid pixels before calculating metrics",
            "image_mean": "unweighted mean over images with a defined metric",
            "all_ignored": "error",
        }
        summary["provenance"] = {
            "kind": "saved_prediction_evaluation",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_manifest_sha256": _sha256(manifest_path),
            "prediction_manifest_sha256": _sha256(prediction_manifest_path),
            "run_config_sha256": config_sha,
            "configuration_recorded": config is not None,
            "synthetic_demo": bool(config and config.get("synthetic_demo") is True),
            "gt_encoding": gt_encoding,
            "pred_encoding": pred_encoding,
            "valid_encoding": valid_encoding,
            "inputs": input_hashes,
        }
        _write_json(stage / "summary.json", summary)
        if config is not None:
            (stage / "run_config.json").write_bytes(run_config_path.read_bytes())
        with (stage / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", "config_id", *COUNT_NAMES, *METRIC_NAMES, "case_path"])
            writer.writeheader()
            writer.writerows(rows)
        (stage / "README.md").write_text(
            "# Saved prediction evaluation\n\n"
            "Scores apply to the supplied prediction masks. Check run_config.json and the input hashes before using them in a report.\n\n"
            "Undefined metrics are JSON null / empty CSV cells. Image means exclude these values.\n\n"
            "Case panels: original, ground truth, prediction, errors. Green=TP, red=FP, blue=FN, black=TN, yellow=ignored. Display scaling does not affect evaluation.\n"
            + ("\nSYNTHETIC DEMO ONLY. These are hand-made masks, not project performance results.\n" if summary["provenance"]["synthetic_demo"] else ""),
            encoding="utf-8",
        )
        stage.rename(output_dir)
    return summary
