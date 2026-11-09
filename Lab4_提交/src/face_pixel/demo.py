"""Reproducible hand-made fixtures; never a real segmentation experiment."""

import argparse
import csv
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from .batch import evaluate_predictions


def create_demo(root) -> tuple[Path, Path, Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cases = [
        ("mixed", [[1, 1, 0], [0, 0, 0]], [[1, 0, 1], [0, 0, 0]]),
        ("perfect", [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        ("no_skin", [[0, 0], [0, 0]], [[0, 0], [0, 0]]),
    ]
    dataset_rows, prediction_rows = [], []
    for index, (sample_id, gt_values, pred_values) in enumerate(cases):
        gt, pred = np.array(gt_values, dtype=np.uint8), np.array(pred_values, dtype=np.uint8)
        rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)
        rgb[gt == 1] = (210, 150, 110)
        rgb[gt == 0] = (20 + index * 10, 80, 140)
        Image.fromarray(rgb).save(root / f"{sample_id}_image.png")
        Image.fromarray(gt).save(root / f"{sample_id}_gt.png")
        Image.fromarray(pred).save(root / f"{sample_id}_pred.png")
        dataset_rows.append({"sample_id": sample_id, "image_path": f"{sample_id}_image.png", "mask_path": f"{sample_id}_gt.png", "split": "test"})
        prediction_rows.append({"sample_id": sample_id, "pred_path": f"{sample_id}_pred.png"})
    for name, rows in (("split_manifest.csv", dataset_rows), ("predictions.csv", prediction_rows)):
        with (root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    config = root / "demo_config.json"
    config.write_text(json.dumps({"config_id": "synthetic-demo-v1", "synthetic_demo": True, "purpose": "Hand-made fixture verification only; not experimental performance."}, indent=2), encoding="utf-8")
    return root / "split_manifest.csv", root / "predictions.csv", config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run synthetic hand-mask examples, not a real experiment.")
    parser.add_argument("--output", default="results/synthetic-demo", help="A new output directory")
    args = parser.parse_args(argv)
    try:
        with tempfile.TemporaryDirectory(prefix="face-pixel-demo-") as temporary:
            manifest, predictions, config = create_demo(temporary)
            summary = evaluate_predictions(manifest, predictions, args.output, split="test", config_id="synthetic-demo-v1", run_config_path=config)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Demo failed: {exc}\n")
    expected = {"tp": 3, "fp": 1, "tn": 9, "fn": 1, "n_valid": 14}
    if any(summary["global"][key] != value for key, value in expected.items()):
        raise RuntimeError("hand-calculated fixture counts did not match")
    print(f"SYNTHETIC DEMO verified: {expected}. Output: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
