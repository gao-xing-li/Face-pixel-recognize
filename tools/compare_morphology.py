"""Render a frozen before/after comparison; test examples are diagnostic only."""

import argparse
from pathlib import Path

import _bootstrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from face_pixel.b_data import PROJECT, read_json, write_json, read_bgr
from face_pixel.data_io import read_dataset_manifest, read_binary_mask
from face_pixel.detection import detect
from face_pixel.evaluation import evaluate
from face_pixel.validation import code_hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--manifest", default="data/processed/part_b/manifest.csv")
    parser.add_argument("--reference", default="configs/morphology_reference.json")
    parser.add_argument("--config", default="configs/final_config.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("output directory already exists")
    current, reference = read_json(args.config), read_json(args.reference)
    for config in (current, reference):
        if config["source_sha256"] != code_hashes() or not config["selection"]["fixed"]:
            parser.error("requires configurations frozen for the current source snapshot")
    rows = read_dataset_manifest(args.manifest)
    row = next((r for r in rows if r["sample_id"] == args.sample_id), None)
    if row is None:
        parser.error("sample ID not found")
    params = read_json(PROJECT / current["model_params"]["path"])
    image = read_bgr(row["image_path"])
    before, after = (detect(image, params, config) for config in (reference, current))
    # Labels are used for presentation/evaluation only, after both predictions.
    gt = read_binary_mask(row["mask_path"], "binary255")
    valid = read_binary_mask(row["valid_mask_path"], "binary255").astype(bool)
    rgb = image[:, :, ::-1]
    errors = np.zeros_like(rgb)
    errors[(after == 1) & (gt == 1) & valid] = [0, 190, 0]
    errors[(after == 1) & (gt == 0) & valid] = [255, 0, 0]
    errors[(after == 0) & (gt == 1) & valid] = [0, 80, 255]
    errors[~valid] = [255, 210, 0]
    output.mkdir(parents=True)
    figure, axes = plt.subplots(1, 5, figsize=(15, 3.5), layout="constrained")
    for ax, pixels, title in zip(axes, (rgb, gt, before, after, errors),
                                  ("Original", "Face ground truth", "V1: fill only", "V2: morphology + fill", "Green=TP / Red=FP / Blue=FN")):
        ax.imshow(pixels, cmap="gray", vmin=0, vmax=1 if pixels.ndim == 2 else 255)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    scope = "TEST FEEDBACK DIAGNOSTIC - not an independent test" if row["split"] == "test" else "Validation comparison"
    figure.suptitle(f"{args.sample_id} | {scope}", fontsize=11)
    figure.savefig(output / "comparison.png", dpi=140)
    plt.close(figure)
    for name, mask in (("before", before), ("after", after)):
        Image.fromarray(mask * 255).save(output / f"{name}.png")
    write_json(output / "comparison.json", {
        "sample_id": args.sample_id, "split": row["split"], "scope": scope,
        "selected_after_configuration_was_frozen": True,
        "reference_config_id": reference["config_id"], "new_config_id": current["config_id"],
        "before": evaluate(before, gt, valid), "after": evaluate(after, gt, valid),
        "added_foreground_pixels": int(np.count_nonzero((after == 1) & (before == 0))),
        "removed_foreground_pixels": int(np.count_nonzero((after == 0) & (before == 1))),
    })
    print(output / "comparison.png")


if __name__ == "__main__":
    main()
