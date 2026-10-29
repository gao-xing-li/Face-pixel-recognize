"""Independently verify saved histograms/parameters using direct RGB pixel arrays."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import _bootstrap
from face_pixel.part_a_common import PROJECT, DATA, RAW, MANIFEST, content_fingerprint, digest


def main():
    manifest = MANIFEST
    output = PROJECT / "results/skin_statistics"
    params_path = PROJECT / "configs/skin_params.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    training = [r for r in all_rows if r["split"] == "train"]
    train_ids = {r["sample_id"] for r in training}
    other_ids = {r["sample_id"] for r in all_rows if r["split"] != "train"}
    assert len(training) == 98 == params["training_image_count"]
    assert set(params["training_sample_ids"]) == train_ids and not train_ids & other_ids
    assert params["manifest_sha256"] == digest(manifest)
    assert params["training_content_fingerprint"] == content_fingerprint(training)
    assert params["statistics_script_sha256"] == digest(PROJECT / "src/face_pixel/modeling.py")
    with (output / "per_image_counts.csv").open(encoding="utf-8-sig", newline="") as stream:
        counts_list = list(csv.DictReader(stream))
    assert len(counts_list) == 98
    counts = {r["sample_id"]: r for r in counts_list}
    assert set(counts) == train_ids
    chunks = []
    for row in training:
        for key, sha in (("image_path", "image_sha256"), ("label_path", "label_sha256")):
            assert digest(PROJECT / row[key]) == row[sha]
        # Independent read path: Pillow RGB -> COLOR_RGB2YCrCb, instead of BGR imdecode.
        with Image.open(PROJECT / row["image_path"]) as im, Image.open(PROJECT / row["label_path"]) as lab:
            rgb = np.asarray(im)
            labels = np.asarray(lab)
        converted = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        selected = converted[np.isin(labels, [1, 2])][:, 1:3]
        chunks.append(selected)
        saved = counts[row["sample_id"]]
        assert saved["split"] == "train"
        assert int(saved["skin_pixels"]) == int(np.count_nonzero(labels == 1))
        assert int(saved["nose_pixels"]) == int(np.count_nonzero(labels == 2))
        assert int(saved["selected_pixels"]) == len(selected)
    pixels = np.concatenate(chunks, axis=0)
    assert len(pixels) == params["skin_pixel_count"]
    assert np.isclose(sum(float(r["overall_pixel_weight"]) for r in counts_list), 1.0)
    differences = {}
    with np.load(output / "histograms.npz") as hist:
        for i, channel in enumerate(("cr", "cb")):
            values = pixels[:, i].astype(np.float64)
            expected = {"mean": float(values.mean()), "variance": float(values.var(ddof=0)),
                        "std": float(values.std(ddof=0))}
            for statistic, value in expected.items():
                difference = abs(value - params[f"{statistic}_{channel}"])
                assert difference < 1e-10, (channel, statistic, difference)
                differences[f"{statistic}_{channel}"] = difference
            assert np.array_equal(hist[channel], np.bincount(values.astype(int), minlength=256))
        independent_joint = np.zeros((256, 256), dtype=np.int64)
        np.add.at(independent_joint, (pixels[:, 0], pixels[:, 1]), 1)
        assert np.array_equal(hist["joint_cr_cb"], independent_joint)
    record = {
        "status": "passed", "training_images": len(training), "selected_pixels": len(pixels),
        "method": "Pillow RGB read, COLOR_RGB2YCrCb, concatenate all selected train pixels, direct NumPy mean/var/std; independent np.add.at joint counts",
        "absolute_parameter_differences": differences, "tolerance": 1e-10,
        "training_ids_match": True, "no_validation_or_test_samples": True,
        "per_image_and_histogram_counts_match": True, "source_hashes_match": True,
        "parameter_file_sha256": digest(params_path), "manifest_sha256": digest(manifest),
    }
    (output / "verification.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
