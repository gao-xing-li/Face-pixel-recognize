import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from subprocess import CompletedProcess

import numpy as np
from PIL import Image

from face_pixel.batch import evaluate_predictions
from face_pixel.data_io import read_binary_mask, read_dataset_manifest
from face_pixel.part_a import derive_face_masks, label_mapping, prepare_part_a


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(rows):
    values = sorted((row["sample_id"], row["image_sha256"], row["label_sha256"]) for row in rows)
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


class PartAAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "part-a"
        self.output = self.root / "normalized"
        (self.source / "data/part_a").mkdir(parents=True)
        (self.source / "data/raw").mkdir()
        (self.source / "configs").mkdir()
        (self.source / "src/face_pixel").mkdir(parents=True)
        (self.source / "src/face_pixel/modeling.py").write_text("# Synthetic fixture only\n", encoding="utf-8")
        labels = [
            np.array([[1, 2, 3, 0], [4, 8, 17, 10]], dtype=np.uint8),
            np.array([[1, 3, 0, 8], [2, 4, 11, 17]], dtype=np.uint8),
            np.array([[0, 1, 2, 3], [6, 7, 12, 13]], dtype=np.uint8),
        ]
        self.rows = []
        for index, (split, label) in enumerate(zip(("train", "val", "test"), labels)):
            image_path, label_path = f"data/raw/{index}_image.png", f"data/raw/{index}_label.png"
            Image.new("RGB", (4, 2), (40 * index, 80, 120)).save(self.source / image_path)
            Image.fromarray(label).save(self.source / label_path)
            self.rows.append({"sample_id": f"fixture_{index}", "image_path": image_path, "label_path": label_path, "split": split, "width": "4", "height": "2", "label_ids_present": " ".join(str(value) for value in np.unique(label)), "image_sha256": digest(self.source / image_path), "label_sha256": digest(self.source / label_path)})
        self.metadata = {
            "sample_count": 3,
            "skin_ids": [1, 2],
            "face_positive_ids": [1, 2, 4, 5, 6, 7, 10, 11, 12],
            "face_negative_ids": [0, 8, 9, 13, 14, 15, 16, 17, 18],
            "evaluation_ignore_ids": [3],
            "label_map": {str(index): f"fixture-class-{index}" for index in range(19)},
            "observed_label_ids": sorted(set(int(value) for label in labels for value in np.unique(label))),
            "split": {"counts": {"train": 1, "val": 1, "test": 1}, "human_split_review": "passed"},
            "annotation_quality_review": {},
        }
        self.params = {"fit_split": "train", "skin_label_ids": [1, 2], "training_sample_ids": ["fixture_0"], "training_image_count": 1, "skin_pixel_count": 2, "threshold_k": None, "statistics_script_sha256": digest(self.source / "src/face_pixel/modeling.py")}
        self.refresh()

    def refresh(self):
        manifest = self.source / "data/split_manifest.csv"
        with manifest.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        content = fingerprint(self.rows)
        self.metadata["split"].update(manifest_sha256=digest(manifest), content_fingerprint=content)
        self.metadata["annotation_quality_review"]["content_fingerprint"] = content
        self.params.update(manifest_sha256=digest(manifest), training_content_fingerprint=fingerprint([row for row in self.rows if row["split"] == "train"]))
        for name, data in (("data/part_a/metadata.json", self.metadata), ("data/part_a/split_report.json", self.metadata["split"]), ("configs/skin_params.json", self.params)):
            (self.source / name).write_text(json.dumps(data), encoding="utf-8")

    def test_full_class_mapping_and_ignored_glasses(self):
        mapping = label_mapping(self.metadata)
        gt, valid = derive_face_masks(np.arange(19, dtype=np.uint8).reshape(1, 19), mapping)
        self.assertEqual(np.flatnonzero(gt).tolist(), [1, 2, 4, 5, 6, 7, 10, 11, 12])
        self.assertEqual(np.flatnonzero(valid == 0).tolist(), [3])
        self.assertEqual(int(gt[0, 8]), 0)  # Ear is negative, not simply label > 0.
        self.assertEqual(int(gt[0, 17]), 0)  # Neck is negative.

    def test_unknown_class_and_invalid_mapping_are_rejected(self):
        mapping = label_mapping(self.metadata)
        with self.assertRaisesRegex(ValueError, "unknown class"):
            derive_face_masks(np.array([[19]], dtype=np.uint8), mapping)
        with self.assertRaises(ValueError):
            derive_face_masks([[1.5]], mapping)
        self.metadata["evaluation_ignore_ids"] = [1, 3]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            label_mapping(self.metadata)
        self.metadata["evaluation_ignore_ids"] = [3]
        self.metadata["face_negative_ids"].remove(18)
        with self.assertRaisesRegex(ValueError, "complete"):
            label_mapping(self.metadata)

    def test_prepare_preserves_source_and_emits_evaluator_format(self):
        before = {str(path.relative_to(self.source)): digest(path) for path in self.source.rglob("*") if path.is_file()}
        report = prepare_part_a(self.source, self.output)
        after = {str(path.relative_to(self.source)): digest(path) for path in self.source.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(report["source_file_hashes_verified"], 6)
        self.assertEqual(report["training_skin_pixels_verified"], 2)
        self.assertEqual(report["validation_wiring_check"]["images"], 1)
        self.assertFalse(report["validation_wiring_check"]["is_detector_evaluation"])
        rows = read_dataset_manifest(self.output / "split_manifest.csv")
        self.assertEqual([row["split"] for row in rows], ["train", "validation", "test"])
        self.assertEqual(rows[1]["part_a_split"], "val")
        self.assertEqual([row["sample_id"] for row in rows], [row["sample_id"] for row in self.rows])
        gt, valid = read_binary_mask(rows[1]["mask_path"]), read_binary_mask(rows[1]["valid_mask_path"])
        self.assertEqual(gt.tolist(), [[1, 0, 0, 0], [1, 1, 1, 0]])
        self.assertEqual(valid.tolist(), [[1, 0, 1, 1], [1, 1, 1, 1]])
        self.assertEqual(report["splits"]["validation"]["ignored_pixels"], 1)
        self.assertEqual(json.loads((self.output / "audit.json").read_text(encoding="utf-8")), report)

    def test_normalized_masks_work_with_existing_batch_entry(self):
        prepare_part_a(self.source, self.output)
        pred = self.root / "background.png"
        Image.fromarray(np.zeros((2, 4), dtype=np.uint8)).save(pred)
        predictions = self.root / "predictions.csv"
        predictions.write_text("sample_id,pred_path\nfixture_1,background.png\n", encoding="utf-8")
        summary = evaluate_predictions(self.output / "split_manifest.csv", predictions, self.root / "evaluation", split="validation", config_id="constant-mask-plumbing-only")
        self.assertEqual([summary["global"][name] for name in ("tp", "fp", "tn", "fn", "n_valid")], [0, 0, 3, 4, 7])

    def test_raw_hash_mismatch_fails_without_partial_output(self):
        (self.source / self.rows[1]["image_path"]).write_bytes(b"damaged")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            prepare_part_a(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".part-a-*")))

    def test_manifest_or_review_mismatch_is_rejected(self):
        with (self.source / "data/split_manifest.csv").open("ab") as stream:
            stream.write(b"\r\n")
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            prepare_part_a(self.source, self.output)
        self.refresh()
        self.metadata["annotation_quality_review"]["content_fingerprint"] = "wrong"
        (self.source / "data/part_a/metadata.json").write_text(json.dumps(self.metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            prepare_part_a(self.source, self.output)

    def test_training_pixel_or_id_mismatch_is_rejected(self):
        self.params["skin_pixel_count"] = 3
        self.refresh()
        with self.assertRaisesRegex(ValueError, "skin-pixel count"):
            prepare_part_a(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.params["training_sample_ids"] = ["fixture_1"]
        self.refresh()
        with self.assertRaisesRegex(ValueError, "training IDs"):
            prepare_part_a(self.source, self.output)

    def test_source_paths_cannot_escape_root(self):
        self.rows[0]["image_path"] = "../outside.png"
        self.refresh()
        with self.assertRaisesRegex(ValueError, "escapes"):
            prepare_part_a(self.source, self.output)

    def test_existing_output_and_bad_revision_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "40-character"):
            prepare_part_a(self.source, self.output, source_revision="main")
        revision = "a" * 40
        with patch("face_pixel.part_a.subprocess.run", return_value=CompletedProcess([], 0, stdout="b" * 40)):
            with self.assertRaisesRegex(ValueError, "does not match"):
                prepare_part_a(self.source, self.output, source_revision=revision)
        with patch("face_pixel.part_a.subprocess.run", side_effect=[CompletedProcess([], 0, stdout=revision), CompletedProcess([], 1)]):
            with self.assertRaisesRegex(ValueError, "tracked modifications"):
                prepare_part_a(self.source, self.output, source_revision=revision)
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare_part_a(self.source, self.output)


if __name__ == "__main__":
    unittest.main()
