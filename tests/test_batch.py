import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from face_pixel.batch import evaluate_predictions
from face_pixel.data_io import read_binary_mask, read_dataset_manifest
from face_pixel.demo import create_demo


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest, self.predictions, self.config = create_demo(self.root / "inputs")
        self.output = self.root / "results"

    def run_batch(self, **kwargs):
        return evaluate_predictions(self.manifest, self.predictions, self.output, split="test", config_id="synthetic-demo-v1", run_config_path=self.config, **kwargs)

    def edit_csv(self, path, mutate):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        mutate(rows)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_end_to_end_demo_and_artifacts(self):
        result = self.run_batch()
        self.assertEqual(result["global"], {"tp": 3, "fp": 1, "tn": 9, "fn": 1, "n_valid": 14, "accuracy": 12 / 14, "precision": 0.75, "recall": 0.75, "f1": 0.75, "iou": 0.6})
        self.assertTrue(result["provenance"]["synthetic_demo"])
        self.assertEqual(result["image_mean_n_defined"]["f1"], 2)
        self.assertEqual(len(list((self.output / "cases").glob("*.png"))), 3)
        text = (self.output / "summary.json").read_text(encoding="utf-8")
        self.assertNotIn("NaN", text)
        self.assertEqual(json.loads(text), result)
        self.assertIn("SYNTHETIC DEMO", (self.output / "README.md").read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256((self.output / "run_config.json").read_bytes()).hexdigest(), result["provenance"]["run_config_sha256"])

    def test_masks_require_declared_encoding(self):
        path = self.root / "mask.png"
        Image.fromarray(np.array([[255, 0]], dtype=np.uint8)).save(path)
        with self.assertRaises(ValueError):
            read_binary_mask(path)
        self.assertEqual(read_binary_mask(path, "binary255").tolist(), [[1, 0]])
        Image.fromarray(np.array([[255, 128]], dtype=np.uint8)).save(path)
        with self.assertRaises(ValueError):
            read_binary_mask(path, "binary255")

    def test_duplicate_ids_rejected(self):
        self.edit_csv(self.manifest, lambda rows: rows.append(dict(rows[0])))
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            self.run_batch()
        self.assertFalse(self.output.exists())

    def test_missing_and_extra_predictions_rejected(self):
        self.edit_csv(self.predictions, lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, "prediction IDs mismatch"):
            self.run_batch()
        self.predictions.write_text("sample_id,pred_path\nmixed,mixed_pred.png\nperfect,perfect_pred.png\nno_skin,no_skin_pred.png\nextra,mixed_pred.png\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "extra"):
            self.run_batch()

    def test_same_image_path_across_splits_rejected(self):
        self.edit_csv(self.manifest, lambda rows: rows.append({**rows[0], "sample_id": "leak", "split": "train"}))
        with self.assertRaisesRegex(ValueError, "image path reused"):
            self.run_batch()

    def test_copied_image_with_another_name_is_rejected(self):
        duplicate = self.manifest.parent / "copied.png"
        duplicate.write_bytes((self.manifest.parent / "mixed_image.png").read_bytes())
        self.edit_csv(self.manifest, lambda rows: rows.append({**rows[0], "sample_id": "leak", "image_path": "copied.png", "split": "train"}))
        with self.assertRaisesRegex(ValueError, "duplicate image content"):
            self.run_batch()

    def test_invalid_mask_leaves_no_partial_results(self):
        Image.fromarray(np.array([[255, 0], [0, 255]], dtype=np.uint8)).save(self.manifest.parent / "perfect_pred.png")
        with self.assertRaises(ValueError):
            self.run_batch()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".evaluation-*")))

    def test_image_alignment_is_checked(self):
        Image.new("RGB", (12, 12), "red").save(self.manifest.parent / "mixed_image.png")
        with self.assertRaisesRegex(ValueError, "different dimensions"):
            self.run_batch()
        self.assertFalse(self.output.exists())

    def test_existing_results_are_preserved(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_batch()
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_config_id_mismatch_rejected(self):
        self.config.write_text('{"config_id": "wrong"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "matching config_id"):
            self.run_batch()

    def test_relative_paths_and_explicit_split_names(self):
        self.assertTrue(read_dataset_manifest(self.manifest)[0]["image_path"].is_absolute())
        self.edit_csv(self.manifest, lambda rows: rows[0].update(split="val"))
        with self.assertRaisesRegex(ValueError, "unknown split"):
            read_dataset_manifest(self.manifest)

    def test_cli_succeeds_and_returns_nonzero_for_bad_predictions(self):
        command = [sys.executable, "-m", "face_pixel.cli", "--manifest", str(self.manifest), "--predictions", str(self.predictions), "--split", "test", "--config-id", "synthetic-demo-v1", "--run-config", str(self.config), "--output", str(self.output)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Evaluated 3 images", result.stdout)
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("already exists", result.stderr)


if __name__ == "__main__":
    unittest.main()
