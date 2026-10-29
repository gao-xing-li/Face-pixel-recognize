import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from face_pixel.b_data import PROJECT, binary_labels, read_json, write_json
from face_pixel.detection import detect, fill_holes, postprocess, validate_model
from face_pixel.validation import settings, validate


class DetectionTests(unittest.TestCase):
    def test_channel_order(self):
        params = dict(mean_cb=85, std_cb=0, mean_cr=255, std_cr=0)
        colors = np.array([[[0, 0, 255], [255, 0, 0], [100, 100, 100]]], dtype=np.uint8)
        np.testing.assert_array_equal(detect(colors, params, settings(1, "none", 5)), [[1, 0, 0]])

    def test_boundary_and_no_unsigned_wrap(self):
        params = dict(mean_cb=127, std_cb=1, mean_cr=129, std_cr=1)
        gray = np.full((1, 1, 3), 100, dtype=np.uint8)
        self.assertEqual(detect(gray, params, settings(1, "none", 5)).item(), 1)
        self.assertEqual(detect(gray, params, settings(0.5, "none", 5)).item(), 0)

    def test_hole_vs_exterior_and_border(self):
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[1:6, 1:6] = 1
        mask[3, 3] = 0
        filled = fill_holes(mask)
        self.assertEqual(int(filled.sum()), 25)
        self.assertEqual(filled[0, 0], 0)
        self.assertTrue(fill_holes(np.ones((3, 3), dtype=np.uint8)).all())
        self.assertFalse(fill_holes(np.zeros((3, 3), dtype=np.uint8)).any())

    def test_largest_component_removes_disconnected_noise(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[1:5, 1:5] = 1
        mask[2, 2] = 0
        mask[8, 8] = 1
        result = postprocess(mask, "largest_fill")
        self.assertEqual(int(result.sum()), 16)
        self.assertEqual(result[8, 8], 0)

    def test_face_mapping_and_glasses(self):
        gt, valid = binary_labels(np.arange(19, dtype=np.uint8).reshape(1, 19))
        self.assertEqual(np.flatnonzero(gt).tolist(), [1, 2, 4, 5, 6, 7, 10, 11, 12])
        self.assertEqual(np.flatnonzero(valid == 0).tolist(), [3])
        for raw in (np.array([[19]]), np.array([[1.0]]), np.zeros((2, 2, 3), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                binary_labels(raw)

    def test_detector_cannot_receive_ground_truth(self):
        self.assertEqual(list(inspect.signature(detect).parameters), ["image_bgr", "params", "config"])

    def test_invalid_metadata_and_configuration(self):
        params = read_json(PROJECT / "configs/skin_params.json")
        validate_model(params)
        params["conversion_channel_order"] = ["Y", "Cb", "Cr"]
        with self.assertRaises(ValueError):
            validate_model(params)
        for k in (float("nan"), 0, -1, True):
            with self.assertRaises(ValueError):
                detect(np.zeros((1, 1, 3), dtype=np.uint8), params, settings(k, "none", 5))

    def test_validation_never_loads_test_labels_and_tie_breaks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.csv"
            manifest.write_text("placeholder")
            write_json(root / "adapter.json", {})
            plan = {"split": "validation", "metric": "global.f1", "k_values": [2, 1],
                    "postprocess_values": ["none", "fill"], "closing_kernel": 5, "tie_break": "simpler then smaller"}
            write_json(root / "plan.json", plan)
            params_path = PROJECT / "configs/skin_params.json"
            params = read_json(params_path)
            adapter = {"source_manifest_sha256": params["manifest_sha256"], "positive_ids": [1, 2], "ignore_ids": [3]}
            rows = [{"sample_id": "v", "split": "validation", "image_path": "val-image", "mask_path": "val-mask", "valid_mask_path": "val-valid"},
                    {"sample_id": "t", "split": "test", "image_path": "test-image", "mask_path": "test-mask"}]
            def load_mask(path, encoding):
                self.assertTrue(path.startswith("val-"))
                return np.ones((2, 2), dtype=np.uint8)
            def load_image(path):
                self.assertEqual(path, "val-image")
                return np.zeros((2, 2, 3), dtype=np.uint8)
            with patch("face_pixel.validation.verify_inputs", return_value=(params, adapter)), \
                 patch("face_pixel.validation.read_dataset_manifest", return_value=rows), \
                 patch("face_pixel.validation._check_selected"), \
                 patch("face_pixel.validation.read_bgr", side_effect=load_image), \
                 patch("face_pixel.validation.read_binary_mask", side_effect=load_mask), \
                 patch("face_pixel.validation.subprocess.check_output", side_effect=["revision\n", ""]), \
                 patch("face_pixel.validation.plot_curve"):
                frozen = validate(manifest, params_path, root / "plan.json", root / "run")
            self.assertEqual(frozen["final"]["threshold"]["k"], 1)
            self.assertEqual(frozen["final"]["postprocess"]["method"], "none")
            self.assertFalse(frozen["final"]["selection"]["test_used"])


if __name__ == "__main__":
    unittest.main()
