import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from face_pixel.data_io import PROJECT, read_json, write_json
from face_pixel.dataset import binary_labels
from face_pixel.detection import detect, fill_holes, postprocess, validate_model
from face_pixel.validation import settings, validate, select_candidates


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

    def test_morphology_seals_a_gap_that_plain_fill_cannot_fill(self):
        mask = np.zeros((35, 35), dtype=np.uint8)
        mask[5:29, 5:29] = 1
        mask[9:25, 9:25] = 0
        mask[5:10, 15:18] = 0  # Interior is connected to exterior through a slit.
        self.assertEqual(postprocess(mask, "largest_fill")[17, 17], 0)
        for method in ("largest_close_fill", "largest_dilate_fill_erode"):
            with self.subTest(method=method):
                result = postprocess(mask, method, 9)
                self.assertEqual(result[17, 17], 1)
                self.assertEqual(result[0, 0], 0)
                self.assertEqual(result.dtype, np.uint8)
                self.assertEqual(result.shape, mask.shape)

    def test_new_morphology_preserves_edge_touching_objects(self):
        mask = np.zeros((35, 35), dtype=np.uint8)
        mask[:20, :20] = 1
        for method in ("largest_close_fill", "largest_dilate_fill_erode"):
            result = postprocess(mask, method, 9)
            self.assertEqual(result[0, 0], 1)
            self.assertEqual(result[-1, -1], 0)

    def test_morphology_discards_islands_before_expanding(self):
        mask = np.zeros((41, 41), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        mask[32, 32] = 1
        for method in ("largest_close_fill", "largest_dilate_fill_erode"):
            result = postprocess(mask, method, 9)
            self.assertEqual(result[32, 32], 0)
            self.assertEqual(result[20, 20], 1)

    def test_morphology_empty_full_masks_and_kernel_one(self):
        for method in ("largest_close_fill", "largest_dilate_fill_erode"):
            for value in (0, 1):
                mask = np.full((11, 13), value, dtype=np.uint8)
                np.testing.assert_array_equal(postprocess(mask, method, 9), mask)
            mask = np.zeros((21, 21), dtype=np.uint8)
            mask[3:17, 3:17] = 1
            mask[8:10, 8:10] = 0
            np.testing.assert_array_equal(postprocess(mask, method, 1), postprocess(mask, "largest_fill"))

    def test_invalid_morphology_kernel(self):
        for value in (0, 2, -3, True, 5.0):
            with self.assertRaises(ValueError):
                postprocess(np.zeros((5, 5), dtype=np.uint8), "largest_close_fill", value)

    def test_completion_selection_respects_f1_budget(self):
        plan = {"metric": "global.recall_with_f1_budget", "max_f1_drop": 0.01,
                "postprocess_values": ["none", "largest_fill", "largest_close_fill"]}
        rows = [dict(candidate="raw", postprocess="none", kernel=5, k=1.25, f1=0.6, recall=0.8),
                dict(candidate="old", postprocess="largest_fill", kernel=5, k=1.25, f1=0.73, recall=0.83),
                dict(candidate="within", postprocess="largest_close_fill", kernel=21, k=1.25, f1=0.725, recall=0.86),
                dict(candidate="outside", postprocess="largest_close_fill", kernel=41, k=1.25, f1=0.71, recall=0.90)]
        _, chosen, policy = select_candidates(rows, plan)
        self.assertEqual(chosen["candidate"], "within")
        self.assertEqual(policy["best_f1_candidate"], "old")
        self.assertAlmostEqual(policy["minimum_f1"], 0.72)
        plan["metric"] = "global.f1"
        _, chosen, _ = select_candidates(rows, plan)
        self.assertEqual(chosen["candidate"], "old")

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

            rows = [{"sample_id": "v", "split": "validation", "image_path": "val-image", "mask_path": "val-mask", "valid_mask_path": "val-valid"},
                    {"sample_id": "t", "split": "test", "image_path": "test-image", "mask_path": "test-mask"}]
            def load_mask(path, encoding="binary01"):
                self.assertTrue(path.startswith("val-"))
                return np.ones((2, 2), dtype=np.uint8)
            def load_image(path):
                self.assertEqual(path, "val-image")
                return np.zeros((2, 2, 3), dtype=np.uint8)
            with patch("face_pixel.validation.read_dataset_manifest", return_value=rows), \
                 patch("face_pixel.validation.read_bgr", side_effect=load_image), \
                 patch("face_pixel.validation.read_binary_mask", side_effect=load_mask):
                frozen = validate(manifest, params_path, root / "plan.json", root / "run")
            self.assertEqual(frozen["final"]["threshold"]["k"], 1)
            self.assertEqual(frozen["final"]["postprocess"]["method"], "none")
            self.assertFalse(frozen["final"]["selection"]["test_used"])


if __name__ == "__main__":
    unittest.main()
