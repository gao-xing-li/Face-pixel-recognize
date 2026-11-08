import sys
from pathlib import Path
import unittest

import cv2
import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments" / "lab4" / "dual_ellipse_morph"
sys.path.insert(0, str(EXPERIMENT))

from dual_ellipse import (choose_eye_pair, ellipse_candidates, reconstruct_low_support,
                          refine, select_candidate)


class DualEllipseMorphTests(unittest.TestCase):
    def config(self):
        return {
            "minimum_core_pixels": 1,
            "weak_kernel": 3,
            "strong_kernel": 5,
            "score_weights": {
                "inner_support": .55, "outer_support": .20,
                "retained_fraction": .20, "boundary_occupancy": .15,
            },
        }

    def test_pose_candidates_shift_along_eye_line(self):
        base = {"center": [50, 50], "axes": [60, 40], "angle_degrees": 90}
        candidates = ellipse_candidates(base, [[30, 30], [70, 30]], .2)
        self.assertEqual([item["name"] for item in candidates], ["center", "image_left", "image_right"])
        np.testing.assert_allclose(candidates[1]["geometry"]["center"], [42, 50])
        np.testing.assert_allclose(candidates[2]["geometry"]["center"], [58, 50])

    def test_relaxed_eye_pair_is_fallback_only(self):
        strict, relaxed = [{"id": "strict"}], [{"id": "relaxed"}]
        self.assertEqual(choose_eye_pair(strict, relaxed), (strict[0], "standard_pair"))
        self.assertEqual(choose_eye_pair([], relaxed), (relaxed[0], "recovered_pair"))
        self.assertEqual(choose_eye_pair([], []), (None, "no_pair"))

    def test_refine_never_adds_pixels_outside_outer(self):
        raw = np.zeros((31, 31), np.uint8)
        raw[10:21, 8:14] = 1
        raw[10:21, 17:23] = 1
        inner = np.zeros_like(raw)
        outer = np.zeros_like(raw)
        cv2.circle(inner, (15, 15), 7, 1, -1)
        cv2.circle(outer, (15, 15), 11, 1, -1)
        status, stages, _ = refine(raw, inner, outer, self.config())
        self.assertEqual(status, "ok")
        self.assertFalse(np.any(stages["final"] > outer))
        self.assertEqual(stages["final"][15, 15], 1)

    def test_component_without_inner_seed_is_removed(self):
        raw = np.zeros((31, 31), np.uint8)
        raw[13:18, 13:18] = 1
        raw[2:5, 2:5] = 1
        inner = np.zeros_like(raw)
        outer = np.ones_like(raw)
        inner[10:21, 10:21] = 1
        status, stages, descriptors = refine(raw, inner, outer, self.config())
        self.assertEqual(status, "ok")
        self.assertEqual(descriptors["retained_components"], 1)
        self.assertFalse(stages["final"][3, 3])

    def test_selection_uses_center_penalty_only_after_image_score(self):
        candidates = [
            {"name": "center", "status": "ok", "descriptors": {"score": .50}},
            {"name": "image_left", "status": "ok", "descriptors": {"score": .51}},
        ]
        self.assertEqual(select_candidate(candidates, .02)["name"], "center")
        candidates[1]["descriptors"]["score"] = .54
        self.assertEqual(select_candidate(candidates, .02)["name"], "image_left")

    def test_low_support_reconstruction_stays_inside_inner(self):
        relaxed = np.zeros((31, 31), np.uint8)
        relaxed[10:21, 8:23] = 1
        seed = np.zeros_like(relaxed)
        seed[13:18, 10:13] = 1
        inner = np.zeros_like(relaxed)
        cv2.circle(inner, (15, 15), 8, 1, -1)
        rebuilt = reconstruct_low_support(relaxed, seed, inner, 3, 1)
        self.assertGreater(rebuilt.sum(), seed.sum())
        self.assertFalse(np.any(rebuilt > inner))

    def test_refine_marks_low_support_reconstruction(self):
        raw = np.zeros((31, 31), np.uint8)
        raw[13:18, 10:13] = 1
        relaxed = np.zeros_like(raw)
        relaxed[9:22, 8:23] = 1
        inner = np.zeros_like(raw)
        outer = np.zeros_like(raw)
        cv2.circle(inner, (15, 15), 8, 1, -1)
        cv2.circle(outer, (15, 15), 12, 1, -1)
        config = {
            **self.config(),
            "low_support_threshold": .5,
            "reconstruction_close_kernel": 3,
            "minimum_reconstruction_seed_pixels": 1,
        }
        status, stages, descriptors = refine(raw, inner, outer, config, relaxed)
        self.assertEqual(status, "ok")
        self.assertTrue(descriptors["low_support_triggered"])
        self.assertGreater(descriptors["reconstruction_pixels"], raw.sum())
        self.assertFalse(np.any(stages["final"] > outer))


if __name__ == "__main__":
    unittest.main()
