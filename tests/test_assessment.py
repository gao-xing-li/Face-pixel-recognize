import unittest

import numpy as np

from face_pixel.assessment import boundary_iou, paired_bootstrap_f1, paired_mean_interval


class BoundaryIoUTests(unittest.TestCase):
    def test_perfect_mask_and_image_border(self):
        full = np.ones((5, 6), dtype=np.uint8)
        result = boundary_iou(full, full)
        self.assertEqual(result["boundary_iou"], 1)
        self.assertEqual(result["boundary_union"], 18)

    def test_shifted_square_has_boundary_iou_one_third(self):
        gt, pred = np.zeros((5, 6), dtype=np.uint8), np.zeros((5, 6), dtype=np.uint8)
        gt[1:4, 1:4], pred[1:4, 2:5] = 1, 1
        result = boundary_iou(pred, gt)
        # Two 3x3 one-pixel outlines have 8 pixels each, with 4 shared pixels.
        self.assertEqual(result["boundary_intersection"], 4)
        self.assertEqual(result["boundary_union"], 12)
        self.assertAlmostEqual(result["boundary_iou"], 1 / 3)

    def test_empty_union_is_undefined_and_missing_region_scores_zero(self):
        empty, full = np.zeros((5, 6), dtype=np.uint8), np.ones((5, 6), dtype=np.uint8)
        self.assertIsNone(boundary_iou(empty, empty)["boundary_iou"])
        self.assertEqual(boundary_iou(empty, full)["boundary_iou"], 0)

    def test_ignore_region_does_not_create_an_evaluation_boundary(self):
        gt = np.ones((8, 8), dtype=np.uint8)
        pred, valid = gt.copy(), gt.copy()
        pred[3:5, 3:5], valid[2:6, 2:6] = 0, 0
        self.assertEqual(boundary_iou(pred, gt, valid)["boundary_iou"], 1)

    def test_band_width_uses_image_diagonal(self):
        mask = np.ones((512, 512), dtype=np.uint8)
        self.assertEqual(boundary_iou(mask, mask)["band_radius_pixels"], 14)

    def test_invalid_masks_and_settings_fail(self):
        for kwargs in ({"dilation_ratio": 0}, {"dilation_ratio": float("nan")}, {"dilation_ratio": True}, {"valid_mask": [[0]]}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    boundary_iou([[1]], [[1]], **kwargs)
        with self.assertRaises(ValueError):
            boundary_iou([[255]], [[1]])
        with self.assertRaises(ValueError):
            boundary_iou([[1, 0]], [[1]])


class PairedBootstrapTests(unittest.TestCase):
    def test_identical_schemes_have_exactly_zero_paired_difference(self):
        counts = np.array([[1, 0, 2, 1], [2, 1, 1, 0]], dtype=np.int64)
        result = paired_bootstrap_f1(counts, counts, n_resamples=500)
        self.assertEqual(result["final_minus_baseline_global_f1"], {"estimate": 0, "low": 0, "high": 0})

    def test_known_constant_improvement_and_reproducibility(self):
        base = np.array([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=np.int64)
        final = np.array([[2, 0, 2, 0], [2, 0, 2, 0]], dtype=np.int64)
        first = paired_bootstrap_f1(base, final, n_resamples=500, seed=17)
        self.assertEqual(first, paired_bootstrap_f1(base, final, n_resamples=500, seed=17))
        self.assertEqual(first["final_minus_baseline_global_f1"], {"estimate": 0.5, "low": 0.5, "high": 0.5})

    def test_counts_must_be_integer_nonnegative_and_genuinely_paired(self):
        counts = np.array([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=np.int64)
        invalid = [counts.astype(float), counts * -1, counts[:1], counts[:, :3]]
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    paired_bootstrap_f1(value, value, n_resamples=500)
        changed_truth = counts.copy()
        changed_truth[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "same true"):
            paired_bootstrap_f1(counts, changed_truth, n_resamples=500)
        with self.assertRaises(ValueError):
            paired_bootstrap_f1(counts, counts, n_resamples=1)
        with self.assertRaises(ValueError):
            paired_bootstrap_f1(np.zeros((2, 4), dtype=np.int64), np.zeros((2, 4), dtype=np.int64))

    def test_mean_difference_preserves_pairs_and_known_shift(self):
        values = [0.2, 0.8, 0.4, 0.9]
        zero = paired_mean_interval(values, values, n_resamples=500)
        self.assertEqual((zero["estimate"], zero["low"], zero["high"]), (0, 0, 0))
        shifted = paired_mean_interval([0.25, 0.5], [0.5, 0.75], n_resamples=500)
        self.assertEqual((shifted["estimate"], shifted["low"], shifted["high"]), (0.25, 0.25, 0.25))

    def test_mean_difference_rejects_empty_unpaired_or_nonfinite_values(self):
        for base, final in (([], []), ([0.5], [0.5]), ([0.5, 0.6], [0.5]), ([0.5, np.nan], [0.5, 0.6])):
            with self.subTest(base=base, final=final):
                with self.assertRaises(ValueError):
                    paired_mean_interval(base, final, n_resamples=500)


if __name__ == "__main__":
    unittest.main()
