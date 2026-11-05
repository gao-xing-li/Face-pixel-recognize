import unittest

import numpy as np

from face_pixel.evaluation import evaluate, summarize


class EvaluationTests(unittest.TestCase):
    def test_hand_calculated_mixed_case(self):
        result = evaluate([[1, 0, 1], [0, 0, 0]], [[1, 1, 0], [0, 0, 0]])
        self.assertEqual([result[key] for key in ("tp", "fp", "tn", "fn", "n_valid")], [1, 1, 3, 1, 6])
        self.assertAlmostEqual(result["accuracy"], 4 / 6)
        for name in ("precision", "recall", "f1"):
            self.assertEqual(result[name], 0.5)
        self.assertAlmostEqual(result["iou"], 1 / 3)

    def test_prediction_first_argument_order(self):
        result = evaluate([[1, 1, 1]], [[1, 0, 0]])
        self.assertAlmostEqual(result["precision"], 1 / 3)
        self.assertEqual(result["recall"], 1)

    def test_perfect_positive_case(self):
        result = evaluate([[1, 0], [0, 1]], [[1, 0], [0, 1]])
        for name in ("accuracy", "precision", "recall", "f1", "iou"):
            self.assertEqual(result[name], 1)

    def test_all_missed_and_all_false_positive(self):
        missed = evaluate([[0, 0]], [[1, 1]])
        false = evaluate([[1, 1]], [[0, 0]])
        self.assertEqual(missed["fn"], 2)
        self.assertIsNone(missed["precision"])
        self.assertEqual(missed["recall"], 0)
        self.assertEqual(missed["f1"], 0)
        self.assertEqual(false["fp"], 2)
        self.assertEqual(false["precision"], 0)
        self.assertIsNone(false["recall"])
        self.assertEqual(false["iou"], 0)

    def test_no_skin_is_undefined_except_accuracy(self):
        result = evaluate([[0, 0]], [[0, 0]])
        self.assertEqual(result["accuracy"], 1)
        for name in ("precision", "recall", "f1", "iou"):
            self.assertIsNone(result[name])

    def test_ignore_mask_excludes_errors(self):
        result = evaluate([[1, 0, 1]], [[1, 1, 0]], [[1, 0, 0]])
        self.assertEqual(result["n_valid"], 1)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"] + result["fn"], 0)

    def test_invalid_inputs_are_rejected(self):
        cases = [
            ([[1]], [[1, 0]], None),
            ([[255]], [[1]], None),
            ([[-1]], [[1]], None),
            ([[0.5]], [[1]], None),
            (np.array([[1.0]]), [[1]], None),
            ([1], [1], None),
            ([[]], [[]], None),
            ([[1]], [[1]], [[0]]),
            ([[1]], [[1]], [[1, 1]]),
            ([[1]], [[1]], [[2]]),
        ]
        for pred, gt, valid in cases:
            with self.subTest(pred=pred, gt=gt, valid=valid):
                with self.assertRaises(ValueError):
                    evaluate(pred, gt, valid)

    def test_boolean_arrays(self):
        result = evaluate(np.array([[True, False]]), np.array([[True, True]]))
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fn"], 1)

    def test_global_pools_counts_instead_of_averaging_scores(self):
        small = evaluate([[1, 0]], [[1, 1]])
        large = evaluate([[1] * 8 + [0] * 2], [[1] * 8 + [0] * 2])
        result = summarize([small, large])
        self.assertEqual(result["global"]["n_valid"], 12)
        self.assertAlmostEqual(result["global"]["f1"], 18 / 19)
        self.assertAlmostEqual(result["image_mean"]["f1"], 5 / 6)
        self.assertNotEqual(result["global"]["f1"], result["image_mean"]["f1"])

    def test_image_means_exclude_undefined_and_report_denominators(self):
        positive = evaluate([[1]], [[1]])
        background = evaluate([[0]], [[0]])
        result = summarize([positive, background])
        self.assertEqual(result["image_mean"]["f1"], 1)
        self.assertEqual(result["image_mean_n_defined"]["f1"], 1)
        self.assertEqual(result["image_mean_n_defined"]["accuracy"], 2)
        undefined = summarize([background])
        self.assertIsNone(undefined["image_mean"]["f1"])
        self.assertEqual(undefined["image_mean_n_defined"]["f1"], 0)

    def test_inconsistent_counts_and_empty_sets_are_rejected(self):
        for rows in ([], [{"tp": 1, "fp": 0, "tn": 0, "fn": 0, "n_valid": 2}], [{"tp": -1, "fp": 0, "tn": 0, "fn": 0, "n_valid": 0}]):
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    summarize(rows)


if __name__ == "__main__":
    unittest.main()
