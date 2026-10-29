"""Small numerical checks, separate from the real-data run."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from face_pixel.modeling import histogram_statistics, image_histograms


class SkinStatisticsTests(unittest.TestCase):
    def test_known_colors_channel_order_and_label_selection(self):
        # BGR red -> Cr=255,Cb=85; BGR blue -> Cr=107,Cb=255 (8-bit OpenCV).
        bgr = np.array([[[0, 0, 255], [255, 0, 0], [0, 255, 0], [255, 255, 255]]], dtype=np.uint8)
        labels = np.array([[1, 2, 0, 3]], dtype=np.uint8)
        hist = image_histograms(bgr, labels)
        self.assertEqual((hist["skin_pixels"], hist["nose_pixels"]), (1, 1))
        self.assertEqual(hist["joint_cr_cb"][255, 85], 1)
        self.assertEqual(hist["joint_cr_cb"][107, 255], 1)
        self.assertEqual(int(hist["joint_cr_cb"].sum()), 2)
        self.assertEqual(histogram_statistics(hist["cr"]), {"mean": 181.0, "variance": 5476.0, "std": 74.0})
        self.assertEqual(histogram_statistics(hist["cb"]), {"mean": 170.0, "variance": 7225.0, "std": 85.0})

    def test_pooling_weights_pixels_instead_of_images(self):
        red = image_histograms(np.array([[[0, 0, 255]]], dtype=np.uint8), np.ones((1, 1), dtype=np.uint8))
        blue = image_histograms(np.array([[[255, 0, 0]] * 3], dtype=np.uint8), np.ones((1, 3), dtype=np.uint8))
        result = histogram_statistics(red["cr"] + blue["cr"])
        self.assertEqual(result["mean"], 144.0)
        self.assertEqual(result["variance"], 4107.0)
        self.assertNotEqual(result["mean"], (255 + 107) / 2)

    def test_histogram_matches_direct_population_statistics(self):
        values = np.array([0, 0, 1, 10, 127, 128, 128, 200, 255], dtype=np.float64)
        result = histogram_statistics(np.bincount(values.astype(int), minlength=256))
        self.assertAlmostEqual(result["mean"], values.mean(), places=12)
        self.assertAlmostEqual(result["variance"], values.var(ddof=0), places=10)
        self.assertAlmostEqual(result["std"], values.std(ddof=0), places=12)

    def test_constant_values_have_zero_std(self):
        bgr = np.full((2, 3, 3), 80, dtype=np.uint8)
        hist = image_histograms(bgr, np.ones((2, 3), dtype=np.uint8))
        self.assertEqual(histogram_statistics(hist["cr"])["std"], 0.0)
        self.assertEqual(histogram_statistics(hist["cb"])["mean"], 128.0)

    def test_invalid_or_empty_training_samples_fail(self):
        bgr = np.zeros((1, 2, 3), dtype=np.uint8)
        for labels in (np.array([[0, 3]], dtype=np.uint8), np.array([[1, 19]], dtype=np.uint8),
                       np.ones((2, 2), dtype=np.uint8), np.ones((1, 2), dtype=np.float32)):
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError):
                    image_histograms(bgr, labels)


if __name__ == "__main__":
    unittest.main()
