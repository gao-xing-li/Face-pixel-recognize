import sys
from pathlib import Path
import unittest

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments" / "lab4" / "eye_ellipse"
sys.path.insert(0, str(EXPERIMENT))

from eye_ellipse import ellipse_from_eyes, ellipse_region, eye_frame, fit_geometry_prior


def prior(**changes):
    model = {
        "center_along_per_eye_distance": 0.0,
        "center_down_per_eye_distance": 0.65,
        "major_axis_per_eye_distance": 3.0,
        "minor_axis_per_eye_distance": 2.0,
        "major_angle_from_down_degrees": 0.0,
    }
    model.update(changes)
    return {"model": model}


class EyeEllipseTests(unittest.TestCase):
    def test_horizontal_eyes_generate_centered_vertical_ellipse(self):
        geometry = ellipse_from_eyes([[40, 30], [60, 30]], prior())
        np.testing.assert_allclose(geometry["center"], [50, 43])
        np.testing.assert_allclose(geometry["axes"], [60, 40])
        self.assertAlmostEqual(geometry["angle_degrees"], 90)
        region = ellipse_region((100, 100), geometry)
        self.assertEqual(region.shape, (100, 100))
        self.assertEqual(region.dtype, np.uint8)
        self.assertEqual(region[43, 50], 1)

    def test_tilted_eye_line_keeps_major_axis_perpendicular(self):
        centers = [[30, 30], [50, 40]]
        geometry = ellipse_from_eyes(centers, prior(center_down_per_eye_distance=1.0))
        midpoint, _, down, distance = eye_frame(centers)
        np.testing.assert_allclose(geometry["center"], midpoint + distance * down)
        eye_angle = np.degrees(np.arctan2(10, 20))
        self.assertAlmostEqual((geometry["angle_degrees"] - eye_angle) % 180, 90)

    def test_degenerate_eye_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            ellipse_from_eyes([[20, 20], [20, 20]], prior())

    def test_geometry_prior_uses_medians(self):
        samples = []
        for value in (1.0, 2.0, 99.0):
            samples.append({
                "center_along_per_eye_distance": 0.0,
                "center_down_per_eye_distance": value,
                "major_axis_per_eye_distance": 3.0,
                "minor_axis_per_eye_distance": 2.0,
                "major_angle_from_down_degrees": 0.0,
            })
        fitted = fit_geometry_prior(samples, minimum_samples=3)
        self.assertEqual(fitted["model"]["center_down_per_eye_distance"], 2.0)


if __name__ == "__main__":
    unittest.main()
