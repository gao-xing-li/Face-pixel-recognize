import copy
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest

from face_pixel.evaluation import evaluate, summarize
from face_pixel.part_b import compare_validation, restore_frozen_source_bytes


class FrozenValidationTests(unittest.TestCase):
    def setUp(self):
        self.expected = summarize([evaluate([[1, 0, 1]], [[1, 1, 0]])])

    def test_matching_validation_counts_and_metrics_pass(self):
        compare_validation(copy.deepcopy(self.expected), self.expected)

    def test_count_or_image_count_mismatch_stops_replay(self):
        for group, key in (("global", "tp"), (None, "n_images")):
            actual = copy.deepcopy(self.expected)
            if group:
                actual[group][key] += 1
            else:
                actual[key] += 1
            with self.assertRaisesRegex(ValueError, "validation"):
                compare_validation(actual, self.expected)

    def test_metric_mismatch_stops_replay(self):
        for group in ("global", "image_mean"):
            actual = copy.deepcopy(self.expected)
            actual[group]["f1"] += 0.01
            with self.assertRaisesRegex(ValueError, "validation replay"):
                compare_validation(actual, self.expected)

    def test_undefined_metric_policy_must_match(self):
        expected = summarize([evaluate([[0]], [[0]])])
        actual = copy.deepcopy(expected)
        actual["global"]["precision"] = 0
        with self.assertRaisesRegex(ValueError, "validation replay"):
            compare_validation(actual, expected)


class FrozenSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "src/face_pixel"
        self.source.mkdir(parents=True)
        self.path = self.source / "__init__.py"
        self.committed = b'"""Synthetic fixture, never a detector."""\nVALUE = 1\n'
        self.path.write_bytes(self.committed)
        for args in (("init", "-q"), ("config", "core.autocrlf", "false"), ("add", "src/face_pixel/__init__.py"), ("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-q", "-m", "Synthetic fixture")):
            subprocess.run(["git", "-C", str(self.root), *args], check=True, capture_output=True)
        self.config = {"source_sha256": {"__init__.py": hashlib.sha256(self.committed).hexdigest()}}

    def test_only_checkout_newlines_are_restored_from_git(self):
        self.path.write_bytes(self.committed.replace(b"\n", b"\r\n"))
        self.assertEqual(restore_frozen_source_bytes(self.root, self.config), ["__init__.py"])
        self.assertEqual(self.path.read_bytes(), self.committed)
        self.assertEqual(restore_frozen_source_bytes(self.root, self.config), [])

    def test_substantive_source_change_is_not_overwritten(self):
        changed = self.committed.replace(b"VALUE = 1", b"VALUE = 2")
        self.path.write_bytes(changed)
        with self.assertRaisesRegex(ValueError, "substantive"):
            restore_frozen_source_bytes(self.root, self.config)
        self.assertEqual(self.path.read_bytes(), changed)

    def test_expected_hash_cannot_be_changed_to_bypass_git(self):
        changed = self.committed.replace(b"VALUE = 1", b"VALUE = 2")
        self.path.write_bytes(changed)
        self.config["source_sha256"]["__init__.py"] = hashlib.sha256(changed).hexdigest()
        with self.assertRaisesRegex(ValueError, "Git object"):
            restore_frozen_source_bytes(self.root, self.config)
        self.assertEqual(self.path.read_bytes(), changed)

    def test_extra_source_file_is_rejected(self):
        (self.source / "extra.py").write_text("# extra\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "file set"):
            restore_frozen_source_bytes(self.root, self.config)


if __name__ == "__main__":
    unittest.main()
