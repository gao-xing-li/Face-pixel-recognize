"""C's independent replay/test of B's frozen checkout in a separate process.

Frozen B files/configurations stay in B's checkout. C supplies independently
derived truth and scoring. Validation replay must pass before test predictions.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import posixpath
import subprocess
import tempfile

import numpy as np
from PIL import Image, __version__ as pillow_version

from .batch import evaluate_predictions
from .data_io import _read_csv, read_binary_mask, read_dataset_manifest
from .evaluation import COUNT_NAMES, METRIC_NAMES
from .part_a import _digest, _json, _under, _write_json, derive_face_masks

FROZEN_B_REVISION = "19e1d7409991c5896dfca9a6ef03694d2276a152"


def _git(root, *args, binary=False):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True)
    return result.stdout if binary else result.stdout.decode("utf-8").strip()


def restore_frozen_source_bytes(root: Path, config: dict) -> list[str]:
    """Restore only Git-committed bytes changed by checkout newline conversion.

    The expected hash must independently match the Git object. Neither arbitrary
    code changes nor edits to the recorded expected hashes are accepted.
    """
    source = root / "src/face_pixel"
    expected = config["source_sha256"]
    if set(expected) != {path.name for path in source.glob("*.py")}:
        raise ValueError("B source file set differs from the frozen configuration")
    repairs = []
    pending = []
    for name, wanted in expected.items():
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("invalid frozen source filename")
        committed = _git(root, "show", f"HEAD:src/face_pixel/{name}", binary=True)
        if hashlib.sha256(committed).hexdigest() != wanted:
            raise ValueError(f"frozen hash does not match B's Git object: {name}")
        current = (source / name).read_bytes()
        if current != committed:
            if current.replace(b"\r\n", b"\n") != committed.replace(b"\r\n", b"\n"):
                raise ValueError(f"B source has a substantive local change: {name}")
            pending.append((source / name, committed))
            repairs.append(name)
    for path, data in pending:
        path.write_bytes(data)
    return repairs


def compare_validation(actual: dict, expected: dict) -> None:
    if actual["n_images"] != expected["n_images"]:
        raise ValueError("validation image count differs from B's published result")
    for name in COUNT_NAMES:
        if actual["global"][name] != expected["global"][name]:
            raise ValueError(f"validation replay differs from B: global.{name}")
    for group in ("global", "image_mean"):
        for name in METRIC_NAMES:
            a, b = actual[group][name], expected[group][name]
            if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-12):
                raise ValueError(f"validation replay differs from B: {group}.{name}")


def prepare_frozen_view(b_root, c_manifest) -> dict:
    """Rebuild B's exact recorded derived view, including portable POSIX paths.

    Run this under B's pinned Pillow environment. Every generated mask and the
    complete manifest must match the frozen artifact before it is published.
    """
    root, c_manifest = Path(b_root).resolve(), Path(c_manifest).resolve()
    canonical = {row["sample_id"]: row for row in read_dataset_manifest(c_manifest)}
    config = _json(root / "configs/final_config.json")
    audit = _json(c_manifest.parent / "audit.json")
    adapter_path = root / "artifacts/part_b/adapter.json"
    adapter = _json(adapter_path)
    if _digest(adapter_path) != config["adapter_sha256"]:
        raise ValueError("published adapter artifact does not match B's frozen configuration")
    original_manifest = root / "data/split_manifest.csv"
    if _digest(original_manifest) != audit["source"]["manifest_sha256"] or _digest(original_manifest) != config["source_manifest_sha256"]:
        raise ValueError("B and C do not use the same A source manifest")
    if _digest(root / "configs/skin_params.json") != audit["source"]["skin_params_sha256"]:
        raise ValueError("B and C do not use the same A statistics")
    original = _read_csv(original_manifest, {"sample_id", "image_path", "label_path", "split", "image_sha256", "label_sha256"})
    recorded = {row["sample_id"]: row for row in adapter["samples"]}
    if len(recorded) != len(original) or set(recorded) != set(canonical) or set(recorded) != {row["sample_id"] for row in original}:
        raise ValueError("A/B/C sample IDs do not match")
    if adapter["positive_ids"] != audit["mapping"]["face_positive_ids"] or adapter["ignore_ids"] != audit["mapping"]["evaluation_ignore_ids"]:
        raise ValueError("B and C face/ignore mappings do not match")
    output = root / "data/processed/part_b"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".c-replay-view-", dir=output.parent) as temporary:
        stage = Path(temporary) / "view"
        (stage / "masks").mkdir(parents=True)
        (stage / "valid").mkdir()
        rows = []
        for row in original:
            sid = row["sample_id"]
            if sid in {".", ".."} or any(char in sid for char in '<>:"/\\|?*'):
                raise ValueError("unsafe source sample ID")
            expected, reference = recorded[sid], canonical[sid]
            split = "validation" if row["split"] == "val" else row["split"]
            if split != reference["split"] or split != expected["split"]:
                raise ValueError("B/C split assignment differs")
            image_path, label_path = (_under(root, row[name]) for name in ("image_path", "label_path"))
            image_sha, label_sha = _digest(image_path), _digest(label_path)
            if image_sha != row["image_sha256"] or image_sha != expected["image_sha256"] or image_sha != reference["image_sha256"]:
                raise ValueError(f"B image hash differs: {sid}")
            if label_sha != row["label_sha256"] or label_sha != expected["raw_label_sha256"] or label_sha != reference["source_label_sha256"]:
                raise ValueError(f"B label hash differs: {sid}")
            with Image.open(label_path) as raw:
                gt, valid = derive_face_masks(np.array(raw), audit["mapping"])
            if not np.array_equal(gt, read_binary_mask(reference["mask_path"])) or not np.array_equal(valid, read_binary_mask(reference["valid_mask_path"])):
                raise ValueError(f"B/C derived truth pixels differ: {sid}")
            for name, array, key in (("masks", gt, "mask_sha256"), ("valid", valid, "valid_sha256")):
                target = stage / name / f"{sid}.png"
                Image.fromarray(array * 255).save(target)
                if _digest(target) != expected[key]:
                    raise ValueError(f"derived PNG does not reproduce B's frozen bytes: {sid}/{name}; use B's locked Pillow version")
            rows.append({"sample_id": sid, "image_path": posixpath.relpath(image_path.as_posix(), output.as_posix()), "mask_path": f"masks/{sid}.png", "split": split, "valid_mask_path": f"valid/{sid}.png"})
        with (stage / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\r\n")
            writer.writeheader()
            writer.writerows(rows)
        if _digest(stage / "manifest.csv") != config["data_manifest_sha256"]:
            raise ValueError("portable derived manifest does not match B's frozen bytes")
        (stage / "adapter.json").write_bytes(adapter_path.read_bytes())
        if output.exists():
            if _digest(output / "manifest.csv") != config["data_manifest_sha256"] or _digest(output / "adapter.json") != config["adapter_sha256"]:
                raise ValueError("existing B derived view differs; use a fresh B checkout")
            for sid, expected in recorded.items():
                if _digest(output / "masks" / f"{sid}.png") != expected["mask_sha256"] or _digest(output / "valid" / f"{sid}.png") != expected["valid_sha256"]:
                    raise ValueError("existing B derived mask differs")
        else:
            stage.rename(output)
    return {"status": "passed", "images": len(rows), "manifest_sha256": config["data_manifest_sha256"], "adapter_sha256": config["adapter_sha256"], "b_c_truth_pixel_equality": True, "frozen_bytes_reproduced": True}


def _command(python, module, args, source_path, cwd):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(source_path), PYTHONNOUSERSITE="1", PYTHONIOENCODING="utf-8")
    result = subprocess.run([str(python), "-m", module, *map(str, args)], cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise ValueError(f"{module} failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout


def run_frozen_part_b(b_root, b_python, c_manifest, output_dir, *, source_revision=FROZEN_B_REVISION) -> dict:
    root, python, manifest, output = (Path(value).resolve() for value in (b_root, b_python, c_manifest, output_dir))
    if output.exists():
        raise ValueError("output already exists; use a new independent run directory")
    if _git(root, "rev-parse", "HEAD") != source_revision:
        raise ValueError("B checkout does not match the pinned revision")
    audit = _json(manifest.parent / "audit.json")
    if audit["status"] != "passed" or _digest(manifest) != audit["normalized_manifest_sha256"]:
        raise ValueError("C manifest no longer matches its independent A audit")
    configs = {label: _json(root / f"configs/{label}_config.json") for label in ("baseline", "final")}
    repairs = restore_frozen_source_bytes(root, configs["final"])
    for config in configs.values():
        if config.get("synthetic_demo") is not False or config["selection"].get("fixed") is not True or config["selection"].get("test_used") is not False or config["selection"].get("split") != "validation":
            raise ValueError("B configuration is not frozen independently of test")
        if config["source_sha256"] != configs["final"]["source_sha256"] or config["source_manifest_sha256"] != audit["source"]["manifest_sha256"]:
            raise ValueError("B frozen configurations have inconsistent source provenance")
        if _digest(root / "configs/validation_plan.json") != config["selection"]["plan_sha256"]:
            raise ValueError("B validation plan changed")
        for name, wanted in config["source_sha256"].items():
            if _digest(root / "src/face_pixel" / name) != wanted:
                raise ValueError("B frozen source hash mismatch")
    if _git(root, "diff", "--name-only", "HEAD", "--", "src/face_pixel", "configs", "artifacts/part_b", "data/split_manifest.csv"):
        raise ValueError("B checkout has tracked changes beyond checkout newline conversion")
    c_source = Path(__file__).resolve().parents[1]
    _command(python, "face_pixel.part_b", ["prepare-view", "--b-root", root, "--manifest", manifest], c_source, root)

    output.parent.mkdir(parents=True, exist_ok=True)
    summaries, runtimes, validation_checks = {}, {}, {}
    with tempfile.TemporaryDirectory(prefix=".independent-b-", dir=output.parent) as temporary:
        stage = Path(temporary) / "run"
        stage.mkdir()
        # Both validation replays must pass before either test configuration runs.
        for split in ("validation", "test"):
            for label, config in configs.items():
                name = f"{split}-{label}"
                predictions = stage / "predictions" / name
                config_path = root / f"configs/{label}_config.json"
                _command(python, "face_pixel.validation", ["predict", "--manifest", root / "data/processed/part_b/manifest.csv", "--config", config_path, "--split", split, "--output", predictions], root / "src", root)
                run = _json(predictions / "run.json")
                if run["ground_truth_read"] is not False or run["metrics_computed"] is not False:
                    raise ValueError("B inference did not declare image-only prediction")
                if run["runtime"] != config["runtime"]:
                    raise ValueError("inference runtime differs from B's frozen runtime")
                summary = evaluate_predictions(manifest, predictions / "predictions.csv", stage / "evaluation" / name, split=split, config_id=config["config_id"], run_config_path=config_path, gt_encoding="binary01", pred_encoding="binary255", valid_encoding="binary01")
                if split == "validation":
                    compare_validation(summary, _json(root / f"artifacts/part_b/{label}_validation/summary.json"))
                    validation_checks[label] = "exact counts and metrics match B's published validation"
                summaries[name], runtimes[label] = summary, run["runtime"]
                print(f"{name}: {summary['n_images']} images, F1={summary['global']['f1']:.6f}, IoU={summary['global']['iou']:.6f}", flush=True)
        comparison = {}
        for split in ("validation", "test"):
            comparison[split] = {name: summaries[f"{split}-final"]["global"][name] - summaries[f"{split}-baseline"]["global"][name] for name in METRIC_NAMES}
        report = {
            "status": "passed", "scope": "independent fixed-configuration validation replay and test evaluation",
            "b_revision": source_revision, "b_algorithm_revision": configs["final"]["code_revision"],
            "a_source": audit["source"], "normalized_content_fingerprint": audit["normalized_content_fingerprint"],
            "c_manifest_sha256": _digest(manifest), "b_config_sha256": {label: _digest(root / f"configs/{label}_config.json") for label in configs},
            "b_source_sha256": configs["final"]["source_sha256"],
            "c_source_sha256": {path.name: _digest(path) for path in Path(__file__).parent.glob("*.py")},
            "checkout_newline_restorations": repairs, "validation_replay": validation_checks,
            "b_runtime": runtimes["final"], "c_runtime": {"numpy": np.__version__, "Pillow": pillow_version},
            "results": {name: {"n_images": value["n_images"], "global": value["global"], "image_mean": value["image_mean"], "image_mean_n_defined": value["image_mean_n_defined"]} for name, value in summaries.items()},
            "final_minus_baseline": comparison, "configuration_changed_after_test": False,
            "limitations": audit["limitations"][:3] + ["Largest-component postprocessing assumes a single dominant face region; not validated for multiple faces", "Results apply to this fixed, appearance-selected 32-image test split"],
        }
        _write_json(stage / "independent_test.json", report)
        (stage / "README.md").write_text("# Independent frozen B replay and test\n\nBoth baseline and final configurations were fixed by B on validation before this test. Validation counts and metrics were replayed exactly before test predictions. No configuration was changed after viewing test results.\n\nPrediction uses B's immutable source in its own environment. Evaluation uses C's independently derived A labels. See independent_test.json for revisions, source hashes, runtimes and results.\n", encoding="utf-8")
        stage.rename(output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="C: reproduce frozen B validation, then independently evaluate the fixed test split")
    commands = parser.add_subparsers(dest="command", required=True)
    view = commands.add_parser("prepare-view", help="rebuild the exact B derived bytes using B's locked Pillow")
    view.add_argument("--b-root", required=True)
    view.add_argument("--manifest", required=True)
    run = commands.add_parser("run")
    run.add_argument("--b-root", required=True)
    run.add_argument("--b-python", required=True)
    run.add_argument("--manifest", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--source-revision", default=FROZEN_B_REVISION)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-view":
            report = prepare_frozen_view(args.b_root, args.manifest)
            print(json.dumps(report, indent=2))
        else:
            run_frozen_part_b(args.b_root, args.b_python, args.manifest, args.output, source_revision=args.source_revision)
            print(f"Independent fixed-configuration test completed: {Path(args.output).resolve()}")
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(2, f"Independent B test failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
