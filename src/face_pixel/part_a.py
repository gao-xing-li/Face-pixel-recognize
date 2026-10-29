"""Read Part A's frozen raw labels and produce the shared evaluation format.

The source manifest, images, labels and training parameters are never rewritten.
Only declared split names/paths and ground-truth representation are adapted.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image, __version__ as pillow_version

from .data_io import _read_csv
from .evaluation import evaluate

SPLIT_NAMES = {"train": "train", "val": "validation", "test": "test"}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _fingerprint(rows) -> str:
    payload = sorted((row["sample_id"], row["image_sha256"], row["label_sha256"]) for row in rows)
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _under(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError("Part A source paths must be project-relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Part A source path escapes its project root")
    return path


def _relative(path: Path, parent: Path) -> str:
    try:
        return Path(os.path.relpath(path, parent)).as_posix()
    except ValueError:  # Different Windows drives; regenerate on the target machine.
        return path.as_posix()


def label_mapping(metadata: dict) -> dict:
    names = ("skin_ids", "face_positive_ids", "face_negative_ids", "evaluation_ignore_ids")
    result = {name: list(metadata[name]) for name in names}
    for name, values in result.items():
        if not values or any(type(value) is not int or not 0 <= value <= 255 for value in values) or len(set(values)) != len(values):
            raise ValueError(f"invalid or duplicated label IDs: {name}")
    known = {int(value) for value in metadata["label_map"]}
    positive, negative, ignored = (set(result[name]) for name in names[1:])
    if positive & negative or positive & ignored or negative & ignored:
        raise ValueError("positive, negative and ignored label IDs must be disjoint")
    if positive | negative | ignored != known:
        raise ValueError("label mapping must cover the complete declared label map")
    if not set(result["skin_ids"]).issubset(positive):
        raise ValueError("modeling skin IDs must be included in face-positive IDs")
    result["known_label_ids"] = sorted(known)
    result["label_map"] = metadata["label_map"]
    result["positive_name"] = "face region including declared facial parts"
    result["ignore_policy"] = "Exclude ignored IDs from all counts and metric denominators; never use them to guide predictions"
    return result


def derive_face_masks(labels, mapping: dict) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.size == 0 or labels.dtype.kind not in "ui":
        raise ValueError("raw labels must be a nonempty 2-D integer class-ID array")
    if not np.isin(labels, mapping["known_label_ids"]).all():
        raise ValueError("raw label contains unknown class IDs")
    gt = np.isin(labels, mapping["face_positive_ids"]).astype(np.uint8)
    valid = (~np.isin(labels, mapping["evaluation_ignore_ids"])).astype(np.uint8)
    return gt, valid


def prepare_part_a(source_root, output_dir, *, source_revision: str | None = None) -> dict:
    """Audit A's whole split and atomically export binary ground truth/validity.

    If source_revision is supplied, require that exact checked-out Git revision.
    Without it, source hashes are still checked but Git provenance is unverified.
    """
    root, output = Path(source_root).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise ValueError("output directory already exists; choose a new adapter run")
    if output.is_relative_to(root / "data" / "raw"):
        raise ValueError("derived output cannot be placed inside the source raw-data directory")
    if source_revision is not None:
        if len(source_revision) != 40 or any(ch not in "0123456789abcdef" for ch in source_revision):
            raise ValueError("source_revision must be a full lowercase 40-character commit SHA")
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        if revision != source_revision:
            raise ValueError("source Git revision does not match the requested Part A revision")
        tracked = subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", "data/split_manifest.csv", "data/part_a", "data/raw/face_segment", "configs/skin_params.json", "src/face_pixel/modeling.py"], capture_output=True)
        if tracked.returncode == 1:
            raise ValueError("Part A source inputs have tracked modifications relative to the pinned revision")
        tracked.check_returncode()

    manifest = root / "data" / "split_manifest.csv"
    metadata_path = root / "data" / "part_a" / "metadata.json"
    split_path = root / "data" / "part_a" / "split_report.json"
    params_path = root / "configs" / "skin_params.json"
    metadata, split_report, params = (_json(path) for path in (metadata_path, split_path, params_path))
    mapping = label_mapping(metadata)
    mapping_bytes = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    mapping_sha = hashlib.sha256(mapping_bytes).hexdigest()
    rows = _read_csv(manifest, {"sample_id", "image_path", "label_path", "split", "width", "height", "label_ids_present", "image_sha256", "label_sha256"})
    manifest_sha = _digest(manifest)
    for label, expected in (("metadata", metadata["split"]["manifest_sha256"]), ("split report", split_report["manifest_sha256"]), ("model parameters", params["manifest_sha256"])):
        if expected != manifest_sha:
            raise ValueError(f"source manifest hash does not match {label}")
    counts = Counter(row["split"] for row in rows)
    if set(counts) != set(SPLIT_NAMES) or dict(counts) != metadata["split"]["counts"] or dict(counts) != split_report["counts"] or len(rows) != metadata["sample_count"]:
        raise ValueError("source split counts do not match A's metadata/report")
    if metadata["split"].get("human_split_review") != "passed" or split_report.get("human_split_review") != "passed":
        raise ValueError("source split review is not recorded as passed")
    content_fingerprint = _fingerprint(rows)
    for record in (metadata["split"], split_report, metadata["annotation_quality_review"]):
        if record["content_fingerprint"] != content_fingerprint:
            raise ValueError("source content fingerprint does not match a recorded review")
    train = [row for row in rows if row["split"] == "train"]
    train_ids = [row["sample_id"] for row in train]
    if sorted(train_ids) != sorted(params["training_sample_ids"]) or len(train_ids) != params["training_image_count"] or params["fit_split"] != "train":
        raise ValueError("training IDs/count do not match the saved skin parameters")
    if params["training_content_fingerprint"] != _fingerprint(train) or params["skin_label_ids"] != mapping["skin_ids"]:
        raise ValueError("training content or modeling label IDs do not match")
    if params.get("statistics_script_sha256") and params["statistics_script_sha256"] != _digest(root / "src" / "face_pixel" / "modeling.py"):
        raise ValueError("A's statistics script hash does not match its parameter provenance")

    stats = {name: {"images": 0, "total_pixels": 0, "valid_pixels": 0, "face_positive_pixels": 0, "face_negative_pixels": 0, "ignored_pixels": 0, "skin_model_pixels": 0} for name in SPLIT_NAMES.values()}
    canonical, per_image, observed, seen_images, seen_paths = [], [], set(), {}, set()
    wiring_images = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".part-a-", dir=output.parent) as temporary:
        stage = Path(temporary) / "prepared"
        (stage / "masks").mkdir(parents=True)
        (stage / "valid").mkdir()
        for row in rows:
            sample_id = row["sample_id"]
            if sample_id in {".", ".."} or any(char in sample_id for char in '<>:"/\\|?*'):
                raise ValueError("sample_id cannot be used as a safe mask filename")
            image_path, label_path = (_under(root, row[name]) for name in ("image_path", "label_path"))
            if image_path in seen_paths:
                raise ValueError("source image path is reused")
            seen_paths.add(image_path)
            image_sha, label_sha = _digest(image_path), _digest(label_path)
            if image_sha != row["image_sha256"] or label_sha != row["label_sha256"]:
                raise ValueError(f"source image/label hash mismatch: {sample_id}")
            if image_sha in seen_images:
                raise ValueError(f"exact duplicate source image: {seen_images[image_sha]} and {sample_id}")
            seen_images[image_sha] = sample_id
            with Image.open(image_path) as image, Image.open(label_path) as label:
                if image.mode != "RGB" or label.mode != "L" or image.size != label.size or image.size != (int(row["width"]), int(row["height"])):
                    raise ValueError(f"source mode/size mismatch: {sample_id}")
                labels = np.array(label)
                image.verify()
            current_ids = set(int(value) for value in np.unique(labels))
            if current_ids != {int(value) for value in row["label_ids_present"].split()}:
                raise ValueError(f"observed label IDs do not match manifest: {sample_id}")
            observed.update(current_ids)
            gt, valid = derive_face_masks(labels, mapping)
            n_valid = int(np.count_nonzero(valid))
            if n_valid == 0:
                raise ValueError(f"all source pixels are ignored: {sample_id}")
            positive = int(np.count_nonzero(gt & valid))
            negative = n_valid - positive
            skin = int(np.count_nonzero(np.isin(labels, mapping["skin_ids"])))
            normalized_split = SPLIT_NAMES[row["split"]]
            values = {"total_pixels": int(labels.size), "valid_pixels": n_valid, "face_positive_pixels": positive, "face_negative_pixels": negative, "ignored_pixels": int(labels.size) - n_valid, "skin_model_pixels": skin}
            stats[normalized_split]["images"] += 1
            for key, value in values.items():
                stats[normalized_split][key] += value
            if normalized_split == "validation":
                # Fixed constants only test plumbing; no detector or parameters are scored.
                for constant, expected in ((0, (0, 0, negative, positive)), (1, (positive, negative, 0, 0))):
                    result = evaluate(np.full_like(gt, constant), gt, valid)
                    if tuple(result[key] for key in ("tp", "fp", "tn", "fn")) != expected:
                        raise RuntimeError("validation wiring check failed")
                wiring_images += 1
            mask_name, valid_name = f"masks/{sample_id}.png", f"valid/{sample_id}.png"
            Image.fromarray(gt).save(stage / mask_name)
            Image.fromarray(valid).save(stage / valid_name)
            canonical.append({"sample_id": sample_id, "image_path": _relative(image_path, output), "mask_path": mask_name, "split": normalized_split, "valid_mask_path": valid_name, "part_a_split": row["split"], "source_dataset_split": row.get("source_split", ""), "source_label_path": _relative(label_path, output), "image_sha256": image_sha, "source_label_sha256": label_sha})
            per_image.append({"sample_id": sample_id, "split": normalized_split, **values})
        if observed != set(metadata["observed_label_ids"]):
            raise ValueError("observed dataset labels do not match metadata")
        if stats["train"]["skin_model_pixels"] != params["skin_pixel_count"]:
            raise ValueError("training skin-pixel count does not match the saved statistics")
        for name, records in (("split_manifest.csv", canonical), ("per_image_counts.csv", per_image)):
            with (stage / name).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
        normalized_fingerprint = hashlib.sha256(json.dumps({"source_content_fingerprint": content_fingerprint, "mapping_sha256": mapping_sha, "splits": sorted((row["sample_id"], row["split"]) for row in canonical)}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        report = {
            "schema_version": 1, "status": "passed", "scope": "dataset audit and ground-truth adaptation only; no detector performance",
            "source": {"repository": "gao-xing-li/Face-pixel-recognize", "branch": "feat/part-a", "revision": source_revision, "git_revision_verified": source_revision is not None, "manifest_sha256": manifest_sha, "metadata_sha256": _digest(metadata_path), "split_report_sha256": _digest(split_path), "skin_params_sha256": _digest(params_path), "content_fingerprint": content_fingerprint},
            "mapping": mapping, "mapping_sha256": mapping_sha, "normalized_content_fingerprint": normalized_fingerprint,
            "normalized_manifest_sha256": _digest(stage / "split_manifest.csv"), "output_encoding": "binary01",
            "splits": stats, "source_file_hashes_verified": len(rows) * 2, "exact_duplicate_image_groups": [],
            "training_provenance_verified": True, "training_skin_pixels_verified": stats["train"]["skin_model_pixels"],
            "training_color_statistics_recomputed": False,
            "validation_wiring_check": {"images": wiring_images, "constant_background_and_foreground_counts": "passed", "is_detector_evaluation": False},
            "runtime_versions": {"numpy": np.__version__, "Pillow": pillow_version},
            "limitations": ["No identity or exhaustive near-duplicate verification", "Appearance-based source selection is not verified demographic annotation", "Raw labels were audited structurally; no new manual per-pixel quality approval", "Color means/variances were not recomputed; saved A statistics were checked for source provenance and skin-pixel count only", "No B detector, selected k or frozen configuration is present; no formal test scores were computed"],
        }
        _write_json(stage / "audit.json", report)
        _write_json(stage / "label_mapping.json", mapping)
        (stage / "README.md").write_text("# Part A normalized evaluation data\n\nGround truth and valid masks use integer 0/1. Relative paths are based on split_manifest.csv. The original A manifest, split, images and labels are unchanged.\n\nAudit and constant-mask validation checks are pipeline verification, not detector performance. Ignore masks are evaluator inputs only; never pass them to a detector.\n", encoding="utf-8")
        stage.rename(output)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit Part A and export binary face ground truth without changing the source split.")
    parser.add_argument("--source-root", required=True, help="Checkout of Part A, containing data/ and configs/")
    parser.add_argument("--source-revision", required=True, help="Full pinned Git commit SHA of Part A")
    parser.add_argument("--output", required=True, help="New output directory for normalized masks and manifest")
    args = parser.parse_args(argv)
    try:
        report = prepare_part_a(args.source_root, args.output, source_revision=args.source_revision)
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"Part A preparation failed: {exc}\n")
    counts = {name: data["images"] for name, data in report["splits"].items()}
    print(f"Part A audit passed: {counts}. Masks and audit: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
