"""Copy selected Face_segment image/label pairs without altering their pixels."""

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


import _bootstrap
from face_pixel.part_a_common import PROJECT, DATA, RAW, MANIFEST, content_fingerprint, digest
# User decision on 2026-10-29; retain sources but exclude this pair from the experiment.
EXCLUDED = {"00584.png": "Duplicate image of 00583.png; user chose to retain 00583.png"}
LABELS = [
    "background", "skin", "nose", "eye_g", "l_eye", "r_eye",
    "l_brow", "r_brow", "l_ear", "r_ear", "mouth", "u_lip",
    "l_lip", "hair", "hat", "ear_r", "neck_l", "neck", "cloth",
]
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-dir", type=Path, required=True)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=RAW)
    args = parser.parse_args()
    selected, original, labels, output = (
        p.resolve() for p in (args.selected_dir, args.original_dir, args.label_dir, args.output_dir)
    )
    output.relative_to(PROJECT)  # Manifests use paths relative to this project.
    if output != RAW.resolve():
        raise ValueError("This branch stores selected data only in data/raw/face_segment")
    previous_path = DATA / "metadata.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else {}
    old_rows = []
    old_manifest_hash = None
    if (MANIFEST).exists():
        old_manifest_hash = digest(MANIFEST)
        with (MANIFEST).open(encoding="utf-8-sig", newline="") as stream:
            old_rows = list(csv.DictReader(stream))
    source_files = sorted(selected.glob("*.png"))
    files = [p for p in source_files if p.name not in EXCLUDED]
    if not files:
        raise ValueError(f"No PNG images in {selected}")

    rows = []
    copies = []
    observed = set()
    # Validate every source pair and all destination collisions before copying.
    for image in files:
        label = labels / image.name
        original_image = original / image.name
        image_hash, label_hash = digest(image), digest(label)
        if image_hash != digest(original_image):
            raise ValueError(f"Selected image differs from original: {image.name}")
        with Image.open(image) as rgb, Image.open(label) as mask:
            rgb.load()
            mask.load()
            if rgb.mode != "RGB" or mask.mode != "L" or rgb.size != mask.size:
                raise ValueError(f"Unexpected mode/size: {image.name}")
            values = [i for i, n in enumerate(mask.histogram()) if n]
            if not set(values) <= set(range(19)):
                raise ValueError(f"Unexpected label IDs: {image.name}: {values}")
            width, height = rgb.size
        observed.update(values)
        image_out = output / "images" / image.name
        label_out = output / "labels" / label.name
        for src, dst, sha in ((image, image_out, image_hash), (label, label_out, label_hash)):
            if dst.exists() and digest(dst) != sha:
                raise FileExistsError(f"Refusing to overwrite a different file: {dst}")
            copies.append((src, dst, sha))
        rows.append({
            "sample_id": f"face_segment_test_g2_{image.stem}",
            "image_path": image_out.relative_to(PROJECT).as_posix(),
            "label_path": label_out.relative_to(PROJECT).as_posix(),
            "split": "unassigned",
            "source_split": "test",
            "source_group": "2",
            "source_filename": image.name,
            "width": width, "height": height,
            "label_ids_present": " ".join(map(str, values)),
            "image_sha256": image_hash, "label_sha256": label_hash,
        })
    expected = {p.name for p in files}
    for folder in (output / "images", output / "labels"):
        extras = {p.name for p in folder.iterdir()} - expected if folder.exists() else set()
        if extras:
            raise ValueError(f"Destination has files outside this selection: {folder}: {sorted(extras)}")

    # Importing the same data must not silently undo a saved experiment split.
    if any(r["split"] != "unassigned" for r in old_rows):
        if content_fingerprint(old_rows) != content_fingerprint(rows):
            raise ValueError("Selection changed after splitting; review the saved split before re-importing")
        assignments = {r["sample_id"]: r["split"] for r in old_rows}
        for row in rows:
            row["split"] = assignments[row["sample_id"]]

    for src, dst, sha in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        if digest(dst) != sha:
            raise IOError(f"Copy verification failed: {dst}")

    DATA.mkdir(parents=True, exist_ok=True)
    with (MANIFEST).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["image_sha256"] for row in rows)
    # A split-only edit does not invalidate the user's image/label quality review.
    review = previous.get("annotation_quality_review")
    if review:
        if (review.get("content_fingerprint") == content_fingerprint(rows) or
            (review.get("manifest_sha256") == old_manifest_hash and
             content_fingerprint(old_rows) == content_fingerprint(rows))):
            review["content_fingerprint"] = content_fingerprint(rows)
        else:
            review = None
    metadata = {
        "dataset": "User-selected subset of FedST Face_segment (derived from CelebAMask-HQ)",
        "selection_method": "User subjective appearance-based selection; not verified demographic labels",
        "source_selected_directory": "Face_segment/test/real_image/2 (subjective selection)",
        "source_image_directory": "Face_segment/test/real_image/2",
        "source_label_directory": "Face_segment/test/real_label/2",
        "source_selection_count": len(source_files),
        "excluded_samples": [
            {"source_filename": p.name, "reason": EXCLUDED[p.name], "decision_date": "2026-10-29"}
            for p in source_files if p.name in EXCLUDED
        ],
        "sample_count": len(rows),
        "split_status": previous.get("split_status", "unassigned") if any(r["split"] != "unassigned" for r in rows) else "unassigned",
        "image_mode": "RGB", "label_mode": "L",
        "observed_label_ids": sorted(observed),
        "label_map": {str(i): name for i, name in enumerate(LABELS)},
        "label_map_reference": "https://github.com/switchablenorms/CelebAMask-HQ/blob/master/face_parsing/Data_preprocessing/g_mask.py",
        "skin_ids": [1, 2],
        "skin_mapping_status": "confirmed for skin-color modeling on 2026-10-29",
        "face_positive_ids": [1, 2, 4, 5, 6, 7, 10, 11, 12],
        "evaluation_ignore_ids": [3],
        "face_negative_ids": [0, 8, 9, 13, 14, 15, 16, 17, 18],
        "binary_mapping_status": "skin and face evaluation IDs confirmed on 2026-10-29; glasses excluded from all metric counts; no derived binary labels generated",
        "annotation_quality_review": review,
        "transforms_applied": [],
        "verification": {
            "all_selected_images_match_original_sha256": True,
            "all_copied_pairs_match_source_sha256": True,
            "all_pairs_have_matching_dimensions": True,
            "exact_duplicate_image_groups": [
                [row["sample_id"] for row in rows if row["image_sha256"] == sha]
                for sha, count in counts.items() if count > 1
            ],
            "identity_and_near_duplicate_review": previous.get("verification", {}).get("identity_and_near_duplicate_review", "not performed"),
            "per_image_annotation_quality_review": (
                "user visual review completed; no obvious problems reported" if review else "not recorded for this manifest"
            ),
        },
    }
    if any(r["split"] != "unassigned" for r in rows) and "split" in previous:
        metadata["split"] = previous["split"]
    (DATA / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "pairs": len(rows),
                      "observed_label_ids": sorted(observed),
                      "exact_duplicate_groups": sum(n > 1 for n in counts.values())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
