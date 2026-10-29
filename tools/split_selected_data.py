"""Assign the selected 162 pairs to train/val/test and build a local review album.

Run: python -X utf8 tools/split_selected_data.py
An existing different split is never silently replaced. Original PNGs stay in place.
"""

import csv
import html
import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import _bootstrap
from face_pixel.part_a_common import PROJECT, DATA, RAW, MANIFEST, content_fingerprint, digest
from build_label_review import write_page


OUT = PROJECT / "results" / "split_review"
SEED = 42
COUNTS = {"train": 98, "val": 32, "test": 32}
TITLES = {"train": "训练集", "val": "验证集", "test": "测试集"}


def similarity_candidates(rows):
    hashes = {}
    for row in rows:
        with Image.open(PROJECT / row["image_path"]) as im:
            pixels = im.convert("L").resize((9, 8), Image.Resampling.LANCZOS).tobytes()
        hashes[row["source_filename"]] = sum(
            (pixels[y * 9 + x] > pixels[y * 9 + x + 1]) << (y * 8 + x)
            for y in range(8) for x in range(8)
        )
    return sorted(
        ({"a": a, "b": b, "dhash_distance": (ha ^ hb).bit_count()}
         for (a, ha), (b, hb) in combinations(sorted(hashes.items()), 2)
         if (ha ^ hb).bit_count() <= 8),
        key=lambda p: (p["dhash_distance"], p["a"], p["b"]),
    )


def make_album(rows, candidates, review_passed=False):
    OUT.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
    by_name = {r["source_filename"]: r for r in rows}
    if candidates:
        sheet = Image.new("RGB", (1200, ((len(candidates) + 3) // 4) * 172), "white")
        draw = ImageDraw.Draw(sheet)
        small_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 13)
        for i, pair in enumerate(candidates):
            x, y = (i % 4) * 300, (i // 4) * 172
            for j, name in enumerate((pair["a"], pair["b"])):
                with Image.open(PROJECT / by_name[name]["image_path"]) as im:
                    sheet.paste(im.resize((140, 140)), (x + j * 145, y + 24))
            draw.text((x, y), f'{pair["a"]} / {pair["b"]} d={pair["dhash_distance"]}',
                      font=small_font, fill="black")
        sheet.save(OUT / "near_duplicate_candidates.png")

    nav = '<nav><a href="index.html">划分说明</a>' + "".join(
        f'<a href="{split}.html">{TITLES[split]} {count} 张</a>' for split, count in COUNTS.items()
    ) + '</nav>'
    for split, count in COUNTS.items():
        members = [r for r in rows if r["split"] == split]
        sheet = Image.new("RGB", (1280, ((count + 7) // 8) * 188), "white")
        draw = ImageDraw.Draw(sheet)
        cards = []
        for i, row in enumerate(members):
            name = row["source_filename"]
            x, y = (i % 8) * 160, (i // 8) * 188
            with Image.open(PROJECT / row["image_path"]) as im:
                sheet.paste(im.resize((156, 156), Image.Resampling.LANCZOS), (x, y))
            draw.text((x + 4, y + 160), name, font=font, fill="#182431")
            # Relative links work in a browser without a server or Javascript.
            cards.append(f'<a class="card" href="../data_review/pages/{Path(name).stem}.html" target="_blank">'
                         f'<img loading="lazy" src="../../data/raw/face_segment/images/{name}" alt="{name}">{html.escape(name)}</a>')
        sheet.save(OUT / f"{split}_contact_sheet.jpg", quality=93)
        write_page(OUT / f"{split}.html", f"{TITLES[split]} · 划分检查",
                   f'<h1>{TITLES[split]} · {count} 张</h1>{nav}'
                   '<p>检查光照、姿态、背景等是否明显失衡。点击图像，在新标签页查看原图与标注对照。</p>'
                   f'<p><a href="{split}_contact_sheet.jpg" target="_blank">打开整组总览图</a></p>'
                   f'<div class="grid">{"".join(cards)}</div>{nav}')
    status = "人工检查已通过" if review_passed else "待人工检查"
    write_page(OUT / "index.html", "Part A 数据划分检查",
               f'<h1>Part A · 数据划分{status}</h1>' + nav +
               '<p>采用约 6∶2∶2：训练 98 张、验证 32 张、测试 32 张。'
               '样本 ID 排序后使用独立随机生成器 random.Random(42) 打乱，一次划分；尚未运行分割算法或根据得分选种子。</p>'
               '<p>原图与标签仍保存在原目录，以 manifest.csv 的 split 字段为准。'
               '请查看三个集合是否有明显分布失衡，并记录需进一步核查的相关照片文件名；当前尚未完成人物身份核验。</p>'
               '<p>低分辨率差值哈希（dHash 64 位，汉明距离 ≤ 8）只用于筛查相似候选，不能证明人物相同或穷尽近重复。'
               f'本次产生 {len(candidates)} 对候选，浏览后未发现明显同照片裁剪或缩放项，未据此建立分组。'
               '<a href="near_duplicate_candidates.png" target="_blank">查看候选对照图</a>。</p>'
               '<p><a href="../../data/split_manifest.csv">统一数据清单</a> · '
               '<a href="../../data/part_a/split_report.json">划分记录</a> · '
               '<a href="../data_review/index.html">全部标注检查图</a></p>')


def main():
    manifest = MANIFEST
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    if len(rows) != sum(COUNTS.values()) or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Expected exactly 162 unique samples; review counts before splitting")
    if len({r["image_sha256"] for r in rows}) != len(rows):
        raise ValueError("Exact duplicate images must be reviewed before splitting")
    for row in rows:
        for path_key, sha_key in (("image_path", "image_sha256"), ("label_path", "label_sha256")):
            if digest(PROJECT / row[path_key]) != row[sha_key]:
                raise ValueError(f'Source changed: {row[path_key]}')

    shuffled = sorted(r["sample_id"] for r in rows)
    random.Random(SEED).shuffle(shuffled)
    assignments, offset = {}, 0
    for split, count in COUNTS.items():
        assignments.update({sid: split for sid in shuffled[offset:offset + count]})
        offset += count
    if any(r["split"] != "unassigned" for r in rows):
        if any(r["split"] != assignments[r["sample_id"]] for r in rows):
            raise ValueError("A different split already exists; refusing to replace it")

    metadata_path = DATA / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    review = metadata.get("annotation_quality_review")
    fingerprint = content_fingerprint(rows)
    if review and review.get("manifest_sha256") == digest(manifest):
        review["content_fingerprint"] = fingerprint
    elif review and review.get("content_fingerprint") != fingerprint:
        raise ValueError("Human quality review does not match current samples")
    candidates = similarity_candidates(rows)
    for row in rows:
        row["split"] = assignments[row["sample_id"]]
    assert dict(Counter(r["split"] for r in rows)) == COUNTS
    previous_split = metadata.get("split", {})
    review_passed = (previous_split.get("human_split_review") == "passed" and
                     previous_split.get("manifest_sha256") == digest(manifest))
    make_album(rows, candidates, review_passed)
    with manifest.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    split_info = {
        "date": "2026-10-29", "seed": SEED, "ratio": [0.6, 0.2, 0.2], "counts": COUNTS,
        "algorithm": "Sort sample IDs; random.Random(seed).shuffle; first 98 train, next 32 val, last 32 test",
        "grouping": "No confirmed same-photo near-duplicate groups; image-level random split",
        "human_split_review": "passed" if review_passed else "pending", "identity_review": "not verified",
        "content_fingerprint": fingerprint, "manifest_sha256": digest(manifest),
        "report_path": "data/part_a/split_report.json",
    }
    if review_passed:
        split_info["human_split_review_record"] = previous_split["human_split_review_record"]
        if "layout_migration" in previous_split:
            split_info["layout_migration"] = previous_split["layout_migration"]
    report = {**split_info, "near_duplicate_screening": {
        "method": "64-bit dHash, Hamming distance <= 8", "candidate_count": len(candidates),
        "visual_review": "Assistant inspected candidate montage; no obvious same-photo crop/scale duplicates observed",
        "limitations": "Not an identity check or exhaustive near-duplicate search",
        "pairs": candidates,
    }}
    (DATA / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata["split_status"] = "assigned_user_reviewed" if review_passed else "assigned_pending_user_review"
    metadata["split"] = split_info
    metadata["verification"]["identity_and_near_duplicate_review"] = (
        "dHash candidate screening and assistant visual check completed; identity not verified; user split review "
        + ("passed" if review_passed else "pending")
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": COUNTS, "seed": SEED, "candidates": len(candidates),
                      "review": str(OUT / "index.html")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
