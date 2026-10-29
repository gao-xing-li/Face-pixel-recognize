"""Build a static, local image/label review album from the selected manifest.

Run from the project root: python -X utf8 tools/build_label_review.py
Requires Pillow. Reads source data without changing labels, splits, or decisions.
"""

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


import _bootstrap
from face_pixel.part_a_common import PROJECT, DATA, RAW, MANIFEST
OUTPUT = PROJECT / "results" / "data_review"
NAMES = ["背景", "脸部皮肤", "鼻子", "眼镜", "左眼", "右眼", "左眉", "右眉",
         "左耳", "右耳", "口腔内部", "上唇", "下唇", "头发", "帽子", "耳饰",
         "项链", "脖子", "衣服"]
# Display colors only; the source mask continues to store IDs 0..18.
COLORS = ["#202733", "#f2ba8c", "#f58231", "#42d4f4", "#e6194b", "#4363d8",
          "#911eb4", "#3cb44b", "#ffe119", "#bfef45", "#800000", "#f032e6",
          "#fabed4", "#469990", "#9a6324", "#aaffc3", "#000075", "#bca6e8", "#a9a9a9"]
STYLE = """
body{font-family:'Microsoft YaHei',sans-serif;margin:24px;background:#f1f4f8;color:#182431}
main{max-width:1584px;margin:auto}a{color:#1859a8}nav{display:flex;gap:24px;margin:18px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:16px}
.card{display:block;background:white;padding:10px;border-radius:8px;text-decoration:none}
.card img{width:100%;display:block;margin-bottom:8px}.panel{width:100%;height:auto;background:white}
p{line-height:1.7}.hint{background:white;padding:14px;border-radius:8px}
"""


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_page(path, title, body):
    path.write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{html.escape(title)}</title><style>{STYLE}</style><main>{body}</main></html>',
        encoding="utf-8",
    )


def relative_url(path, parent):
    from urllib.parse import quote
    return quote(Path(os.path.relpath(path, parent)).as_posix(), safe="/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"),
                        help="Chinese-capable TrueType/OpenType font")
    args = parser.parse_args()
    title_font = ImageFont.truetype(str(args.font), 25)
    label_font = ImageFont.truetype(str(args.font), 19)
    small_font = ImageFont.truetype(str(args.font), 16)
    with (MANIFEST).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty manifest")
    names = [row["source_filename"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("Repeated filenames in manifest")
    for name in names:
        if Path(name).name != name or Path(name).suffix.lower() != ".png":
            raise ValueError(f"Unexpected source filename: {name}")

    # Check the complete input list before generating pages.
    for row in rows:
        for field, hash_field in (("image_path", "image_sha256"), ("label_path", "label_sha256")):
            path = (PROJECT / row[field]).resolve()
            path.relative_to(RAW.resolve())
            if sha256(path) != row[hash_field]:
                raise ValueError(f"Source hash differs from manifest: {path}")

    for folder in ("pages", "panels", "thumbnails"):
        (OUTPUT / folder).mkdir(parents=True, exist_ok=True)
    palette = []
    for color in COLORS:
        palette.extend(bytes.fromhex(color[1:]))
    palette += [0] * (768 - len(palette))
    cards, records = [], []
    for index, row in enumerate(rows):
        name = row["source_filename"]
        stem = Path(name).stem
        with Image.open(PROJECT / row["image_path"]) as source, Image.open(PROJECT / row["label_path"]) as raw:
            source.load()
            raw.load()
            if source.mode != "RGB" or raw.mode != "L" or source.size != raw.size:
                raise ValueError(f"Invalid image/label format: {name}")
            present = [i for i, count in enumerate(raw.histogram()) if count]
            if any(i >= len(NAMES) for i in present):
                raise ValueError(f"Unknown label ID: {name}")
            colored = Image.frombytes("P", raw.size, raw.tobytes())
            colored.putpalette(palette)
            colored = colored.convert("RGB")
            blended = Image.blend(source, colored, 0.45)
            # Background stays unchanged in the overlay, all other classes are shown.
            foreground = raw.point([0] + [255] * 255)
            overlay = Image.composite(blended, source, foreground)
            width, height = source.size
            panel = Image.new("RGB", (width * 3, height + 222), "#ffffff")
            draw = ImageDraw.Draw(panel)
            for column, (picture, heading) in enumerate(zip(
                (source, colored, overlay), ("原图", "彩色标签", "半透明叠加（标签 45%）")
            )):
                draw.text((column * width + 12, 12), heading, font=title_font, fill="#182431")
                panel.paste(picture, (column * width, 54))
            draw.text((12, height + 64), f"{index + 1}/{len(rows)}  {name}  |  原始多类别标注检查 · 非算法预测",
                      font=label_font, fill="#182431")
            for class_id, (label, color) in enumerate(zip(NAMES, COLORS)):
                x = 12 + (class_id % 7) * (width * 3 // 7)
                y = height + 102 + (class_id // 7) * 31
                draw.rectangle((x, y + 3, x + 19, y + 22), fill=color)
                draw.text((x + 26, y), f"{class_id} {label}", font=small_font,
                          fill="#182431" if class_id in present else "#aaaaaa")
            draw.text((12, height + 198), "图例灰字：本图未出现该类别。背景在叠加图中保留原图颜色。",
                      font=small_font, fill="#566575")
            panel.save(OUTPUT / "panels" / name)
            thumbnail = source.copy()
            thumbnail.thumbnail((224, 224))
            thumbnail.save(OUTPUT / "thumbnails" / name)

        page = OUTPUT / "pages" / f"{stem}.html"
        nav = '<nav><a href="../index.html">返回总览</a>'
        if index:
            nav += f'<a href="{Path(names[index - 1]).stem}.html">← 上一张</a>'
        if index + 1 < len(rows):
            nav += f'<a href="{Path(names[index + 1]).stem}.html">下一张 →</a>'
        nav += f'<a href="../panels/{name}" target="_blank">打开原始尺寸检查图</a></nav>'
        body = f'<h1>{index + 1} / {len(rows)} · {html.escape(name)}</h1>{nav}'
        body += f'<img class="panel" src="../panels/{name}" alt="{name} 原图、标签与叠加图">'
        body += nav
        body += '<p>查看：标签是否对齐、是否有明显漏标或错标。疑似问题请记录文件名和现象；本页面不会自动判定通过或修改数据。</p>'
        body += f'<p>本图出现的类别：{html.escape("、".join(f"{i} {NAMES[i]}" for i in present))}</p>'
        body += f'<p><a href="{relative_url(PROJECT / row["image_path"], page.parent)}">源图像</a> · '
        body += f'<a href="{relative_url(PROJECT / row["label_path"], page.parent)}">源标签（类别值，直接显示接近黑色）</a></p>'
        write_page(page, f"{name} · 标签人工检查", body)
        cards.append(f'<a class="card" href="pages/{stem}.html"><img loading="lazy" src="thumbnails/{name}" alt="{name}">{index + 1}. {name}</a>')
        records.append({"sample_id": row["sample_id"], "source_filename": name,
                        "image_sha256": row["image_sha256"], "label_sha256": row["label_sha256"],
                        "present_label_ids": present, "panel": f"panels/{name}",
                        "page": f"pages/{stem}.html"})
    body = f'<h1>Part A · 图像与标签人工检查</h1><p>共 {len(rows)} 对，按数据清单顺序排列。</p>'
    body += '<p class="hint">点击缩略图查看“原图｜彩色标签｜半透明叠加图”，详情页可翻页或打开原始尺寸。'
    body += '重点查看标签是否明显偏移、漏标或错标，也可留意近重复照片。请另行记录问题文件名与现象。'
    body += '页面不保存检查进度；浏览过不代表检查通过。当前显示原始多类别标签，尚未合并为肤色或脸部二值真值。</p>'
    body += f'<div class="grid">{"".join(cards)}</div>'
    write_page(OUTPUT / "index.html", "Part A 标签人工检查", body)
    (OUTPUT / "review_manifest.json").write_text(json.dumps({
        "sample_count": len(records), "source_manifest_sha256": sha256(MANIFEST),
        "overlay_alpha": 0.45, "human_review_status": "not recorded by this tool",
        "classes": [{"id": i, "name": name, "color": COLORS[i]} for i, name in enumerate(NAMES)],
        "samples": records,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": len(records), "index": str(OUTPUT / "index.html")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
