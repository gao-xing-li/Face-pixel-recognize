"""Generate an offline gallery for every image using the existing fixed detectors."""

import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import tempfile

import _bootstrap  # noqa: F401: enable project-local imports
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from face_pixel.assessment import boundary_iou
from face_pixel.data_io import PROJECT, _read_csv, read_binary_mask, read_json, write_json
from face_pixel.dataset import binary_labels
from face_pixel.detection import detect, validate_model
from face_pixel.evaluation import evaluate, summarize
from face_pixel.visualization import _font_path


def error_image(pred, gt, valid):
    """Solid colors match the earlier five-panel comparison figures."""
    pred, gt, valid = (value.astype(bool) for value in (pred, gt, valid))
    canvas = np.zeros((*gt.shape, 3), dtype=np.uint8)
    for mask, color in ((pred & gt & valid, (35, 175, 95)),
                        (pred & ~gt & valid, (235, 66, 65)),
                        (~pred & gt & valid, (45, 105, 240)),
                        (~valid, (255, 195, 0))):
        canvas[mask] = color
    return Image.fromarray(canvas)


def board(rgb, gt, valid, predictions, sid, split, path, font_path):
    tile, gap, top, caption = 512, 16, 72, 40
    width = tile * 3 + gap * 4
    height = top + (tile + caption + gap) * 2 + 50
    canvas = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.truetype(font_path, 27)
    font = ImageFont.truetype(font_path, 23)
    draw.text((gap, 16), f"{sid}  |  {split}  |  原方案与 41×41 补全", font=title, fill="#162e38")
    panels = [
        ("原图", Image.fromarray(rgb), False),
        ("原方案：最大连通域 → 填孔", Image.fromarray(predictions['original'] * 255), True),
        ("补全：膨胀 → 填孔 → 腐蚀", Image.fromarray(predictions['final'] * 255), True),
        ("脸部真值（黄色为忽略区）", Image.fromarray(np.where(valid[..., None].astype(bool), np.repeat((gt * 255)[..., None], 3, axis=2), [255, 195, 0]).astype(np.uint8)), True),
        ("原方案误差", error_image(predictions['original'], gt, valid), True),
        ("补全方案误差", error_image(predictions['final'], gt, valid), True),
    ]
    for index, (label, picture, categorical) in enumerate(panels):
        row, col = divmod(index, 3)
        x, y = gap + col * (tile + gap), top + row * (tile + caption + gap)
        draw.text((x, y), label, font=font, fill="#162e38")
        picture = picture.convert('RGB')
        picture.thumbnail((tile, tile), Image.Resampling.NEAREST if categorical else Image.Resampling.LANCZOS)
        canvas.paste(picture, (x + (tile - picture.width) // 2, y + caption + (tile - picture.height) // 2))
    draw.text((gap, height - 39), "绿 TP 正确检出 · 红 FP 误检 · 蓝 FN 漏检 · 黄 忽略区 · 黑 TN 正确背景", font=font, fill="#435961")
    canvas.save(path)


def build(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    configs = {name: read_json(PROJECT / 'configs' / f'{name}_config.json') for name in ('original', 'final')}
    params = read_json(PROJECT / 'configs/skin_params.json')
    validate_model(params)
    rows = _read_csv(PROJECT / 'data/split_manifest.csv', {'sample_id', 'image_path', 'label_path', 'split'})
    font_path = _font_path()
    git = ['git', '-c', f'safe.directory={PROJECT.as_posix()}', '-c', 'core.quotepath=false', '-C', str(PROJECT)]
    revision, dirty = None, []
    if (PROJECT / '.git').exists():
        try:
            revision = subprocess.run(git + ['rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
            dirty = subprocess.run(git + ['status', '--short', '--untracked-files=no'], capture_output=True, text=True, encoding='utf-8', check=True).stdout.splitlines()
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
    records, scores, archive_checks = [], {}, 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.review-gallery-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'gallery'
        (stage / 'images').mkdir(parents=True)
        (stage / 'thumbs').mkdir()
        for index, row in enumerate(rows):
            sid = row['sample_id']
            if not re.fullmatch(r'[A-Za-z0-9_-]+', sid):
                raise ValueError(f'Invalid sample ID: {sid}')
            split = {'val': 'validation'}.get(row['split'], row['split'])
            if split not in {'train', 'validation', 'test'}:
                raise ValueError(f'Invalid split: {split}')
            with Image.open(PROJECT / row['image_path']) as im:
                rgb = np.array(im.convert('RGB'))
            with Image.open(PROJECT / row['label_path']) as im:
                gt, valid = binary_labels(np.array(im))
            predictions, metrics = {}, {}
            for name, config in configs.items():
                pred = detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), params, config)
                predictions[name] = pred
                metrics[name] = {**evaluate(pred, gt, valid), **boundary_iou(pred, gt, valid)}
                scores.setdefault((split, name), []).append(metrics[name])
                if split != 'train':
                    version = 'v1' if name == 'original' else 'v2'
                    archived = PROJECT / f'artifacts/part_c/supplementary_morphology/predictions/{split}-{version}/predictions/{sid}.png'
                    if not np.array_equal(pred, read_binary_mask(archived, 'binary255')):
                        raise ValueError(f'Current prediction differs from archived result: {split}/{name}/{sid}')
                    archive_checks += 1
            board(rgb, gt, valid, predictions, sid, split, stage / 'images' / f'{sid}.png', font_path)
            thumb = Image.fromarray(rgb)
            thumb.thumbnail((96, 96))
            thumb.save(stage / 'thumbs' / f'{sid}.jpg', quality=85)
            records.append({'id': sid, 'short_id': sid.rsplit('_', 1)[-1], 'split': split,
                            'metrics': metrics, 'delta_f1': metrics['final']['f1'] - metrics['original']['f1']})
            if (index + 1) % 20 == 0 or index + 1 == len(rows):
                print(f'Rendered {index + 1}/{len(rows)}', flush=True)
        summaries = {}
        for split in ('train', 'validation', 'test'):
            summaries[split] = {}
            for name in configs:
                values = scores[(split, name)]
                defined = [m['boundary_iou'] for m in values if m['boundary_iou'] is not None]
                summaries[split][name] = {**summarize(values), 'mean_boundary_iou': float(np.mean(defined)) if defined else None,
                                          'n_defined_boundary_iou': len(defined)}
        archived_summary = read_json(PROJECT / 'artifacts/part_c/supplementary_morphology/comparison.json')
        for split in ('validation', 'test'):
            for name, version in (('original', 'v1'), ('final', 'v2')):
                if summaries[split][name]['global'] != archived_summary['results'][f'{split}-{version}']['global']:
                    raise ValueError(f'Global metrics differ from archive: {split}/{name}')
        payload = {'records': records, 'summaries': summaries}
        template = Path(__file__).with_name('review_gallery.html').read_text(encoding='utf-8')
        (stage / 'index.html').write_text(template.replace('__GALLERY_DATA__', json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')), encoding='utf-8')
        write_json(stage / 'summary.json', summaries)
        write_json(stage / 'provenance.json', {'source_head': revision, 'tracked_worktree_changes': dirty,
                   'configs': configs, 'skin_params': params, 'n_images': len(records),
                   'archived_prediction_matches': archive_checks, 'archived_global_metrics_match': True,
                   'note': 'Fixed existing configurations; no parameter search. Train is descriptive, test is post-development comparison.'})
        with (stage / 'per_image.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            flat = [{'sample_id': r['id'], 'split': r['split'], 'delta_f1': r['delta_f1'],
                     **{f'{name}_{key}': value for name, values in r['metrics'].items() for key, value in values.items()}}
                    for r in records]
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
        (stage / 'README.md').write_text('# 全图像形态学对照图册\n\n双击 index.html 即可离线浏览。分享时复制整个目录。\n\n覆盖 98 张训练图、32 张验证图、32 张原测试图；默认展示验证集。使用现有 original/final 配置重新生成固定预测，没有选参。验证/测试的 128 张预测及四组总体指标与历史归档一致。训练集只作描述，原测试集是开发后比较。\n\nimages/ 为可单独分享的六宫格原尺寸 PNG；per_image.csv 为逐图指标，summary.json 区分总体像素指标与逐图均值，provenance.json 保存配置及代码来源。\n\n重新生成：在项目根目录执行 python tools/build_review_gallery.py --output results/新的目录名。\n', encoding='utf-8')
        stage.rename(output)
    return {'output': str(output), 'n_images': len(records), 'archived_prediction_matches': archive_checks}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New directory for a portable offline gallery')
    print(json.dumps(build(parser.parse_args().output), ensure_ascii=False))
