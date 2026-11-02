"""将原始类别标签转换为脸部真值，保持已有的数据划分。"""

import argparse
from collections import Counter
import os
from pathlib import Path
import re
import tempfile

import numpy as np
from PIL import Image

from .data_io import PROJECT, _read_csv, write_csv, write_json

POSITIVE_IDS = [1, 2, 4, 5, 6, 7, 10, 11, 12]
IGNORE_IDS = [3]


def binary_labels(labels):
    """皮肤、鼻子和五官为脸部，眼镜像素不参与评价。"""
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.dtype.kind not in 'ui' or not np.isin(labels, range(19)).all():
        raise ValueError('Expected a 2-D integer label image with IDs 0..18')
    face = np.isin(labels, POSITIVE_IDS).astype(np.uint8)
    valid = (~np.isin(labels, IGNORE_IDS)).astype(np.uint8)
    return face, valid


def prepare_dataset(manifest, output, source_root=None):
    """读取原始清单，输出 0/1 真值、有效区域及统一清单。"""
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    source_root = Path(source_root).resolve() if source_root else manifest.parent.parent
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    rows = _read_csv(manifest, {'sample_id', 'image_path', 'label_path', 'split'})
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared, seen_images = [], set()
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / 'data'
        (stage / 'masks').mkdir(parents=True)
        (stage / 'valid').mkdir()
        for row in rows:
            sid = row['sample_id']
            if not re.fullmatch(r'[A-Za-z0-9_-]+', sid):
                raise ValueError('Use letters, digits, underscores or hyphens in sample IDs')
            split = 'validation' if row['split'] == 'val' else row['split']
            if split not in {'train', 'validation', 'test'}:
                raise ValueError(f'Unknown split: {split}')
            image_path = (source_root / row['image_path']).resolve()
            label_path = (source_root / row['label_path']).resolve()
            if image_path in seen_images:
                raise ValueError(f'Image path reused: {image_path}')
            seen_images.add(image_path)
            with Image.open(image_path) as image, Image.open(label_path) as label:
                if image.size != label.size or image.mode != 'RGB':
                    raise ValueError(f'Image and label dimensions or mode differ: {sid}')
                face, valid = binary_labels(np.array(label))
            Image.fromarray(face).save(stage / 'masks' / f'{sid}.png')
            Image.fromarray(valid).save(stage / 'valid' / f'{sid}.png')
            prepared.append({
                'sample_id': sid,
                'image_path': os.path.relpath(image_path, output),
                'mask_path': f'masks/{sid}.png',
                'valid_mask_path': f'valid/{sid}.png',
                'source_label_path': os.path.relpath(label_path, output),
                'split': split,
            })
        write_csv(stage / 'manifest.csv', prepared)
        summary = {'counts': dict(Counter(r['split'] for r in prepared)),
                   'positive_ids': POSITIVE_IDS, 'ignore_ids': IGNORE_IDS,
                   'gt_encoding': 'binary01', 'valid_encoding': 'binary01'}
        write_json(stage / 'dataset.json', summary)
        stage.rename(output)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(PROJECT / 'data/split_manifest.csv'))
    parser.add_argument('--source-root')
    parser.add_argument('--output', default=str(PROJECT / 'data/processed/face'))
    args = parser.parse_args(argv)
    print(prepare_dataset(args.manifest, args.output, args.source_root))


if __name__ == '__main__':
    main()
