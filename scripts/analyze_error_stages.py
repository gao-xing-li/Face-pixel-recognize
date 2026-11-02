"""按颜色阈值、最大区域、填孔拆开计算误检和漏检。"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from face_pixel.data_io import read_bgr, read_binary_mask, read_dataset_manifest, read_json, write_csv, write_json, PROJECT
from face_pixel.detection import detect, fill_holes
from face_pixel.evaluation import evaluate, summarize
from face_pixel.validation import settings


def diagnose(manifest, output, params_path):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new output directory')
    params = read_json(params_path)
    scores = {name: [] for name in ('baseline_k150', 'threshold_k125', 'largest_k125', 'final_fill')}
    per_image, semantic = [], []
    label_counts = {name: np.zeros(19, dtype=np.int64) for name in scores}
    for row in read_dataset_manifest(manifest):
        if row['split'] != 'test':
            continue
        image = read_bgr(row['image_path'])
        gt = read_binary_mask(row['mask_path'])
        valid = read_binary_mask(row['valid_mask_path']) if row.get('valid_mask_path') else np.ones_like(gt)
        baseline = detect(image, params, settings(1.5, 'none'))
        raw = detect(image, params, settings(1.25, 'none'))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
        largest = np.zeros_like(raw)
        if count > 1:
            selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            largest = (labels == selected).astype(np.uint8)
        stages = dict(zip(scores, (baseline, raw, largest, fill_holes(largest))))
        source_label = np.array(Image.open(Path(manifest).parent / row['source_label_path']))
        for name, mask in stages.items():
            result = evaluate(mask, gt, valid)
            scores[name].append(result)
            per_image.append({'sample_id': row['sample_id'], 'stage': name, **result})
            label_counts[name] += np.bincount(source_label[(mask & valid).astype(bool)], minlength=19)
    summary = {name: summarize(rows) for name, rows in scores.items()}
    for label in range(19):
        semantic.append({'label_id': label, **{name: int(counts[label]) for name, counts in label_counts.items()}})
    output.mkdir(parents=True)
    write_json(output / 'diagnosis.json', summary)
    write_csv(output / 'stage_metrics.csv', [{'stage': name, **score['global']} for name, score in summary.items()])
    write_csv(output / 'per_image_stages.csv', per_image)
    write_csv(output / 'semantic_region_errors.csv', semantic)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(PROJECT / 'data/processed/face/manifest.csv'))
    parser.add_argument('--params', default=str(PROJECT / 'configs/skin_params.json'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(diagnose(args.manifest, args.output, args.params))
