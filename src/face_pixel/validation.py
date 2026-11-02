"""在验证集选择参数，根据固定配置生成预测。"""

import argparse
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from .data_io import (PROJECT, read_bgr, read_json, write_csv, write_json,
                      read_binary_mask, read_dataset_manifest)
from .detection import METHODS, detect, validate_model
from .evaluation import evaluate, summarize


def settings(k, method, kernel=5):
    return {'threshold': {'rule': 'cbcr_box', 'channels': ['Cb', 'Cr'], 'k': k},
            'preprocess': 'none', 'evaluation_size': 'original; no resizing',
            'postprocess': {'method': method, 'kernel': kernel}}


def select_candidates(candidates, plan):
    """按 F1，或在 F1 损失预算内按召回率选参。"""
    methods = plan['postprocess_values']

    def f1_rank(row):
        return row['f1'], -methods.index(row['postprocess']), -row['kernel'], -row['k']

    best_f1 = max(candidates, key=f1_rank)
    baseline = max((r for r in candidates if r['postprocess'] == 'none'), key=f1_rank)
    if plan['metric'] == 'global.f1':
        return baseline, best_f1, {'best_f1_candidate': best_f1['candidate']}
    budget = plan['max_f1_drop']
    if not np.isfinite(budget) or not 0 <= budget <= 1:
        raise ValueError('F1 budget must be in [0, 1]')
    reference = max((r for r in candidates if r['postprocess'] == 'largest_fill'), key=f1_rank)
    floor = reference['f1'] - budget
    eligible = [r for r in candidates if r['postprocess'] != 'none' and r['f1'] >= floor]
    chosen = max(eligible, key=lambda r: (r['recall'], *f1_rank(r)))
    return baseline, chosen, {'reference_f1': reference['f1'], 'minimum_f1': floor,
                              'best_f1_candidate': best_f1['candidate']}


def validate(manifest, params_path, plan_path, output):
    params, plan = read_json(params_path), read_json(plan_path)
    validate_model(params)
    if plan['split'] != 'validation' or plan['metric'] not in ('global.f1', 'global.recall_with_f1_budget'):
        raise ValueError('Parameter selection uses validation only')
    methods, ks = plan['postprocess_values'], plan['k_values']
    if not methods or methods[0] != 'none' or any(m not in METHODS for m in methods):
        raise ValueError('Invalid postprocessing grid')
    if not ks or any(type(k) not in (int, float) or not np.isfinite(k) or k <= 0 for k in ks):
        raise ValueError('Invalid threshold grid')
    kernels = plan.get('kernel_values', [plan['closing_kernel']])
    if not kernels or any(type(k) is not int or k < 1 or k % 2 == 0 for k in kernels):
        raise ValueError('Kernels must be odd positive integers')
    rows = [r for r in read_dataset_manifest(manifest) if r['split'] == 'validation']
    if not rows:
        raise ValueError('Empty validation split')
    # 只加载验证图像和标签，测试标签不参与参数选择。
    loaded = [(r, read_bgr(r['image_path']), read_binary_mask(r['mask_path']),
               read_binary_mask(r['valid_mask_path']) if r.get('valid_mask_path') else None) for r in rows]
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates, configurations = [], {}
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / 'validation'
        stage.mkdir()
        for method in methods:
            method_kernels = sorted(kernels) if method in ('largest_close_fill', 'largest_dilate_fill_erode') else [plan['closing_kernel']]
            for k in sorted(ks):
                for kernel in method_kernels:
                    name = f'candidate_{len(candidates):02d}'
                    config = settings(k, method, kernel)
                    scores = [{'sample_id': r['sample_id'], **evaluate(detect(image, params, config), gt, valid)}
                              for r, image, gt, valid in loaded]
                    summary = summarize(scores)
                    directory = stage / name
                    directory.mkdir()
                    write_csv(directory / 'per_image.csv', scores)
                    write_json(directory / 'summary.json', summary)
                    candidates.append({'candidate': name, 'k': k, 'postprocess': method, 'kernel': kernel,
                                       **summary['global'], 'image_mean_f1': summary['image_mean']['f1']})
                    configurations[name] = config
        baseline, final, policy = select_candidates(candidates, plan)
        selected = {}
        for label, choice in (('baseline', baseline), ('final', final)):
            selected[label] = {'config_id': label, **configurations[choice['candidate']],
                               'selection': {'split': 'validation', 'metric': plan['metric'], 'test_used': False}}
            write_json(stage / f'{label}_config.json', selected[label])
        write_csv(stage / 'candidates.csv', candidates)
        write_json(stage / 'selection.json', {'baseline': baseline, 'final': final, 'policy': policy,
                                             'n_validation_images': len(rows)})
        plot_curve(stage / "validation_curve.png", candidates, methods, len(rows))
        stage.rename(output)
    return selected


def plot_curve(path, rows, methods, n_images=32):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kernel_search = any(len({r["kernel"] for r in rows if r["postprocess"] == method}) > 1 for method in methods)
    metrics = ("f1", "iou", "recall") if kernel_search else ("f1", "iou")
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 4), layout="constrained")
    for ax, metric in zip(axes, metrics):
        for method in methods:
            group = [r for r in rows if r["postprocess"] == method]
            if kernel_search and len({r["kernel"] for r in group}) == 1:
                ax.axhline(group[0][metric], linestyle="--", label=method)
            else:
                ax.plot([r["kernel"] if kernel_search else r["k"] for r in group], [r[metric] for r in group], marker="o", label=method)
        ax.set(xlabel="Morphology kernel (pixels)" if kernel_search else "k", ylabel=f"Global {metric.upper()}", title=f"{n_images} validation images; test scores not used")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)



def predict(manifest, config_path, split, output, params_path=None):
    """检测只读取原图和参数，真值在评价阶段单独读取。"""
    config = read_json(config_path)
    params = read_json(params_path or PROJECT / 'configs/skin_params.json')
    validate_model(params)
    rows = [r for r in read_dataset_manifest(manifest) if r['split'] == split]
    if not rows:
        raise ValueError(f'Empty split: {split}')
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / 'prediction'
        (stage / 'predictions').mkdir(parents=True)
        predictions = []
        for row in rows:
            sid = row['sample_id']
            if Path(sid).name != sid or any(c in sid for c in '<>:"/\\|?*') or sid in ('.', '..'):
                raise ValueError('Invalid sample ID')
            mask = detect(read_bgr(row['image_path']), params, config)
            relative = f'predictions/{sid}.png'
            Image.fromarray(mask * 255).save(stage / relative)
            predictions.append({'sample_id': sid, 'pred_path': relative})
        write_csv(stage / 'predictions.csv', predictions)
        write_json(stage / 'run_config.json', config)
        stage.rename(output)
    return {'n_images': len(rows), 'split': split}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('validate', 'predict'):
        command = commands.add_parser(name)
        command.add_argument('--manifest', default=str(PROJECT / 'data/processed/face/manifest.csv'))
        command.add_argument('--params', default=str(PROJECT / 'configs/skin_params.json'))
        command.add_argument('--output', required=True)
        if name == 'validate':
            command.add_argument('--plan', default=str(PROJECT / 'configs/validation_plan.json'))
        else:
            command.add_argument('--config', required=True)
            command.add_argument('--split', choices=('validation', 'test'), required=True)
    args = parser.parse_args(argv)
    if args.command == 'validate':
        result = validate(args.manifest, args.params, args.plan, args.output)
    else:
        result = predict(args.manifest, args.config, args.split, args.output, args.params)
    print(result)


if __name__ == '__main__':
    main()
