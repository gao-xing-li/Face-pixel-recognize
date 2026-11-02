"""统一运行检测、评价和方法对比。"""

import argparse
from pathlib import Path

from .batch import evaluate_predictions
from .data_io import PROJECT, read_json, write_json
from .evaluation import METRIC_NAMES
from .validation import predict


def run_experiment(manifest, output, *, configs=None, params=None, splits=('validation', 'test')):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    configs = configs or [PROJECT / 'configs' / f'{name}_config.json' for name in ('baseline', 'original', 'final')]
    named = [(read_json(p)['config_id'], Path(p)) for p in configs]
    if len({name for name, _ in named}) != len(named):
        raise ValueError('Use a different config_id for each method')
    if any(not name.replace('-', '').replace('_', '').isalnum() for name, _ in named):
        raise ValueError('Use simple config IDs for output directory names')
    results = {}
    for split in splits:
        for name, config_path in named:
            key = f'{split}-{name}'
            predictions = output / 'predictions' / key
            predict(manifest, config_path, split, predictions, params)
            summary = evaluate_predictions(manifest, predictions / 'predictions.csv', output / 'evaluation' / key,
                                           split=split, config_id=name, run_config_path=config_path,
                                           pred_encoding='binary255')
            results[key] = {k: summary[k] for k in ('n_images', 'global', 'image_mean')}
            print(f"{key}: F1={summary['global']['f1']:.6f}, IoU={summary['global']['iou']:.6f}", flush=True)
    comparison = {}
    if len(named) >= 2:
        before, after = named[-2][0], named[-1][0]
        for split in splits:
            comparison[split] = {m: results[f'{split}-{after}']['global'][m] - results[f'{split}-{before}']['global'][m]
                                 for m in METRIC_NAMES}
    report = {'methods': [name for name, _ in named], 'results': results,
              'comparison': comparison, 'comparison_methods': [name for name, _ in named[-2:]],
              'parameters': {name: read_json(p) for name, p in named}}
    write_json(output / 'experiment.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(PROJECT / 'data/processed/face/manifest.csv'))
    parser.add_argument('--params', default=str(PROJECT / 'configs/skin_params.json'))
    parser.add_argument('--configs', nargs='+', help='按顺序比较的配置，默认运行基线、原方法、补全方法')
    parser.add_argument('--split', choices=('validation', 'test'), help='默认分别运行验证和测试')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    run_experiment(args.manifest, args.output, configs=args.configs, params=args.params,
                   splits=(args.split,) if args.split else ('validation', 'test'))


if __name__ == '__main__':
    main()
