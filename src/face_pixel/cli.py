"""Command line entry point for scoring saved validation/test predictions."""

import argparse
from pathlib import Path

from .batch import evaluate_predictions


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved binary skin masks with strict pairing.")
    parser.add_argument("--manifest", required=True, help="A's global data/split_manifest.csv")
    parser.add_argument("--predictions", required=True, help="B's sample_id,pred_path CSV for one split")
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--run-config", help="Actual JSON configuration with matching config_id; required by the formal-test handoff")
    parser.add_argument("--output", required=True, help="A new directory; existing results are never overwritten")
    for name in ("gt", "pred", "valid"):
        parser.add_argument(f"--{name}-encoding", choices=("binary01", "binary255"), default="binary01")
    args = parser.parse_args(argv)
    try:
        summary = evaluate_predictions(
            args.manifest, args.predictions, args.output,
            split=args.split, config_id=args.config_id, run_config_path=args.run_config,
            gt_encoding=args.gt_encoding, pred_encoding=args.pred_encoding, valid_encoding=args.valid_encoding,
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")
    print(f"Evaluated {summary['n_images']} images. Results: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
