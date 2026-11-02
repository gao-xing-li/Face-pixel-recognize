"""在同一项目中比较原方法和形态学补全。"""
import _bootstrap
from face_pixel.data_io import PROJECT
from face_pixel.experiment import run_experiment
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(PROJECT / "data/processed/face/manifest.csv"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_experiment(args.manifest, args.output,
                   configs=[PROJECT / "configs/original_config.json", PROJECT / "configs/final_config.json"])
