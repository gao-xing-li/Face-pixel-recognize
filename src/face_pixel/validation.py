"""B's validation and frozen image-only prediction commands; scores use C code."""

import argparse
import json
from pathlib import Path
import platform
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image

from .b_data import (PROJECT, digest, prepare, read_bgr, read_json, verify_inputs, write_csv, write_json)
from .data_io import read_binary_mask, read_dataset_manifest
from .detection import METHODS, detect
from .evaluation import evaluate, summarize


def code_hashes():
    return {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def runtime():
    return {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__}


def settings(k, method, kernel):
    return {"threshold": {"rule": "cbcr_box", "channels": ["Cb", "Cr"], "k": k, "inclusive": True},
            "preprocess": "none", "postprocess": {"method": method, "kernel": kernel},
            "evaluation_size": "original; no resizing"}


def _check_selected(row, adapter, include_labels):
    expected = next(r for r in adapter["samples"] if r["sample_id"] == row["sample_id"])
    if expected["split"] != row["split"] or digest(row["image_path"]) != expected["image_sha256"]:
        raise ValueError("image or split no longer matches adapter")
    if include_labels:
        if digest(row["mask_path"]) != expected["mask_sha256"] or digest(row["valid_mask_path"]) != expected["valid_sha256"]:
            raise ValueError("derived labels changed")


def validate(manifest, params_path, plan_path, output):
    manifest, params_path, plan_path, output = map(lambda p: Path(p).resolve(), (manifest, params_path, plan_path, output))
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    params, adapter = verify_inputs(manifest, params_path)
    plan = read_json(plan_path)
    if plan["split"] != "validation" or plan["metric"] != "global.f1":
        raise ValueError("selection is restricted to validation/global.f1")
    methods = plan["postprocess_values"]
    if not methods or methods[0] != "none" or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError("invalid postprocess grid")
    ks = plan["k_values"]
    if not ks or len(set(ks)) != len(ks) or any(type(k) not in (int, float) or not np.isfinite(k) or k <= 0 for k in ks):
        raise ValueError("invalid k grid")
    selected = [r for r in read_dataset_manifest(manifest) if r["split"] == "validation"]
    if not selected:
        raise ValueError("validation split is empty")
    loaded = []
    for row in selected:
        _check_selected(row, adapter, include_labels=True)
        loaded.append((row, read_bgr(row["image_path"]), read_binary_mask(row["mask_path"], "binary255"),
                       read_binary_mask(row["valid_mask_path"], "binary255")))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--", "src/face_pixel", "configs/validation_plan.json"], cwd=PROJECT, text=True).strip():
        raise ValueError("commit detector/evaluator/validation plan before the formal run for a stable code_revision")
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates, configs = [], {}
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".validate-") as temp:
        stage = Path(temp) / "run"
        stage.mkdir()
        for method in methods:
            for k in sorted(ks):
                name = f"candidate_{len(candidates):02d}"
                config = settings(k, method, plan["closing_kernel"])
                records = []
                for row, image, gt, valid in loaded:
                    records.append({"sample_id": row["sample_id"], **evaluate(detect(image, params, config), gt, valid)})
                summary = summarize(records)
                if summary["global"]["f1"] is None:
                    raise ValueError("undefined selection score")
                directory = stage / name
                directory.mkdir()
                write_csv(directory / "per_image.csv", records)
                write_json(directory / "summary.json", summary)
                candidates.append({"candidate": name, "k": k, "postprocess": method, **summary["global"],
                                   "image_mean_f1": summary["image_mean"]["f1"]})
                configs[name] = config
                print(f"{name}: k={k:g} post={method} F1={summary['global']['f1']:.6f}", flush=True)
        def rank(row):
            return (row["f1"], -methods.index(row["postprocess"]), -row["k"])
        best = max(candidates, key=rank)
        baseline = max((r for r in candidates if r["postprocess"] == "none"), key=rank)
        frozen = {}
        for label, chosen in (("baseline", baseline), ("final", best)):
            config = {"schema_version": 1, "config_id": f"part-b-v1-{label}", "synthetic_demo": False,
                      "code_revision": revision, "source_sha256": code_hashes(), "runtime": runtime(),
                      "data_manifest_sha256": digest(manifest), "source_manifest_sha256": adapter["source_manifest_sha256"],
                      "adapter_sha256": digest(manifest.parent / "adapter.json"),
                      "model_params": {"path": str(params_path.relative_to(PROJECT)), "sha256": digest(params_path)},
                      "labels": {"positive_ids": adapter["positive_ids"], "ignore_ids": adapter["ignore_ids"],
                                 "gt_encoding": "binary255", "valid_encoding": "binary255", "pred_encoding": "binary255"},
                      **configs[chosen["candidate"]],
                      "selection": {"fixed": True, "split": "validation", "metric": "global.f1", "score": chosen["f1"],
                                    "candidate": chosen["candidate"], "plan_sha256": digest(plan_path),
                                    "reason": "Maximum validation global F1 within the prespecified grid; " + plan["tie_break"],
                                    "test_used": False}}
            frozen[label] = config
            write_json(stage / f"{label}_config.json", config)
        write_csv(stage / "candidates.csv", candidates)
        write_json(stage / "selection.json", {"baseline": baseline, "final": best, "n_validation_images": len(loaded),
                                               "plan": plan, "test_evaluated": False})
        plot_curve(stage / "validation_curve.png", candidates, methods)
        stage.rename(output)
    return frozen


def plot_curve(path, rows, methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, metric in zip(axes, ("f1", "iou")):
        for method in methods:
            group = [r for r in rows if r["postprocess"] == method]
            ax.plot([r["k"] for r in group], [r[metric] for r in group], marker="o", label=method)
        ax.set(xlabel="k", ylabel=f"Global {metric.upper()}", title="32 validation images; test not evaluated")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def predict(manifest, config_path, split, output):
    if split not in ("validation", "test"):
        raise ValueError("prediction split must be validation or test")
    manifest, config_path, output = map(lambda p: Path(p).resolve(), (manifest, config_path, output))
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    config = read_json(config_path)
    if config.get("synthetic_demo") is not False or not config.get("selection", {}).get("fixed"):
        raise ValueError("requires a frozen real-data configuration")
    if config["selection"].get("split") != "validation" or config["selection"].get("test_used") is not False:
        raise ValueError("configuration was not selected independently of test")
    if config["source_sha256"] != code_hashes():
        raise ValueError("code differs from frozen configuration; use its code_revision")
    params_path = PROJECT / config["model_params"]["path"]
    params, adapter = verify_inputs(manifest, params_path)
    if digest(params_path) != config["model_params"]["sha256"] or digest(manifest) != config["data_manifest_sha256"] or digest(manifest.parent / "adapter.json") != config["adapter_sha256"]:
        raise ValueError("configuration input hash mismatch")
    selected = [r for r in read_dataset_manifest(manifest) if r["split"] == split]
    if not selected:
        raise ValueError("empty split")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".predict-") as temp:
        stage = Path(temp) / "run"
        (stage / "predictions").mkdir(parents=True)
        rows = []
        for row in selected:
            _check_selected(row, adapter, include_labels=False)
            image = read_bgr(row["image_path"])
            mask = detect(image, params, config)
            relative = f"predictions/{row['sample_id']}.png"
            Image.fromarray(mask * 255).save(stage / relative)
            rows.append({"sample_id": row["sample_id"], "pred_path": relative})
        write_csv(stage / "predictions.csv", rows)
        (stage / "run_config.json").write_bytes(config_path.read_bytes())
        write_json(stage / "run.json", {"split": split, "config_id": config["config_id"], "n_images": len(rows),
                                       "config_sha256": digest(config_path), "runtime": runtime(),
                                       "ground_truth_read": False, "metrics_computed": False})
        stage.rename(output)
    return {"output": str(output), "n_images": len(rows), "split": split}


def main(argv=None):
    parser = argparse.ArgumentParser(description="B: adapt A data, validate on validation only, and predict with fixed configurations")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", default="data/processed/part_b")
    val = commands.add_parser("validate")
    val.add_argument("--manifest", default="data/processed/part_b/manifest.csv")
    val.add_argument("--params", default="configs/skin_params.json")
    val.add_argument("--plan", default="configs/validation_plan.json")
    val.add_argument("--output", required=True)
    pred = commands.add_parser("predict")
    pred.add_argument("--manifest", default="data/processed/part_b/manifest.csv")
    pred.add_argument("--config", required=True)
    pred.add_argument("--split", choices=("validation", "test"), required=True)
    pred.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            report = prepare(args.output)
            result = {"output": args.output, "counts": report["counts"], "derived_manifest_sha256": report["derived_manifest_sha256"]}
        elif args.command == "validate":
            frozen = validate(args.manifest, args.params, args.plan, args.output)
            result = {name: conf["selection"] for name, conf in frozen.items()}
        else:
            result = predict(args.manifest, args.config, args.split, args.output)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"B pipeline failed: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
