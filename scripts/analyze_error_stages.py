"""Explain frozen B results with intermediate masks and original semantic labels.

This is post-test diagnosis of an unchanged pipeline. It does not select a new
configuration or evaluate a new independent test set.
"""

import argparse
from collections import defaultdict
import csv
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import tempfile

import cv2
import numpy as np
from PIL import Image

from face_pixel.data_io import read_dataset_manifest
from face_pixel.evaluation import evaluate, summarize
from face_pixel.visualization import _collect, _font_path


STAGES = ("baseline_k150", "threshold_k125", "largest_k125", "final_fill")
TITLES = ("基线阈值 k=1.50", "颜色阈值 k=1.25", "保留最大连通域", "填孔后的最终结果")
ZH = {0: "背景", 1: "皮肤", 2: "鼻子", 3: "眼镜，忽略", 4: "左眼", 5: "右眼",
      6: "左眉", 7: "右眉", 8: "左耳", 9: "右耳", 10: "口腔", 11: "上唇",
      12: "下唇", 13: "头发", 14: "帽子", 15: "耳饰", 16: "项链", 17: "脖颈", 18: "衣服"}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def transitions(before, after, gt, valid):
    before, after, gt, valid = (value.astype(bool) for value in (before, after, gt, valid))
    removed, added = before & ~after & valid, ~before & after & valid
    return {
        "removed_tp": int((removed & gt).sum()),
        "removed_fp": int((removed & ~gt).sum()),
        "added_tp": int((added & gt).sum()),
        "added_fp": int((added & ~gt).sum()),
    }


def check_transition(before, after, change):
    assert after["tp"] - before["tp"] == change["added_tp"] - change["removed_tp"]
    assert after["fp"] - before["fp"] == change["added_fp"] - change["removed_fp"]
    assert after["fn"] - before["fn"] == change["removed_tp"] - change["added_tp"]


def make_figures(entries, metrics, labels, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font = FontProperties(fname=_font_path())
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False,
                         "font.size": 11, "figure.facecolor": "white"})
    colors = ("#B87432", "#667085", "#4C74A8", "#087E8B")
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.6), layout="constrained")
    x = np.arange(4)
    for key, title, color in (("fp", "误检 FP", "#DA645B"), ("fn", "漏检 FN", "#507BC3")):
        offset = -.18 if key == "fp" else .18
        values = [metrics[s]["global"][key] / 10000 for s in STAGES]
        bars = axes[0].bar(x + offset, values, .34, label=title, color=color)
        axes[0].bar_label(bars, fmt="%.1f", padding=3, fontsize=10)
    axes[0].set(ylabel="像素数，万", title="各阶段分别保留了多少误检与漏检", xticks=x, xticklabels=TITLES)
    axes[0].tick_params(axis="x", labelrotation=12)
    axes[0].legend(frameon=False)
    for key, title, marker in (("precision", "Precision", "o"), ("recall", "Recall", "s"), ("f1", "F1", "D")):
        values = np.array([metrics[s]["global"][key] for s in STAGES]) * 100
        axes[1].plot(x, values, marker=marker, label=title)
    axes[1].set(ylabel="%", title="同一批测试图上的流程诊断", xticks=x, xticklabels=TITLES)
    axes[1].tick_params(axis="x", labelrotation=12)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("固定流程的阶段变化，未重新选择参数", fontsize=16)
    fig.savefig(out / "stage_effects.png", dpi=180)
    fig.savefig(out / "stage_effects.svg")
    plt.close(fig)

    chosen = [next(e for e in entries if e["sample_id"].endswith(sid)) for sid in ("00610", "00609", "00718")]
    fig, axes = plt.subplots(3, 6, figsize=(17, 10), layout="constrained")
    for i, entry in enumerate(chosen):
        for j, (title, picture) in enumerate((("原图", entry["rgb"]), ("人工脸部真值", entry["gt"]),
                                             *[(TITLES[n], entry["stages"][stage]) for n, stage in enumerate(STAGES)])):
            ax = axes[i, j]
            if j == 0:
                ax.imshow(picture)
            else:
                ax.imshow(picture, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            if i == 0:
                ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            if j >= 2:
                m = entry["stage_metrics"][STAGES[j - 2]]
                ax.set_xlabel(f"F1 {m['f1']:.3f}   R {m['recall']:.3f}")
            if j == 0:
                ax.set_ylabel(f"{entry['sample_id'].rsplit('_', 1)[-1]}\n{('成功', '严重漏检', '明显误检')[i]}", fontsize=12)
    fig.suptitle("同一张图在各阶段如何变化，所有面板保持原尺寸比例", fontsize=17)
    fig.savefig(out / "case_stage_comparison.png", dpi=160)
    plt.close(fig)

    negative = sorted([v for v in labels if v["role"] == "negative" and v["n_pixels"]],
                      key=lambda v: v["final_fill_predicted_positive"], reverse=True)
    fig, ax = plt.subplots(figsize=(9.4, 4.6), layout="constrained")
    xs = np.arange(len(negative))
    for stage, offset, color in (("baseline_k150", -.2, colors[0]), ("final_fill", .2, colors[-1])):
        values = [v[f"{stage}_predicted_positive"] / 10000 for v in negative]
        bars = ax.bar(xs + offset, values, .37, label="基线" if stage == "baseline_k150" else "最终", color=color)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set(xticks=xs, xticklabels=[v["label_zh"] for v in negative], ylabel="误检像素数，万",
           title="误检来自哪些原始标签类别")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    fig.savefig(out / "false_positive_sources.png", dpi=180)
    fig.savefig(out / "false_positive_sources.svg")
    plt.close(fig)


def diagnose(manifest, run, b_root, output):
    manifest, run, b_root, output = (Path(p).resolve() for p in (manifest, run, b_root, output))
    if output.exists():
        raise ValueError("use a new diagnosis output directory")
    entries, frozen = _collect(manifest, run)
    if len(entries) != 32:
        raise ValueError("expected the recorded 32-image test split")
    configs = {name: load_json(b_root / f"configs/{name}_config.json") for name in ("baseline", "final")}
    for name, config in configs.items():
        if digest(b_root / f"configs/{name}_config.json") != frozen["b_config_sha256"][name]:
            raise ValueError("frozen configuration changed")
        for file, expected in config["source_sha256"].items():
            if digest(b_root / "src/face_pixel" / file) != expected:
                raise ValueError(f"frozen B source changed: {file}")
    if configs["baseline"]["threshold"]["k"] != 1.5 or configs["baseline"]["postprocess"]["method"] != "none":
        raise ValueError("baseline pipeline differs from the diagnosis design")
    if configs["final"]["threshold"]["k"] != 1.25 or configs["final"]["postprocess"]["method"] != "largest_fill":
        raise ValueError("final pipeline differs from the diagnosis design")
    runtime = {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__}
    if runtime != configs["final"]["runtime"]:
        raise ValueError("use B's locked runtime")
    spec = importlib.util.spec_from_file_location("frozen_b_detection_for_diagnosis", b_root / "src/face_pixel/detection.py")
    detector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(detector)
    params = load_json(b_root / "configs/skin_params.json")
    if digest(b_root / "configs/skin_params.json") != configs["final"]["model_params"]["sha256"]:
        raise ValueError("A statistics changed")
    detector.validate_model(params)
    audit = load_json(manifest.parent / "audit.json")
    mapping = audit["mapping"]
    source_rows = {v["sample_id"]: v for v in read_dataset_manifest(manifest) if v["split"] == "test"}
    stages = defaultdict(list)
    label_totals = np.zeros(19, dtype=np.int64)
    label_predictions = {stage: np.zeros(19, dtype=np.int64) for stage in STAGES}
    changes = defaultdict(lambda: defaultdict(int))
    input_hashes, per_image = [], []
    candidate_config = deepcopy(configs["final"])
    candidate_config["postprocess"]["method"] = "none"
    for entry in entries:
        sid, gt, valid = entry["sample_id"], entry["gt"], entry["valid"]
        image_bgr = np.ascontiguousarray(entry["rgb"][:, :, ::-1])
        baseline = detector.detect(image_bgr, params, configs["baseline"])
        raw = detector.detect(image_bgr, params, candidate_config)
        component_count, components, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
        largest = np.zeros_like(raw)
        if component_count > 1:
            largest_id = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            largest = (components == largest_id).astype(np.uint8)
        final = detector.fill_holes(largest)
        assert np.array_equal(final, detector.detect(image_bgr, params, configs["final"]))
        assert np.array_equal(baseline, entry["baseline"]), f"baseline differs: {sid}"
        assert np.array_equal(final, entry["final"]), f"final differs: {sid}"
        assert np.all(raw <= baseline) and np.all(largest <= raw) and np.all(largest <= final)
        entry["stages"] = dict(zip(STAGES, (baseline, raw, largest, final)))
        entry["stage_metrics"] = {}
        entry["raw_component_count"] = component_count - 1
        for stage, mask in entry["stages"].items():
            metric = evaluate(mask, gt, valid)
            stages[stage].append(metric)
            entry["stage_metrics"][stage] = metric
            per_image.append({"sample_id": sid, "stage": stage, "foreground_pixels": int(mask.sum()), **metric})
        entry["transitions"] = {}
        for before, after in zip(STAGES[:-1], STAGES[1:]):
            change = transitions(entry["stages"][before], entry["stages"][after], gt, valid)
            key = f"{before}_to_{after}"
            entry["transitions"][key] = change
            check_transition(entry["stage_metrics"][before], entry["stage_metrics"][after], change)
            for name, value in change.items():
                changes[key][name] += value
        row = source_rows[sid]
        label_path = (manifest.parent / row["source_label_path"]).resolve()
        label_sha = digest(label_path)
        if label_sha != row["source_label_sha256"]:
            raise ValueError(f"raw semantic label changed: {sid}")
        with Image.open(label_path) as image:
            labels = np.array(image)
        assert labels.shape == gt.shape and set(np.unique(labels)).issubset(set(range(19)))
        assert np.array_equal(np.isin(labels, mapping["face_positive_ids"]).astype(np.uint8), gt)
        assert np.array_equal((~np.isin(labels, mapping["evaluation_ignore_ids"])).astype(np.uint8), valid)
        label_totals += np.bincount(labels.ravel(), minlength=19)
        for stage, mask in entry["stages"].items():
            label_predictions[stage] += np.bincount(labels[mask.astype(bool)], minlength=19)
        input_hashes.append({"sample_id": sid, "source_label_sha256": label_sha})
    summaries = {stage: summarize(stages[stage]) for stage in STAGES}
    for stage, original_name in ((STAGES[0], "baseline"), (STAGES[-1], "final")):
        assert summaries[stage]["global"] == frozen["results"][f"test-{original_name}"]["global"]
    for before, after in zip(STAGES[:-1], STAGES[1:]):
        check_transition(summaries[before]["global"], summaries[after]["global"], changes[f"{before}_to_{after}"])
    label_rows = []
    for label in range(19):
        role = "positive" if label in mapping["face_positive_ids"] else "ignored" if label in mapping["evaluation_ignore_ids"] else "negative"
        row = {"label_id": label, "label": mapping["label_map"][str(label)], "label_zh": ZH[label],
               "role": role, "n_pixels": int(label_totals[label])}
        for stage in STAGES:
            selected = int(label_predictions[stage][label])
            row[f"{stage}_predicted_positive"] = selected
            row[f"{stage}_prediction_rate"] = selected / row["n_pixels"] if row["n_pixels"] else None
            row[f"{stage}_share_of_fp"] = selected / summaries[stage]["global"]["fp"] if role == "negative" and summaries[stage]["global"]["fp"] else None
        label_rows.append(row)
    for stage in STAGES:
        assert sum(r[f"{stage}_predicted_positive"] for r in label_rows if r["role"] == "positive") == summaries[stage]["global"]["tp"]
        assert sum(r[f"{stage}_predicted_positive"] for r in label_rows if r["role"] == "negative") == summaries[stage]["global"]["fp"]
    details = [{"sample_id": e["sample_id"], "raw_component_count": e["raw_component_count"],
                "foreground_pixels": {k: int(v.sum()) for k, v in e["stages"].items()},
                "stage_metrics": e["stage_metrics"], "transitions": e["transitions"]} for e in entries]
    report = {
        "scope": "Post-test mechanism diagnosis of a fixed pipeline; no new model selection or independent test set",
        "n_images": len(entries), "stage_order": list(STAGES), "stage_titles": dict(zip(STAGES, TITLES)),
        "global_and_image_metrics": summaries, "pixel_transitions": dict(changes),
        "label_analysis": label_rows, "per_image": details,
        "provenance": {"b_revision": frozen["b_revision"], "manifest_sha256": digest(manifest),
                       "original_independent_test_sha256": digest(run / "independent_test.json"),
                       "b_config_sha256": frozen["b_config_sha256"], "b_source_sha256": frozen["b_source_sha256"],
                       "diagnosis_script_sha256": digest(__file__), "raw_label_inputs": input_hashes, "runtime": runtime},
        "checks": {"all_32_baseline_masks_identical_to_frozen": True, "all_32_final_masks_identical_to_frozen": True,
                   "raw_labels_match_c_binary_ground_truth": True, "per_class_counts_reconcile_to_tp_fp": True,
                   "stage_transitions_reconcile_to_global_counts": True},
        "limitations": ["Sequential differences depend on the specified order and interacting steps; they are not additive causal effects of independently randomized treatments",
                        "Intermediate masks and semantic subgroups are post-test descriptions, not candidate configurations selected for deployment",
                        "Semantic region labels are used only to score errors, never to construct predictions",
                        "A contains only 32 fixed, appearance-selected test images; estimates do not establish general population performance"]
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".error-diagnosis-", dir=output.parent) as temporary:
        stage_dir = Path(temporary) / "ready"
        stage_dir.mkdir()
        (stage_dir / "diagnosis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_csv(stage_dir / "stage_metrics.csv", [{"stage": s, "title": TITLES[i], **summaries[s]["global"]} for i, s in enumerate(STAGES)])
        write_csv(stage_dir / "per_image_stages.csv", per_image)
        write_csv(stage_dir / "semantic_region_errors.csv", label_rows)
        make_figures(entries, summaries, label_rows, stage_dir)
        (stage_dir / "README.md").write_text("# 固定流程的事后错误诊断\n\n四个阶段依次为基线阈值、最终颜色阈值、最大连通域和填孔。原有 32 张测试图的基线与最终预测均逐像素核对一致。中间阶段只用于解释，不构成新方案选参。语义标签仅用于统计错误来源。\n\n数值见 diagnosis.json 和三份 CSV，stage_effects 看误检与漏检变化，case_stage_comparison 看具体样例，false_positive_sources 看误检原始类别。\n", encoding="utf-8")
        stage_dir.rename(output)
    print(json.dumps({"output": str(output), "n_images": len(entries),
                      "stages": {s: summaries[s]["global"] for s in STAGES}, "transitions": dict(changes),
                      "checks": report["checks"]}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--b-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    diagnose(args.manifest, args.run, args.b_root, args.output)


if __name__ == "__main__":
    main()
