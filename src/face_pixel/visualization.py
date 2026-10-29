"""Report-ready figures and supplementary assessment of immutable predictions."""

import argparse
import csv
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .assessment import boundary_iou, paired_bootstrap_f1, paired_mean_interval
from .data_io import read_binary_mask, read_dataset_manifest, read_prediction_manifest
from .evaluation import COUNT_NAMES, METRIC_NAMES, evaluate, summarize
from .part_a import _digest, _json, _write_json

BASE_COLOR, FINAL_COLOR = "#B87432", "#087E8B"


def _font_path(requested=None):
    if requested:
        if not Path(requested).is_file():
            raise ValueError("font file not found")
        return str(Path(requested).resolve())
    for path in ("C:/Windows/Fonts/msyh.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if Path(path).is_file():
            return path
    raise ValueError("provide --font with a TrueType/OpenType font supporting Chinese")


def _collect(manifest, run):
    import cv2

    frozen = _json(run / "independent_test.json")
    if _digest(manifest) != frozen["c_manifest_sha256"]:
        raise ValueError("manifest differs from the independently evaluated run")
    source_rows = [row for row in read_dataset_manifest(manifest) if row["split"] == "test"]
    expected_ids = {row["sample_id"] for row in source_rows}
    predictions, references = {}, {}
    for name in ("baseline", "final"):
        predictions[name] = read_prediction_manifest(run / f"predictions/test-{name}/predictions.csv")
        summary = _json(run / f"evaluation/test-{name}/summary.json")
        references[name] = {row["sample_id"]: row for row in summary["provenance"]["inputs"]}
        if set(predictions[name]) != expected_ids or set(references[name]) != expected_ids:
            raise ValueError("test sample IDs differ from the original evaluation")
    entries = []
    for row in source_rows:
        sid = row["sample_id"]
        if sid in {".", ".."} or any(char in sid for char in '<>:"/\\|?*'):
            raise ValueError("sample ID is not a safe figure filename")
        gt = read_binary_mask(row["mask_path"])
        valid = read_binary_mask(row["valid_mask_path"]) if row.get("valid_mask_path") else np.ones_like(gt)
        with Image.open(row["image_path"]) as image:
            rgb = np.array(image.convert("RGB"))
        if rgb.shape[:2] != gt.shape:
            raise ValueError("original image and labels are not aligned")
        entry = {"sample_id": sid, "rgb": rgb, "gt": gt, "valid": valid, "metrics": {}, "boundaries": {}}
        for name in ("baseline", "final"):
            reference = references[name][sid]
            paths = {"image_sha256": row["image_path"], "gt_sha256": row["mask_path"], "pred_sha256": predictions[name][sid]}
            if row.get("valid_mask_path"):
                paths["valid_sha256"] = row["valid_mask_path"]
            if any(_digest(path) != reference[key] for key, path in paths.items()):
                raise ValueError(f"input bytes changed since the independent test: {sid}/{name}")
            pred = read_binary_mask(predictions[name][sid], "binary255")
            entry[name] = pred
            entry["metrics"][name] = evaluate(pred, gt, valid)
            entry["boundaries"][name] = boundary_iou(pred, gt, valid)
        face = (gt & valid).astype(bool)
        y = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0]
        entry["face_mean_y"] = float(y[face].mean()) if face.any() else None
        entry["face_fraction"] = float(face.sum() / valid.sum())
        entries.append(entry)
    for name in ("baseline", "final"):
        calculated = summarize(entry["metrics"][name] for entry in entries)
        if any(calculated["global"][key] != frozen["results"][f"test-{name}"]["global"][key] for key in COUNT_NAMES):
            raise ValueError("recomputed pixel counts differ from the frozen test")
    return entries, frozen


def _cutout(entry, *, transparent=False):
    rgb, mask = entry["rgb"], entry["final"].astype(bool)
    if transparent:
        return Image.fromarray(np.dstack((rgb, mask.astype(np.uint8) * 255)), "RGBA")
    yy, xx = np.indices(mask.shape)
    background = np.where(((yy // 16 + xx // 16) % 2)[..., None] == 0, 231, 247)
    return Image.fromarray(np.where(mask[..., None], rgb, background).astype(np.uint8))


def _error_overlay(entry):
    rgb, gt, pred, valid = entry["rgb"], entry["gt"].astype(bool), entry["final"].astype(bool), entry["valid"].astype(bool)
    canvas = rgb.astype(float) * 0.35
    for condition, color in ((pred & gt & valid, (35, 175, 95)), (pred & ~gt & valid, (235, 66, 65)), (~pred & gt & valid, (45, 105, 240)), (~valid, (255, 195, 0))):
        canvas[condition] = rgb[condition] * 0.28 + np.array(color) * 0.72
    return Image.fromarray(canvas.clip(0, 255).astype(np.uint8))


def _panel(entry, kind):
    if kind == "original":
        return Image.fromarray(entry["rgb"])
    if kind == "cutout":
        return _cutout(entry)
    if kind == "errors":
        return _error_overlay(entry)
    return Image.fromarray(entry[kind] * 255).convert("RGB")


def _selection(entries):
    f1 = lambda entry: entry["metrics"]["final"]["f1"]
    delta = lambda entry: f1(entry) - entry["metrics"]["baseline"]["f1"]
    median = float(np.median([f1(entry) for entry in entries]))
    choices = [("F1 最高", max(entries, key=f1)), ("最接近中位数", min(entries, key=lambda entry: abs(f1(entry) - median))), ("F1 提升最多", max(entries, key=delta)), ("F1 下降最多", min(entries, key=delta)), ("Precision 最低", min(entries, key=lambda entry: entry["metrics"]["final"]["precision"] if entry["metrics"]["final"]["precision"] is not None else -1))]
    selected, positions = [], {}
    for role, entry in choices:
        if entry["sample_id"] not in positions:
            positions[entry["sample_id"]] = len(selected)
            selected.append((role, entry))
        else:
            index = positions[entry["sample_id"]]
            selected[index] = (selected[index][0] + " / " + role, entry)
    preview_roles = {"F1 最高", "F1 下降最多", "Precision 最低"}
    return selected, [(role, entry) for role, entry in selected if any(label in role for label in preview_roles)]


def _board(rows, columns, title, path, font_path):
    tile, gap, top, heading, caption, bottom = 360, 18, 76, 44, 54, 72
    width = len(columns) * tile + (len(columns) + 1) * gap
    height = top + heading + len(rows) * (tile + caption + gap) + bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    large, normal, small = (ImageFont.truetype(font_path, size) for size in (36, 28, 23))
    draw.text((gap, 18), title, font=large, fill="#18212D")
    for index, (_, label) in enumerate(columns):
        draw.text((gap + index * (tile + gap), top), label, font=normal, fill="#18212D")
    for row_index, (role, entry) in enumerate(rows):
        y = top + heading + row_index * (tile + caption + gap)
        base, final = entry["metrics"]["baseline"], entry["metrics"]["final"]
        label = f"{entry['sample_id'].rsplit('_', 1)[-1]}  |  {role}  |  F1 {base['f1']:.3f} → {final['f1']:.3f}  |  最终 P={final['precision']:.3f}，R={final['recall']:.3f}"
        draw.text((gap, y), label, font=small, fill="#25344A")
        for index, (kind, _) in enumerate(columns):
            picture = _panel(entry, kind)
            method = Image.Resampling.NEAREST if kind in {"gt", "baseline", "final"} else Image.Resampling.LANCZOS
            image.paste(picture.resize((tile, tile), method), (gap + index * (tile + gap), y + caption))
    draw.text((gap, height - 55), "绿色：正确检出 TP   红色：误检 FP   蓝色：漏检 FN   黄色：忽略区", font=small, fill="#25344A")
    image.save(path, dpi=(300, 300))


def _single(entry, path, font_path):
    tile, gap, caption, header = 320, 18, 44, 80
    columns = [("original", "原图"), ("gt", "人工真值"), ("baseline", "基线预测"), ("final", "最终预测"), ("cutout", "最终预测抠图"), ("errors", "最终误差叠加")]
    width, height = 3 * tile + 4 * gap, 2 * (tile + caption + gap) + header + 80
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title, label = ImageFont.truetype(font_path, 28), ImageFont.truetype(font_path, 25)
    base, final = entry["metrics"]["baseline"], entry["metrics"]["final"]
    draw.text((gap, 10), entry["sample_id"], font=title, fill="#18212D")
    draw.text((gap, 45), f"F1：{base['f1']:.3f} → {final['f1']:.3f}    IoU：{base['iou']:.3f} → {final['iou']:.3f}", font=label, fill="#25344A")
    for index, (kind, text) in enumerate(columns):
        row, col = divmod(index, 3)
        x, y = gap + col * (tile + gap), header + row * (tile + caption + gap)
        draw.text((x, y), text, font=label, fill="#25344A")
        picture = _panel(entry, kind)
        resample = Image.Resampling.NEAREST if kind in {"gt", "baseline", "final"} else Image.Resampling.LANCZOS
        image.paste(picture.resize((tile, tile), resample), (x, y + caption))
    draw.text((gap, height - 65), "绿色 TP 正确检出   红色 FP 误检   蓝色 FN 漏检", font=label, fill="#25344A")
    draw.text((gap, height - 33), "棋盘格为抠图显示背景；指标始终使用原始 512×512 像素。", font=ImageFont.truetype(font_path, 21), fill="#4E5969")
    image.save(path, dpi=(300, 300))


def _plots(entries, summary, output, font_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import PercentFormatter

    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path).get_name(), "font.size": 11, "axes.titlesize": 12, "axes.unicode_minus": False, "svg.fonttype": "path", "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=240, bbox_inches="tight", facecolor="white")
        fig.savefig(output / f"{name}.svg", bbox_inches="tight", facecolor="white")
        plt.close(fig)

    b = np.array([entry["metrics"]["baseline"]["f1"] for entry in entries])
    f = np.array([entry["metrics"]["final"]["f1"] for entry in entries])
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), layout="constrained")
    for improved, marker, color, label in ((True, "o", FINAL_COLOR, "最终改善"), (False, "x", BASE_COLOR, "最终下降")):
        selected = (f >= b) if improved else (f < b)
        axes[0].scatter(b[selected], f[selected], marker=marker, color=color, s=42, label=f"{label}（{selected.sum()}）")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#8B929C", linewidth=1)
    axes[0].set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="基线逐图 F1", ylabel="最终逐图 F1", title="每张图是否改善？")
    axes[0].set_aspect("equal")
    axes[0].legend(loc="upper left", fontsize=9)
    for values, color, label in ((b, BASE_COLOR, "基线"), (f, FINAL_COLOR, "最终")):
        axes[1].step(np.r_[0, np.sort(values), 1], np.r_[0, np.arange(1, len(values) + 1) / len(values), 1], where="post", color=color, label=f"{label}；中位数 {np.median(values):.3f}")
    axes[1].set(xlim=(0, 1), ylim=(0, 1.02), xlabel="逐图 F1", ylabel="不超过该 F1 的图像比例", title="低分图与整体稳定性")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(loc="upper left", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.18)
    save(fig, "sample_stability")

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7), layout="constrained")
    for index, (name, color, label) in enumerate((("baseline", BASE_COLOR, "基线"), ("final", FINAL_COLOR, "最终"))):
        values = [entry["boundaries"][name]["boundary_iou"] for entry in entries]
        values = np.array([value for value in values if value is not None])
        jitter = np.linspace(-0.1, 0.1, len(values))
        axes[0].scatter(index + jitter, values, color=color, s=22, alpha=0.65)
        axes[0].plot([index - 0.22, index + 0.22], [values.mean()] * 2, color=color, linewidth=3)
        axes[0].text(index + 0.25, values.mean(), f"均值 {values.mean():.3f}", va="center", fontsize=9)
    axes[0].set(xticks=[0, 1], xticklabels=["基线", "最终"], xlim=(-0.5, 1.8), ylim=(0, 1), ylabel="逐图 Boundary IoU", title="边界是否贴合？（带宽为对角线的 2%）")
    errors = {name: summary["pixel_metrics"][name] for name in ("baseline", "final")}
    fp = [errors[name]["fp"] / errors[name]["n_valid"] for name in errors]
    fn = [errors[name]["fn"] / errors[name]["n_valid"] for name in errors]
    axes[1].bar([0, 1], fp, color="#D95B59", label="FP 误检")
    axes[1].bar([0, 1], fn, bottom=fp, color="#4D79B7", label="FN 漏检")
    for index in range(2):
        axes[1].text(index, fp[index] / 2, f"{fp[index]:.2%}", ha="center", va="center", color="white")
        axes[1].text(index, fp[index] + fn[index] / 2, f"{fn[index]:.2%}", ha="center", va="center", color="white")
    axes[1].set(xticks=[0, 1], xticklabels=["基线", "最终"], ylabel="占全部有效像素的比例", title="错误来自误检还是漏检？", ylim=(0, max(np.array(fp) + fn) * 1.25))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(loc="upper right", fontsize=9)
    save(fig, "boundary_and_errors")

    fig, ax = plt.subplots(figsize=(9.5, 4.7), layout="constrained")
    intervals = [("总体 F1 差值", summary["uncertainty"]["global_f1"]["final_minus_baseline_global_f1"]), ("逐图平均 F1 差值", summary["uncertainty"]["mean_image_f1_delta"])]
    for index, (label, interval) in enumerate(intervals):
        ax.plot([interval["low"] * 100, interval["high"] * 100], [index, index], color=FINAL_COLOR, linewidth=2)
        ax.scatter([interval["estimate"] * 100], [index], color=FINAL_COLOR, s=50)
        ax.text(interval["high"] * 100 + 0.5, index, f"{interval['estimate'] * 100:+.2f} [{interval['low'] * 100:+.2f}, {interval['high'] * 100:+.2f}]", va="center", fontsize=10)
    ax.axvline(0, color="#8B929C", linestyle="--", linewidth=1)
    lo = min(item[1]["low"] * 100 for item in intervals)
    hi = max(item[1]["high"] * 100 for item in intervals)
    ax.set(yticks=[0, 1], yticklabels=[item[0] for item in intervals], xlabel="最终减去基线（百分点）", xlim=(min(0, lo) - 2, hi + max(8, (hi - lo) * 0.8)), ylim=(-0.6, 1.6), title="改善幅度有多稳定？配对图像 Bootstrap 95% 区间")
    ax.grid(axis="x", alpha=0.18)
    save(fig, "uncertainty")

    groups = summary["brightness_groups"]
    fig, ax = plt.subplots(figsize=(10.4, 4.8), layout="constrained")
    x = np.arange(len(groups))
    for shift, name, color, label in ((-0.18, "baseline", BASE_COLOR, "基线"), (0.18, "final", FINAL_COLOR, "最终")):
        values = [group[name]["f1"] for group in groups]
        ax.bar(x + shift, values, width=0.34, color=color, label=label)
        for index, value in enumerate(values):
            ax.text(index + shift, value + 0.015, f"{value:.3f}", ha="center", fontsize=10)
    labels = [f"{group['name']}（n={group['n_images']}）\nY {group['min_face_mean_y']:.1f}–{group['max_face_mean_y']:.1f}" for group in groups]
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 1), xlabel="按真实脸部区域的平均 Y 排序等分；Y 编码 0–255", ylabel="组内全局 F1", title="探索性亮度分组：同一模型在不同样本上的表现")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.18)
    save(fig, "brightness_groups")


def create_visual_assessment(manifest_path, run_dir, output_dir, *, font=None, n_resamples=10000, seed=42) -> dict:
    manifest, run, output = (Path(value).resolve() for value in (manifest_path, run_dir, output_dir))
    if output.exists():
        raise ValueError("output directory exists; choose a new visualization directory")
    font_path = _font_path(font)
    entries, frozen = _collect(manifest, run)
    records = []
    for entry in entries:
        row = {"sample_id": entry["sample_id"], "face_mean_y": entry["face_mean_y"], "face_fraction": entry["face_fraction"]}
        for name in ("baseline", "final"):
            row.update({f"{name}_{key}": value for key, value in entry["metrics"][name].items()})
            row[f"{name}_boundary_iou"] = entry["boundaries"][name]["boundary_iou"]
        row["delta_f1"] = row["final_f1"] - row["baseline_f1"]
        records.append(row)
    distributions = {}
    for name in ("baseline", "final"):
        values = np.array([entry["metrics"][name]["f1"] for entry in entries])
        boundaries = [entry["boundaries"][name]["boundary_iou"] for entry in entries]
        defined = [value for value in boundaries if value is not None]
        distributions[name] = {"n_images": len(values), "mean_f1": float(values.mean()), "median_f1": float(np.median(values)), "q25_f1": float(np.quantile(values, 0.25)), "q75_f1": float(np.quantile(values, 0.75)), "minimum_f1": float(values.min()), "maximum_f1": float(values.max()), "mean_boundary_iou": float(np.mean(defined)) if defined else None, "n_defined_boundary_iou": len(defined)}
    count_arrays = [np.array([[entry["metrics"][name][key] for key in ("tp", "fp", "tn", "fn")] for entry in entries], dtype=np.int64) for name in ("baseline", "final")]
    uncertainties = {"global_f1": paired_bootstrap_f1(*count_arrays, n_resamples=n_resamples, seed=seed), "mean_image_f1_delta": paired_mean_interval([row["baseline_f1"] for row in records], [row["final_f1"] for row in records], n_resamples=n_resamples, seed=seed)}
    groups = []
    eligible = [index for index, entry in enumerate(entries) if entry["face_mean_y"] is not None]
    for name, indices in zip(("较低亮度", "中间亮度", "较高亮度"), np.array_split(sorted(eligible, key=lambda index: entries[index]["face_mean_y"]), 3)):
        if not len(indices):
            continue
        group = {"name": name, "n_images": len(indices), "sample_ids": [entries[index]["sample_id"] for index in indices], "min_face_mean_y": min(entries[index]["face_mean_y"] for index in indices), "max_face_mean_y": max(entries[index]["face_mean_y"] for index in indices)}
        for scheme in ("baseline", "final"):
            group[scheme] = summarize(entries[index]["metrics"][scheme] for index in indices)["global"]
        groups.append(group)
    selected, preview = _selection(entries)
    summary = {"scope": "Post-hoc visualization and supplementary evaluation of fixed saved test predictions", "n_images": len(entries), "source_b_revision": frozen["b_revision"], "independent_test_sha256": _digest(run / "independent_test.json"), "manifest_sha256": _digest(manifest), "pixel_metrics": {name: frozen["results"][f"test-{name}"]["global"] for name in ("baseline", "final")}, "distributions": distributions, "image_comparison": {"improved_f1": sum(row["delta_f1"] > 0 for row in records), "declined_f1": sum(row["delta_f1"] < 0 for row in records), "same_f1": sum(row["delta_f1"] == 0 for row in records)}, "boundary_policy": {"dilation_ratio": 0.02, "band_radius_pixels": sorted({entry["boundaries"]["final"]["band_radius_pixels"] for entry in entries}), "aggregation": "unweighted image mean; undefined values excluded", "ignored_region_extension": "exclude ignored pixels and their band-radius neighborhood"}, "uncertainty": uncertainties, "brightness_groups": groups, "selected_examples": [{"role": role, "sample_id": entry["sample_id"]} for role, entry in selected], "limitations": ["Supplementary metrics and subgroups were chosen after seeing the original test results; they are descriptive, not a new independent benchmark", "Image-bootstrap intervals describe resampling variability within these 32 selected images, not guaranteed population coverage", "Pixel brightness is not a verified illumination condition or demographic label; brightness groups do not establish causality", "No new inference, model selection or parameter changes were performed"], "references": {"boundary_iou_paper": "https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html", "boundary_iou_reference": "https://github.com/bowenc0221/boundary-iou-api/blob/master/boundary_iou/utils/boundary_utils.py", "paired_bootstrap": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html"}}
    import cv2
    import matplotlib
    import PIL
    summary["runtime_versions"] = {"numpy": np.__version__, "Pillow": PIL.__version__, "opencv": cv2.__version__, "matplotlib": matplotlib.__version__}
    summary["analysis_source_sha256"] = {name: _digest(Path(__file__).parent / name) for name in ("assessment.py", "visualization.py")}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".visual-assessment-", dir=output.parent) as temporary:
        stage = Path(temporary) / "figures"
        (stage / "per_image").mkdir(parents=True)
        (stage / "transparent_cutouts").mkdir()
        _write_json(stage / "analysis_summary.json", summary)
        with (stage / "per_image_assessment.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        _board(preview, [("original", "原图"), ("gt", "人工真值"), ("cutout", "B 最终预测抠图"), ("errors", "误差叠加")], "B 预测效果：成功、漏检与误检", stage / "prediction_examples.png", font_path)
        _board(selected, [("original", "原图"), ("gt", "人工真值"), ("baseline", "基线预测"), ("final", "最终预测"), ("errors", "最终误差")], "基线与最终方案：代表样例对照", stage / "baseline_final_comparison.png", font_path)
        for index, entry in enumerate(entries):
            _single(entry, stage / "per_image" / f"{entry['sample_id']}.png", font_path)
            _cutout(entry, transparent=True).save(stage / "transparent_cutouts" / f"{entry['sample_id']}.png")
            if (index + 1) % 8 == 0:
                print(f"Rendered {index + 1}/{len(entries)} image comparisons", flush=True)
        _plots(entries, summary, stage, font_path)
        (stage / "README.md").write_text("# B 预测可视化与补充评价\n\nprediction_examples.png：成功、严重漏检与误检样例。baseline_final_comparison.png：按明确规则选取的代表样例。per_image/：全部测试图的六宫格；transparent_cutouts/：直接使用 B 最终预测作为 alpha 的透明 PNG。棋盘格仅为抠图显示背景。\n\n误差图：绿色 TP，红色 FP，蓝色 FN，黄色忽略区。图片仅为显示缩放，全部指标在原始像素尺寸计算；预测/真值/配置原文件均未改变。\n\n其余 PNG/SVG 为逐图稳定性、Boundary IoU 与误检漏检、配对 Bootstrap 区间、探索性亮度分组。analysis_summary.json 和 per_image_assessment.csv 记录定义及结果。\n\n边界带宽为图像对角线的 2%（512×512 对应 14 像素）。Bootstrap 按图像配对重采样 10000 次、种子 42，区间为 percentile 95%。补充评估为事后描述，不能当成新的独立测试；亮度分组也不能证明光照因果。\n", encoding="utf-8")
        stage.rename(output)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Visualize fixed B predictions and assess stability, boundaries and uncertainty")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run", required=True, help="Completed independent test directory")
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--font", help="Chinese-capable font file")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        result = create_visual_assessment(args.manifest, args.run, args.output, font=args.font, n_resamples=args.bootstrap_resamples, seed=args.seed)
    except (ValueError, OSError, KeyError, ImportError) as error:
        parser.exit(2, f"Visualization failed: {error}\n")
    print(f"Visualized {result['n_images']} fixed test predictions: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
