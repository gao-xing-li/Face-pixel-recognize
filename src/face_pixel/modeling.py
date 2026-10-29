"""Fit Cr/Cb skin statistics using all label-1/2 pixels from the approved train split."""

import csv
import json
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from PIL import Image

from .part_a_common import PROJECT, DATA, RAW, MANIFEST, content_fingerprint, digest


OUTPUT = PROJECT / "results" / "skin_statistics"
PARAMS = PROJECT / "configs" / "skin_params.json"
SKIN_IDS = [1, 2]


def image_histograms(bgr, labels):
    """Return exact integer histograms; joint histogram axes are [Cr, Cb]."""
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected uint8 BGR image")
    if labels.dtype != np.uint8 or labels.ndim != 2 or bgr.shape[:2] != labels.shape:
        raise ValueError("Expected matching uint8 label image")
    if np.any(labels > 18):
        raise ValueError("Unknown label ID")
    mask = (labels == 1) | (labels == 2)
    if not mask.any():
        raise ValueError("Training image contains no skin/nose pixels")
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1][mask].astype(np.int64)
    cb = ycrcb[:, :, 2][mask].astype(np.int64)
    return {
        "cr": np.bincount(cr, minlength=256).astype(np.int64),
        "cb": np.bincount(cb, minlength=256).astype(np.int64),
        "joint_cr_cb": np.bincount(cr * 256 + cb, minlength=65536).reshape(256, 256).astype(np.int64),
        "skin_pixels": int(np.count_nonzero(labels == 1)),
        "nose_pixels": int(np.count_nonzero(labels == 2)),
    }


def histogram_statistics(histogram):
    counts = np.asarray(histogram, dtype=np.int64)
    if counts.shape != (256,) or np.any(counts < 0) or counts.sum() <= 0:
        raise ValueError("Expected nonempty 256-bin histogram")
    n = int(counts.sum())
    values = np.arange(256, dtype=np.float64)
    mean = float(np.dot(values, counts) / n)
    variance = float(np.dot((values - mean) ** 2, counts) / n)
    return {"mean": mean, "variance": variance, "std": float(np.sqrt(variance))}


def training_rows(manifest, metadata):
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if Counter(r["split"] for r in rows) != {"train": 98, "val": 32, "test": 32}:
        raise ValueError("Expected the approved 98/32/32 split")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate sample ID")
    split = metadata["split"]
    if split["human_split_review"] != "passed" or split["manifest_sha256"] != digest(manifest):
        raise ValueError("The current manifest has not passed split review")
    if metadata["skin_ids"] != SKIN_IDS:
        raise ValueError("Unexpected skin modeling label definition")
    return [r for r in rows if r["split"] == "train"]


def read_training_pair(row):
    paths = []
    for field, sha_field in (("image_path", "image_sha256"), ("label_path", "label_sha256")):
        path = (PROJECT / row[field]).resolve()
        path.relative_to(RAW.resolve())
        if digest(path) != row[sha_field]:
            raise ValueError(f"Hash mismatch: {path}")
        paths.append(path)
    image_path, label_path = paths
    # Read metadata without converting either source; imdecode handles Unicode paths.
    with Image.open(image_path) as source, Image.open(label_path) as mask:
        if source.mode != "RGB" or mask.mode != "L" or source.size != mask.size:
            raise ValueError(f'Unexpected source mode/size: {row["sample_id"]}')
        if source.size != (int(row["width"]), int(row["height"])):
            raise ValueError("Dimensions do not match manifest")
    bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    labels = cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if bgr is None or labels is None:
        raise ValueError(f'Cannot decode: {row["sample_id"]}')
    return bgr, labels


def save_figures(cr_hist, cb_hist, joint, cr_stats, cb_stats):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    values = np.arange(256)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    for ax, name, hist, stats, color in zip(axes, ("Cr", "Cb"), (cr_hist, cb_hist),
                                          (cr_stats, cb_stats), ("#c84a55", "#3478b8")):
        ax.bar(values, hist / hist.sum(), width=1, color=color, alpha=0.8)
        ax.axvline(stats["mean"], color="#222222", label="Mean")
        ax.axvline(stats["mean"] - stats["std"], color="#555555", linestyle="--", label="Mean +/- 1 std (reference)")
        ax.axvline(stats["mean"] + stats["std"], color="#555555", linestyle="--")
        ax.set(xlim=(0, 255), xlabel=f"{name} (uint8, 0-255)", ylabel="Fraction of selected skin pixels",
               title=f'{name}: mean={stats["mean"]:.3f}, std={stats["std"]:.3f}')
        ax.legend(fontsize=8)
    fig.suptitle("Training skin-color distributions | labels 1+2 | equal pixel weights")
    fig.savefig(OUTPUT / "channel_distributions.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.6, 6.2), layout="constrained")
    plotted = np.ma.masked_equal(joint, 0)
    heatmap = ax.imshow(plotted, origin="lower", extent=(-0.5, 255.5, -0.5, 255.5),
                        interpolation="nearest", cmap="viridis", norm=LogNorm(vmin=1, vmax=int(joint.max())))
    ax.set(xlim=(0, 255), ylim=(0, 255), xlabel="Cb (uint8, 0-255)", ylabel="Cr (uint8, 0-255)",
           title="Training skin pixels: joint Cr/Cb distribution")
    fig.colorbar(heatmap, ax=ax, label="Pixel count per bin (log color scale)")
    fig.savefig(OUTPUT / "joint_distribution.png", dpi=170)
    plt.close(fig)


def main():
    manifest = MANIFEST
    metadata = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))
    rows = training_rows(manifest, metadata)
    cr_hist = np.zeros(256, dtype=np.int64)
    cb_hist = np.zeros(256, dtype=np.int64)
    joint = np.zeros((256, 256), dtype=np.int64)
    image_counts = []
    for row in rows:
        bgr, labels = read_training_pair(row)
        hist = image_histograms(bgr, labels)
        cr_hist += hist["cr"]
        cb_hist += hist["cb"]
        joint += hist["joint_cr_cb"]
        count = hist["skin_pixels"] + hist["nose_pixels"]
        image_counts.append({"sample_id": row["sample_id"], "source_filename": row["source_filename"],
                             "split": "train", "skin_pixels": hist["skin_pixels"],
                             "nose_pixels": hist["nose_pixels"], "selected_pixels": count,
                             "image_pixels": int(labels.size), "selected_fraction": count / labels.size})
    n = int(cr_hist.sum())
    if n != int(cb_hist.sum()) or n != int(joint.sum()) or n != sum(r["selected_pixels"] for r in image_counts):
        raise RuntimeError("Inconsistent histogram/count totals")
    if not np.array_equal(joint.sum(axis=1), cr_hist) or not np.array_equal(joint.sum(axis=0), cb_hist):
        raise RuntimeError("Joint histogram axes disagree with marginal histograms")
    for row in image_counts:
        row["overall_pixel_weight"] = row["selected_pixels"] / n
    cr_stats, cb_stats = histogram_statistics(cr_hist), histogram_statistics(cb_hist)
    params = {
        "schema_version": 1, "fit_split": "train", "training_image_count": len(rows),
        "skin_pixel_count": n, "skin_label_ids": SKIN_IDS,
        "color_space": "OpenCV YCrCb", "input_color_order": "BGR", "input_dtype": "uint8",
        "conversion": "cv2.COLOR_BGR2YCrCb", "conversion_channel_order": ["Y", "Cr", "Cb"],
        "model_channel_indices": {"Cr": 1, "Cb": 2}, "pixel_value_range": [0, 255],
        "preprocessing": "none; no resizing or color enhancement",
        "sampling": "all selected training pixels; no random sampling",
        "weighting": "equal weight per pixel; images contribute in proportion to selected pixel count",
        "variance_ddof": 0,
        "mean_cr": cr_stats["mean"], "variance_cr": cr_stats["variance"], "std_cr": cr_stats["std"],
        "mean_cb": cb_stats["mean"], "variance_cb": cb_stats["variance"], "std_cb": cb_stats["std"],
        "threshold_k": None, "threshold_status": "not selected; validation belongs to Part B",
        "training_sample_ids": [r["sample_id"] for r in rows],
        "manifest_path": "data/split_manifest.csv", "manifest_sha256": digest(manifest),
        "training_content_fingerprint": content_fingerprint(rows),
        "statistics_script_sha256": digest(Path(__file__)),
        "runtime_versions": {"python": platform.python_version(), "opencv": cv2.__version__,
                             **{package: version(package) for package in ("numpy", "Pillow", "matplotlib", "opencv-python-headless")}},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PARAMS.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT / "per_image_counts.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(image_counts[0]))
        writer.writeheader()
        writer.writerows(image_counts)
    np.savez_compressed(OUTPUT / "histograms.npz", cr=cr_hist, cb=cb_hist, joint_cr_cb=joint)
    save_figures(cr_hist, cb_hist, joint, cr_stats, cb_stats)
    summary = f"""# 训练集肤色统计结果

输入为已确认划分的 **{len(rows)} 张训练图像**，取标签 1（皮肤）与 2（鼻子），共 **{n:,} 个像素**。
全部像素等权；不采样、不缩放、不做颜色增强；验证与测试图像不参与统计。

| 通道 | 均值 | 总体方差（ddof=0） | 标准差 |
| --- | ---: | ---: | ---: |
| Cr | {cr_stats['mean']:.8f} | {cr_stats['variance']:.8f} | {cr_stats['std']:.8f} |
| Cb | {cb_stats['mean']:.8f} | {cb_stats['variance']:.8f} | {cb_stats['std']:.8f} |

[正式参数](../../configs/skin_params.json)保留完整精度、训练样本 ID、来源哈希和依赖版本。
[逐图计数](per_image_counts.csv)记录每张图的皮肤、鼻子像素数及对总体统计的权重。
[频数数据](histograms.npz)包含 `cr[256]`、`cb[256]` 和 `joint_cr_cb[256,256]`；联合频数轴顺序为 Cr、Cb。

![Cr/Cb 分布](channel_distributions.png)

虚线为均值 ± 1 个标准差，仅供观察分布，不是已经选定的分割阈值。

![联合分布](joint_distribution.png)

联合分布横轴为 Cb、纵轴为 Cr；对数颜色仅用于展示频数，未对统计像素重新加权。
这些结果描述训练数据，不是分割效果或泛化能力验证。阈值系数 k、填孔和最终评价尚未实现。

在项目根目录复现（Windows）：

```powershell
.\\.venv\\Scripts\\python.exe -X utf8 tools/compute_skin_statistics.py
```
"""
    (OUTPUT / "README.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"training_images": len(rows), "skin_pixels": n, "Cr": cr_stats, "Cb": cb_stats,
                      "parameters": str(PARAMS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
