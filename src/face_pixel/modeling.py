"""用训练图中皮肤和鼻子的全部像素计算 Cr/Cb 均值、总体方差和标准差。"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from .data_io import PROJECT, _read_csv, read_bgr, write_json, write_csv

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


def save_figures(cr_hist, cb_hist, joint, cr_stats, cb_stats, output):
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
    fig.savefig(output / "channel_distributions.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.6, 6.2), layout="constrained")
    plotted = np.ma.masked_equal(joint, 0)
    heatmap = ax.imshow(plotted, origin="lower", extent=(-0.5, 255.5, -0.5, 255.5),
                        interpolation="nearest", cmap="viridis", norm=LogNorm(vmin=1, vmax=int(joint.max())))
    ax.set(xlim=(0, 255), ylim=(0, 255), xlabel="Cb (uint8, 0-255)", ylabel="Cr (uint8, 0-255)",
           title="Training skin pixels: joint Cr/Cb distribution")
    fig.colorbar(heatmap, ax=ax, label="Pixel count per bin (log color scale)")
    fig.savefig(output / "joint_distribution.png", dpi=170)
    plt.close(fig)



def fit_statistics(manifest, params_path, output, source_root=None):
    manifest = Path(manifest).resolve()
    source_root = Path(source_root).resolve() if source_root else manifest.parent.parent
    rows = [r for r in _read_csv(manifest, {"sample_id", "split", "image_path", "label_path"}) if r["split"] == "train"]
    if not rows:
        raise ValueError("Empty training split")
    cr_hist = np.zeros(256, dtype=np.int64)
    cb_hist = np.zeros(256, dtype=np.int64)
    joint = np.zeros((256, 256), dtype=np.int64)
    counts = []
    for row in rows:
        image = read_bgr(source_root / row["image_path"])
        labels = np.array(Image.open(source_root / row["label_path"]))
        hist = image_histograms(image, labels)
        cr_hist += hist["cr"]
        cb_hist += hist["cb"]
        joint += hist["joint_cr_cb"]
        counts.append({"sample_id": row["sample_id"], "skin_pixels": hist["skin_pixels"], "nose_pixels": hist["nose_pixels"]})
    cr, cb = histogram_statistics(cr_hist), histogram_statistics(cb_hist)
    params = {"color_space": "OpenCV YCrCb", "input_color_order": "BGR", "fit_split": "train",
              "skin_label_ids": [1, 2], "training_image_count": len(rows), "skin_pixel_count": int(cr_hist.sum()),
              "conversion_channel_order": ["Y", "Cr", "Cb"], "variance_ddof": 0}
    for name, statistics in (("cr", cr), ("cb", cb)):
        for key, value in statistics.items():
            params[f"{key}_{name}"] = value
    write_json(params_path, params)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_image_counts.csv", counts)
    np.savez_compressed(output / "histograms.npz", cr=cr_hist, cb=cb_hist, joint_cr_cb=joint)
    save_figures(cr_hist, cb_hist, joint, cr, cb, output)
    return params


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(PROJECT / "data/split_manifest.csv"))
    parser.add_argument("--params", default=str(PROJECT / "configs/skin_params.json"))
    parser.add_argument("--output", default=str(PROJECT / "results/skin_statistics"))
    args = parser.parse_args(argv)
    print(fit_statistics(args.manifest, args.params, args.output))


if __name__ == "__main__":
    main()
