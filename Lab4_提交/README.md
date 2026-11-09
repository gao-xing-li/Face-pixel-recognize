# 第四次 Project 人脸分割形态学改进

实验报告为 [报告.pdf](报告.pdf)。提交包由 `feat/lab4` 的 `ce9a2bd1d05ae4f30258c0872245e0a0ae3f9cfe` 导出。提交版运行入口支持读取重算的肤色参数，并可从普通 ZIP 解压目录运行。

包内包含 162 张原图及其多类标签。数据固定划分为训练 98 张、验证 32 张、测试 32 张，路径与划分见 [数据清单](data/split_manifest.csv)，来源和标签定义见 [数据元信息](data/part_a/metadata.json)。

## 解压后复跑

在本 README 所在目录使用 Python 3.10 或更新版本。图册生成使用 Windows 的微软雅黑字体或 Linux 的 Noto Sans CJK 字体。

```text
python -m pip install -e .
face-pixel-fit --params results/reproduce/skin_params.json --output results/reproduce/skin_statistics
face-pixel-prepare --output data/processed/face
face-pixel --params results/reproduce/skin_params.json --output results/reproduce/lab3
python experiments/lab4/dual_ellipse_morph/build_gallery.py --params results/reproduce/skin_params.json --output results/reproduce/lab4
```

前两步重新统计训练集肤色并生成评价所需的二值真值。Lab3 在验证集与测试集比较纯颜色阈值、最大连通块填孔及形态学补全。Lab4 使用同一份重算的肤色参数，从训练集估计眼位与椭圆几何关系，在 32 张验证图上完成双椭圆约束修整。再次运行时选择新的派生数据与输出目录，并用 `face-pixel --manifest` 指向新生成的清单。

可另外生成 Lab3 测试集的逐图对照。

```text
face-pixel-visualize --manifest data/processed/face/manifest.csv --run results/reproduce/lab3 --before original --after final --output results/reproduce/lab3-visualization
```

## 查看结果

- Lab3 指标位于 `results/reproduce/lab3/experiment.json`。初版最大连通块填孔在验证集和测试集的全局 F1 分别为 72.70% 和 70.69%。
- Lab4 新生成的图册位于 `results/reproduce/lab4/index.html`，整体指标和逐图对照保存在同目录的 `summary.json` 和 `per_image.csv`。
- 已附带 [Lab4 最终验证图册](results/lab4_dual_ellipse_morph_v2_20261108/index.html) 和 [指标归档](results/lab4_dual_ellipse_morph_v2_20261108/summary.json)。在同一批验证图上，Lab3 初版到 Lab4 的全局 F1 为 72.70% → 82.89%，IoU 为 57.11% → 70.79%。这些数字对应验证集开发结果。

## 重生成 Lab4 中间图册

在提交包根目录运行下列命令。各命令可以独立执行，使用随包的原图、标签和固定肤色参数。完成后打开对应输出目录的 `index.html`。再次运行时更换 `--output` 为未使用的目录。

| 输出目录末级名称 | 生成的图册 |
| --- | --- |
| `lab3_comparison` | 162 张原图的 Lab3 初版与 41×41 补全对照，复查归档的验证与测试掩膜 |
| `connected_components` | 162 张原始阈值、最大连通块及填孔的逐阶段对照 |
| `ellipse_initial` | 162 张原始肤色区域的单块及近邻组合椭圆 |
| `estimate_refine` | 32 张验证图从候选椭圆到一轮区域修整的过程图 |
| `ellipse_top3` | 32 张验证图在原始、5×5 闭运算、5×5 开运算后的前三块椭圆对照，分别位于 `raw/`、`close/`、`open/` |
| `eye_no_prior` | 无位置先验的眼睛候选与配对 |
| `eye_midpoint_prior` | 加入双眼中点弱先验后的配对对照 |
| `eye_two_point_prior` | 左右眼分别约束搜索的三轮配对对照 |
| `eye_ellipse` | 检测双眼生成单椭圆，与固定平均眼位生成的椭圆对照 |
| `dual_ellipse_initial` | 初版双椭圆分区形态学，含内外椭圆及处理后掩膜 |

```powershell
python tools/build_review_gallery.py --output results/reproduce/history/lab3_comparison
python experiments/lab4/connected_components/build_gallery.py --output results/reproduce/history/connected_components
python experiments/lab4/ellipse_initial/build_gallery.py --output results/reproduce/history/ellipse_initial
python experiments/lab4/estimate_refine/run.py --output results/reproduce/history/estimate_refine
python experiments/lab4/ellipse_check/build_gallery.py --preprocess all --output results/reproduce/history/ellipse_top3
python experiments/lab4/eye_no_prior/build_gallery.py --output results/reproduce/history/eye_no_prior
python experiments/lab4/eye_midpoint_prior/build_gallery.py --output results/reproduce/history/eye_midpoint_prior
python experiments/lab4/eye_check/build_gallery.py --output results/reproduce/history/eye_two_point_prior
python experiments/lab4/eye_ellipse/build_gallery.py --output results/reproduce/history/eye_ellipse
python experiments/lab4/dual_ellipse_initial/build_gallery.py --output results/reproduce/history/dual_ellipse_initial
```

最终加入低支持重建的双椭圆图册用“解压后复跑”中的最后一条命令生成。历史图册按需生成，其中 `estimate_refine` 会输出多候选过程图，完整运行约占 470 MB 磁盘空间。
