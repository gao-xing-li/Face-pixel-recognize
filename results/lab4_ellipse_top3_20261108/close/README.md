# Lab4 前三连通块椭圆检查

当前运行范围为弱预操作、选取面积前三个连通块、分别拟合与可视化椭圆。

## 运行

在仓库根目录执行：

```powershell
python experiments/lab4/ellipse_check/build_gallery.py --output results/lab4_ellipse_top3_20261108
```

默认只运行 32 张验证图，并分别生成原始、5×5 闭运算和 5×5 开运算三个页面，入口为输出目录的 `index.html`。每种输入独立按面积取前三块，各自拟合一个椭圆，没有双块、三块组合。原始阈值保留供对照。

只查看原始输入可添加 `--preprocess raw`。还支持 `--preprocess close` 或 `--preprocess open`。扩展图像划分可添加 `--splits validation test train`。每次指定新输出目录。

图册可以离线打开，分享时复制整个输出目录。可选择图像及面积第 1、2、3 的候选，查看实际拟合点、原图上的椭圆和真值对照。

## 参数与文件

`config.json` 中 `max_components=3`、`combine_candidates=false` 控制本轮范围。`min_component_area=1`，按面积顺序取前三块，数量不足则按实际数量处理。少于五个有效轮廓点或几何退化时，记录拟合状态。

起步沿用肤色统计与 k=1.25。开闭运算核为 5×5 椭圆，外围补零后操作，再裁回原尺寸。单块轮廓最多采样 256 点。`inner_scale=0.6` 仅用于描述中心肤色支持。

每个页面保存 `candidates.json`、`per_candidate.csv`、`summary.json`、`settings.json` 及源程序快照。真实标签仅用于拟合后的粗定位测量。页面中的 F1、IoU 来自实心椭圆区域，并非椭圆修整后的人脸分割结果。

`ellipse.py` 复用共享候选与拟合逻辑。未提供 max_components 的旧配置仍保留原来的面积过滤行为，combine_candidates 默认保持旧的组合逻辑。

## 历史结果

- `results/lab4_ellipse_check_20261108/` 是此前全部 162 张图、多个单块及组合的探索图册，其实际参数保存于该目录。
- `results/lab4_estimate_refine_20261108/` 是此前验证集的一次完整修整实验，作为历史记录保留。
- 当前精简入口使用 `results/lab4_ellipse_top3_20261108/`。

详细讨论记录位于 `docs/lab4/椭圆检查设计.md`。
