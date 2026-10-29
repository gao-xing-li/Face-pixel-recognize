# 公共接口约定 v0.1

本约定对应首次公共骨架与 C 评价模块。数据集、正式标签和 A/B 算法尚待接入；下述检测接口是 B 的接入建议，评价接口已经实现。内部实现可以调整，跨模块语义变化需同步本文件。

## 1. 架构与责任

同一 Python 工程内通过函数和文件交接。A 准备统一数据及训练统计，B 完成检测、验证选参及代码整合，C 核查测试数据、评价固定方案并整合报告。

```text
A 数据及唯一划分 ──训练集──> A 统计参数
       │                        │
       ├──验证集───────────────> B 检测、验证选参
       │                        │
       └──测试集──> C 独立测试 <──B 固定代码、配置及预测结果

B 验证与 C 测试共用 evaluate / summarize，使用相同指标口径。
```

当前已实现 `src/face_pixel/evaluation.py`、`data_io.py`、`batch.py`、`cli.py` 和模拟样例 `demo.py`。A 后续可新增 `modeling.py`，B 后续可新增 `detection.py`、`validation.py` 和总入口；不预建会被误认为已完成的算法占位实现。

## 2. 数据与标签

- A 确定并记录正类是人体皮肤还是脸部皮肤，原始语义类别如何映射为正类。当前没有正式标签定义。
- 内部掩膜为 `(H, W)`、整数或布尔 0/1，1=约定正类，0=其他；评价函数拒绝浮点概率、255、负数及未知标签。
- 原图、真值、预测、有效区域必须严格对齐。评价入口不自动缩放、阈值化、翻转或修正方向。
- 有效区域 `valid_mask` 为同尺寸 0/1，1=计入评价，0=忽略。忽略位置的真值仍应填合法的 0/1 值。整张图全部忽略会报错，由 A/C 核查后记录排除，不静默丢弃。
- PNG 文件可使用 `binary01`（0/1）或 `binary255`（0/255），通过命令参数显式指定。原始多类标签、彩色标注、概率图由 A/B 先转换，不能用简单“大于零”代替正式标签映射。
- 默认评价尺寸为原图尺寸。B 若在内部缩放，负责把预测恢复到该尺寸并记录方法；离散标签变换使用保持类别值的方式。

建模与检测建议统一 BGR uint8、0～255，OpenCV `COLOR_BGR2YCrCb` 的通道为 **Y、Cr、Cb**。阈值使用标准差，不能直接用方差替代。C 的 Pillow 读取仅用于掩膜及案例展示，不参与 B 的颜色模型。

## 3. A 的唯一数据划分清单

路径建议：`data/split_manifest.csv`；示例文件为 `data/split_manifest.example.csv`，其中只有虚构路径。

```csv
sample_id,image_path,mask_path,split,valid_mask_path
sample_001,raw/images/sample_001.jpg,processed/masks/sample_001.png,train,
sample_002,raw/images/sample_002.jpg,processed/masks/sample_002.png,validation,
sample_003,raw/images/sample_003.jpg,processed/masks/sample_003.png,test,processed/valid/sample_003.png
```

`sample_id` 全局唯一；`split` 只接受 `train`、`validation`、`test`。所有相对路径均相对于**该 CSV 所在目录**，可以使用 `/` 分隔路径。`valid_mask_path` 可省略整列或留空，表示所有像素有效。可以增加来源、场景等元数据列。

加载器拒绝重复 ID、复用原图路径、未知集合名和缺少必填字段。批量评价还读取全局清单中所有原图并检查 SHA-256，拒绝同一文件内容以不同名称重复出现；因此评价机器需要有完整清单中的原图。只有选定集合的真值和预测需要参与指标计算。

文件哈希不能发现不同编码、裁剪或近重复图片，也不能替代身份去重；A 记录实际去重方法及身份信息限制，C 复核。A 统一修改清单，B/C 不另建口径不同的正式划分。

## 4. A → B：模型参数；B → C：固定配置

A 在 `configs/skin_params.json` 交付正式统计，至少说明颜色转换、实际通道及单位、均值、标准差、训练划分版本、正类映射和预处理。没有真实统计前不填写模拟数值到正式参数文件。

B 的建议接口：

```python
detect(image_bgr, params, config) -> pred_mask
```

输入为 BGR uint8 图像和实际参数/配置，输出为原图尺寸 uint8 0/1 掩膜。预处理、阈值及辅助处理均包含在 B 的检测流程内，避免 C 重复应用。

正式测试前，B 交接 `configs/final_config.json`、实际用到的统计文件及可运行代码，配置应记录：

| 字段 | 内容 |
| --- | --- |
| `config_id` | 本次方案的稳定标识；批量评价会核对它与命令参数一致 |
| `code_revision` | 能获取的实际代码提交版本 |
| `data_manifest_sha256` | 所用划分清单的哈希或等价可追溯版本 |
| `model_params` | 正式训练统计文件的位置、版本或哈希 |
| `threshold` | 最终 k、使用的颜色分量及判别规则 |
| `preprocess`、`postprocess` | 实际步骤、顺序、参数；没有则显式说明 |
| `evaluation_size` | 原图尺寸，以及内部缩放和恢复方式 |
| `selection` | 验证选参指标、选定理由、是否已固定 |

这些字段的内部结构由 B 决定，C 按交接清单核查。目前程序仅强制校验 `config_id` 并保存配置原文件和哈希，不宣称自动认证方案已固定或不存在数据泄漏。正式测试还需 C 人工核对训练统计、代码和划分对应关系。

## 5. 已实现的共用评价接口

```python
from face_pixel import evaluate, summarize

metrics = evaluate(pred_mask, gt_mask, valid_mask=None)
summary = summarize([metrics_1, metrics_2])
```

`evaluate` 的顺序是**预测在前、真值在后**。返回普通字典：`tp, fp, tn, fn, n_valid, accuracy, precision, recall, f1, iou`。同名指标统一针对正类 1。

| 指标 | 定义 |
| --- | --- |
| Accuracy | `(TP + TN) / n_valid` |
| Precision | `TP / (TP + FP)` |
| Recall | `TP / (TP + FN)` |
| F1 | `2TP / (2TP + FP + FN)` |
| IoU | `TP / (TP + FP + FN)` |

分母为零时返回 Python `None`，JSON 保存为 `null`，CSV 留空。全背景且预测全背景时 Accuracy=1，其余四项未定义。存在漏检且没有正类预测时，Precision 未定义，Recall/F1/IoU=0。

`summarize` 返回：

- `global`：先累加有效像素的 TP/FP/TN/FN 再计算，是正式报告建议的总体口径。
- `image_mean`：每张图等权平均，各指标排除未定义值，作为补充结果。
- `image_mean_n_defined`：每项逐图平均实际纳入的图像数。
- `n_images`：本次纳入的图像总数。

这里的逐图平均不等于某些库中对多个类别进行的 macro average。汇总函数从计数重新计算指标，不信任外部传入的已算分数。

手算混合样例：

```python
gt   = [[1, 1, 0], [0, 0, 0]]
pred = [[1, 0, 1], [0, 0, 0]]
# TP=1, FP=1, TN=3, FN=1, n_valid=6
# Accuracy=4/6, Precision=Recall=F1=0.5, IoU=1/3
```

## 6. B → C：预测结果清单

B 为每个方案、每个集合单独输出 CSV：

```csv
sample_id,pred_path
sample_003,predictions/sample_003.png
```

路径相对于预测清单目录。它必须与本次选定集合的 ID 完全一致，不能缺少样本或混入其他集合。文件使用无损二值 PNG，B 说明保存编码。每次运行给定 `config_id`，基础版和最终版分别输出、分别评价。

```text
face-pixel-evaluate --manifest data/split_manifest.csv --predictions results/final-test/predictions.csv --split test --config-id final-v1 --run-config configs/final_config.json --gt-encoding binary255 --pred-encoding binary255 --output results/evaluation-final-test
```

示例命令中的编码须按实际数据修改。默认编码为 `binary01`。有有效区域文件时再按实际情况指定 `--valid-encoding`。B 在验证集调用同一入口，将 `--split` 改为 `validation` 并使用对应预测清单。

输出为新目录；已存在的结果不会覆盖。缺文件、标签错误、配对或尺寸不符时整体失败，不发布部分统计：

```text
evaluation-final-test/
├── summary.json       # 总体计数、指标、逐图平均、口径、输入文件哈希
├── per_image.csv      # 每图指标及对应案例路径
├── run_config.json    # 提供 --run-config 时原样复制
├── cases/             # 原图—真值—预测—错误图
└── README.md          # 结果解释及错误图图例
```

错误图绿色=TP，红色=FP，蓝色=FN，黑色=TN，黄色=忽略；展示缩放不改变指标。大型预测及案例默认不入库，正式可分享图表由 B/C 按最终提交要求选择保存。

## 7. 实验边界

验证集用于选参，测试集用于固定方案后的评价。C 可复用评价程序，仍独立核查数据、配置及结果。如果根据测试结果改方案，相关图像已参与开发，需要重新安排独立测试。当前 demo 与手算样例只检验程序正确性，不能写入正式性能结论。

参考：[OpenCV 颜色转换](https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html)、[F1 定义与未定义情况](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)。
