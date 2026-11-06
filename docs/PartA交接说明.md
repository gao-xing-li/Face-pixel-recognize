# Part A 交接说明

更新日期：2026-11-02。A 的数据与统计已接入当前整合项目，不再需要另取 A 副本或调用旧适配入口。本文说明当前可用成果及保持不变的 A 决策；独立 A 分支 `1389a77` 仍保留当时的完整交付。

## 1. 可用成果

| 内容 | 入口 |
| --- | --- |
| 162 对原图与原始标签 | `data/raw/face_segment/images/`、`labels/`，共 324 个已跟踪 PNG |
| 唯一源清单 | [split_manifest.csv](../data/split_manifest.csv) |
| 来源、检查与划分 | [metadata.json](../data/part_a/metadata.json)、[split_report.json](../data/part_a/split_report.json)、[数据说明](../data/README.md) |
| 当前训练参数 | [skin_params.json](../configs/skin_params.json) |
| 原统计图与频数 | [统计结果](../results/skin_statistics/README.md)、[histograms.npz](../results/skin_statistics/histograms.npz) |
| 统计实现及数值检查 | [modeling.py](../src/face_pixel/modeling.py)、[test_skin_statistics.py](../tests/test_skin_statistics.py) |
| 分布解释与设计历史 | [统计初步分析](PartA统计结果初步分析.md)、[设计与决策记录](PartA设计与决策记录.md) |
| Part A 报告正文与图表依据 | [background.tex](../report/人脸分割实验报告/sections/background.tex)、[来源记录](../report/人脸分割实验报告/sources/part_a/report_sources.json) |

源图和原始多类标签不变。原始清单字段为 `label_path`，路径相对项目根目录，集合名称保留 val；运行准备转换为 mask_path、validation 和派生清单相对路径。

## 2. 来源、规模和检查范围

样本由实验者从 Face_segment 原始 test 的目录 2 主观外观筛选，不是官方亚洲人群属性标签。最初 163 对，完全重复图像保留 00583、排除 00584，最终 162 对，均为 512×512，原图 RGB、标签 L 模式，编码 0--18。

图像与标签配对、尺寸、标签编码和原图重复检查已完成；人工浏览未发现明显问题。固定种子 42，按图像划分 train=98、val=32、test=32，人工分布检查通过。近似候选浏览未发现明显同照片裁剪或缩放重复，但没有核验人物身份互斥或穷尽近重复。

仅使用原数据库 test 来源，原 train 可按需补充；实验者反馈其中部分图像含敏感元素、筛查较费力，当前不扩充。源目录 test 不等于本实验重新划分的测试集。使用仓库数据不依赖个人下载路径。

## 3. 两种标签用途

| 用途 | 原标签 |
| --- | --- |
| 肤色统计 | `{1,2}`，皮肤和鼻子 |
| 完整脸部评价正类 | `{1,2,4,5,6,7,10,11,12}`，另纳入眼、眉、口腔和上下唇 |
| 评价忽略 | `{3}`，眼镜，不参加任何混淆计数 |
| 评价负类 | 其余类别，包括耳朵、头发、脖颈、背景等 |

`dataset.py` 生成 0/1 脸部真值与有效区域。检测器没有真值或眼镜位置参数，忽略规则只在计分时使用。不能把肤色统计样本范围改成完整脸部后继续沿用原参数。

## 4. 训练统计与解释

98 张训练图中皮肤 5,838,561 像素、鼻子 548,991 像素，共 6,387,552。原尺寸、无颜色增强，BGR 转 OpenCV YCrCb，输出顺序 Y、Cr、Cb，仅统计 Cr、Cb。像素等权，不随机采样，不先对每图均值再平均，方差为总体方差 ddof=0。

| 通道 | 均值 | 总体方差 | 标准差 |
| --- | ---: | ---: | ---: |
| Cr | 151.14949561 | 87.88095189 | 9.37448409 |
| Cb | 109.11018055 | 89.93800609 | 9.48356505 |

均值与标准差作为颜色判别的统计依据，不自动确定 k。Cr/Cb 相关系数约 -0.7644，联合分布呈倾斜趋势；单独限制两通道会忽略部分相关关系。训练肤色矩形覆盖率不等于最终脸部 Recall、F1 或 IoU。样本像素多也不代表同等数量的独立人物或场景。

当前 `configs/skin_params.json` 已精简为数值和基本统计口径，不再包含原完整训练 ID、源码哈希或 threshold_k。原完整参数与核查证据仍在 A 提交及报告 sources/part_a 副本。不要把历史哈希解释为当前精简模型文件的哈希。

## 5. 当前运行与复核

在项目根目录执行：

```text
python -m pip install -e .
face-pixel-prepare
face-pixel-fit --params results/refit/skin_params.json --output results/refit
python -m unittest discover -s tests -p test_skin_statistics.py -v
```

重算建议使用新结果目录，不覆盖现有参数和原始图表。当前重算入口生成参数、逐图计数、频数和两张分布图，不生成历史独立复核 JSON 或初步分析附加统计。整合代码不再带旧哈希和人工审核门禁；当前原始固定数据及审核记录仍保留。可重建派生数据，但不需要重新筛选或划分。

2026-11-02 使用当前整合代码重算，98 张、6,387,552 像素及六个统计值与现有参数完全一致；5 项统计小数组检查包含在通过的 57 项整合检查内。原独立复核记录 [verification.json](../results/skin_statistics/verification.json) 对应 A 当时的完整参数，不因整合而覆盖。

旧的导入、人工检查册、随机划分和独立复核脚本可在 A 提交 `1389a77` 查看，当前包不再提供这些工具或 `part_a_common.py`。它们不是使用已提交数据的前置条件。

## 6. 后续职责边界

A 的数据、颜色统计、分布解释和报告内容已交付。分割规则、k、形态学方法及验证选择由 B 负责；独立测试、附加比较、结果解释和报告整合由 C 负责。当前已有原方案和补全方案，进展见 [完整运行逻辑](完整运行逻辑与独立测试.md)。本文不重新规定 B/C 的细分排期或工作方式。
