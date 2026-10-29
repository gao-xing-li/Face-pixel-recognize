# Face-pixel-recognize

数字图像处理实验：基于颜色信息的亚洲人脸分割。

项目以肤色统计模型为基础，在多幅图像上完成像素分割、结果展示与评价。优先使用 YCbCr 颜色空间和肤色均值、方差统计，先完成基础版本，再根据验证效果选择辅助处理。主要使用 Python、OpenCV、NumPy、Matplotlib。

当前 C 分支已接通 **A 数据 → B 固定检测方案 → C 独立评价**。A 固定版本提供 162 对原图与标签及 98/32/32 划分；B 的基线与最终方案已在验证集完全复现，再对 32 张测试图独立评价。最终方案测试 F1=0.706933、IoU=0.546710；源数据、算法和参数未根据测试结果修改。

## 文档入口

| 文档 | 用途 |
| --- | --- |
| [分工文档](分工文档.md) | 整合后的执行方案，明确职责、技术路线、交接和验收要求 |
| [公共接口 v0.2](docs/interfaces.md) | 已实现评价接口、数据/预测清单、指标口径及 A/B 接入建议 |
| [C 模块启动与交接](docs/C模块启动与交接.md) | C 当前可提供给 B 的内容、A/B 交接清单及后续工作 |
| [Part A 接入核查与 B 接入方式](docs/PartA接入核查.md) | 固定数据版本、标签转换、真实数据核查结果及运行命令 |
| [完整运行逻辑与独立测试](docs/完整运行逻辑与独立测试.md) | A/B/C 数据流、最终算法、运行入口、测试结果与失败案例 |
| [预测可视化与补充评估](docs/预测可视化与补充评估.md) | 抠图、误差叠加、逐图稳定性、边界与亮度分组 |
| [分工方案对比与采纳说明](docs/分工方案对比与采纳说明.md) | 逐项说明两版差别、最终采用哪一部分及调整理由 |
| [补充分工方案原文](docs/参考方案/补充分工方案.md) | 保留本次提供的另一版分工，供追溯对照 |

## 本次整合采用的内容

- 以补充分工方案为执行主线：A 负责数据与建模，B 负责分割、验证调参和代码整合，C 负责独立测试、分析和报告整合；各负责人自主决定内部实现。
- 采用补充方案的推进方式：先跑通基础流程，再按验证效果选择形态学、连通区域等辅助处理；B 自行准备验证评价程序，不等待 C 才能调参。
- 保留仓库原方案的协作底线：颜色通道和参数含义一致、样本 ID 与标签一一对应、训练/验证/测试分开、指标定义一致、最终配置可复现。
- 将原方案的固定接口、文件名和阈值候选范围改为建议；将数据集选择、辅助处理和最终参数交给相应负责人。

## 任务范围

本实验评价的是**像素级肤色分割**。肤色掩膜可能包含手臂等其他皮肤，不能直接当作人脸定位或身份识别结果。“亚洲人脸”的样本依据及实际适用范围需要在数据说明中写清楚。

正式实验按“准备数据与划分 → 训练肤色统计 → 验证选参 → 固定配置 → 独立测试 → 报告整合”推进。B 后续补齐检测、调参及完整实验的复现命令。

## 安装与首次检查

需要 Python 3.10 或更新版本，建议在独立虚拟环境中运行。C 的评价入口只需 NumPy 与 Pillow：

```text
python -m pip install -e ".[analysis]"
python -m unittest discover -s tests -v
face-pixel-demo --output results/synthetic-demo
```

demo 使用三个手工小掩膜，自动核对总计数 TP=3、FP=1、TN=9、FN=1、有效像素=14，并输出指标表与对照图。所有输出明确标记 **SYNTHETIC DEMO**，不能作为真实实验性能。每次运行使用新输出目录，避免覆盖结果。

`analysis` 额外安装 OpenCV headless 与 Matplotlib，用于补充评估和作图；只调用原有像素评价函数时，最小依赖仍可通过 `requirements.txt` 安装。

需要 OpenCV 和 Matplotlib 开发 A/B 模块时，可安装：

```text
python -m pip install -e ".[detection]"
```

## B 可直接复用的评价函数

```python
from face_pixel import evaluate, summarize

metrics = evaluate([[1, 0, 1], [0, 0, 0]], [[1, 1, 0], [0, 0, 0]])
# 预测在前、真值在后：TP=1、FP=1、TN=3、FN=1
summary = summarize([metrics])
```

掩膜必须为同尺寸二维整数/布尔 0/1。函数返回 Accuracy、Precision、Recall、F1、IoU 及计数；分母为零返回 None。`global` 先累加像素计数，`image_mean` 为单独标注的逐图平均，详见接口文档。

## 先接入 A 的真实数据

A 的原始标签为 0–18 多类编号，清单使用 `label_path`、`val` 及项目根目录相对路径。适配层保留原文件及划分，生成脸部二值真值、有效区域和共用格式清单。按 A 的约定，皮肤与鼻子用于建模，脸部评价还包括五官，眼镜类别 3 不计入指标。

在 C 分支安装后，使用独立的 A 分支副本：

```text
git -c core.autocrlf=false clone --branch feat/part-a --single-branch https://github.com/gao-xing-li/Face-pixel-recognize.git ../part-a-source
git -C ../part-a-source checkout --detach 1389a77158ab484dbdbbbb4e4e2f4d1c46a9625b
face-pixel-prepare-part-a --source-root ../part-a-source --source-revision 1389a77158ab484dbdbbbb4e4e2f4d1c46a9625b --output data/processed/part_a
```

已有 A 副本时直接提供其根目录，不必重新下载。原始数据始终保留在 A 副本，派生掩膜在 C 的 `data/processed/part_a`；数据和参数不从 A 分支重复提交到本分支。详细结果见接入核查文档。

## 接入 B 的保存预测

B 已在 `feat/part-b` 提供固定代码及配置。推荐使用 C 的完整复现入口，自动核对 B 源码字节与输入哈希，重建 B 冻结的数据视图，复现两套验证结果后才运行测试：

```text
face-pixel-test-part-b run --b-root ../part-b-source --b-python ../part-b-env/Scripts/python.exe --manifest data/processed/part_a/split_manifest.csv --output results/independent-test
```

B 副本固定为 `19e1d7409991c5896dfca9a6ef03694d2276a152`；其独立环境按 B 的锁定依赖安装，详见完整运行逻辑文档。B 与 C 保持独立副本，避免把额外模块放进 B 的冻结源码目录后使哈希失效。main 继续保留公共骨架，本分支提供 C 的整合运行入口及测试证据。

```text
face-pixel-evaluate --manifest data/processed/part_a/split_manifest.csv --predictions results/final-test/predictions.csv --split test --config-id final-v1 --run-config configs/final_config.json --gt-encoding binary01 --pred-encoding binary255 --valid-encoding binary01 --output results/evaluation-final-test
```

适配后的真值和有效区域均为 `binary01`，预测编码按 B 的实际文件填写。B 验证时使用 `--split validation` 和对应的 32 张预测清单。最终方案确定后再运行测试集命令；当前入口只评价已有预测，尚未调用 B 的检测程序。输出包含逐图/总体指标、配置原文件及哈希、输入文件哈希和案例图，异常会报错并避免发布部分结果。

## 公共目录与接入责任

已有独立测试输出后，可直接生成报告插图和补充统计：

```text
face-pixel-visualize --manifest data/processed/part_a/split_manifest.csv --run results/independent-test --output results/visual-assessment
```

输出全部 32 张图的六宫格及透明抠图、代表样例总览、PNG/SVG 统计图、逐图 CSV 和分析 JSON。入口先核对原始测试输入哈希，不重新预测或调参。中文字体默认使用 Windows 微软雅黑或 Noto CJK，也可通过 `--font` 指定。

```text
src/face_pixel/
  evaluation.py       # 已实现：B/C 共用的单图指标与汇总
  data_io.py          # 已实现：统一清单及显式二值掩膜读取
  batch.py, cli.py    # 已实现：保存预测的批量评价
  demo.py             # 已实现：模拟样例生成与核对
  part_a.py           # 已实现：A 原始标签审计、脸部真值/有效区域转换
  part_b.py           # 已实现：B 固定验证复现、跨平台字节核查与独立测试
  assessment.py       # 已实现：Boundary IoU 与按图像配对 Bootstrap
  visualization.py    # 已实现：预测抠图、误差叠加与报告插图
data/                 # A 维护数据说明与唯一划分
configs/              # A 的统计、B 的固定配置，正式值待接入
results/              # 运行结果，默认不提交
report/               # A/B 提交各自章节，C 整合
tests/                # 手算、边界、配对与端到端检查
```

A 后续可在包内新增 `modeling.py`；B 可新增 `detection.py`、`validation.py` 和整合入口。内部实现自主决定，跨模块输入输出变化时同步接口文档。

补充方案记录的课程提交时间为 **11 月 2 日（含）前**，提交渠道为指定的 163 邮箱；具体收件地址和最终要求由小组按课程通知核对。

