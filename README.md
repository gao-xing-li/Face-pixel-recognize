# Face-pixel-recognize

基于颜色信息的脸部分割实验。B 集成分支已接入 main 公共约定、A 的正式数据与统计、C 的共用评价模块。

## 当前状态

A 提供 162 对图像/原始标签，划分为训练 98、验证 32、测试 32。v1 完成 36 组验证，C 已在其分支完成对应独立测试。本次 v2 又比较 15 组形态学方案；其结果为验证及开发后诊断，新独立测试需另留数据。

当前 v2 面部补全方案：**Cb/Cr 双通道均值 ± 1.25 标准差 → 最大 8 连通域 → 41×41 椭圆核膨胀 → 填孔 → 同核腐蚀**。32 张验证图的 Recall=0.862249、F1=0.719776、IoU=0.562226。相对 v1 最大连通域＋填孔，漏检像素减少 20.37%，同时误检增加，F1 下降 0.724 个百分点。新版按“F1 最多下降 1 个百分点、优先补全面部”的目标选定。

v1 的固定配置原文件保存在 `configs/v1/`，旧验证成果仍在 `artifacts/part_b/`。v1 独立测试已由 C 在其分支完成。此次 v2 由原测试样例反馈触发，因此原测试集对 v2 仅能作开发后回看，不能继续称为独立测试。详见 [形态学改进与交接](docs/形态学改进与交接.md)。

- [B 正式交接说明](docs/PartB交接说明.md)
- [B 算法与验证报告章节](report/B算法与验证.md)
- [验证记录、曲线与案例](artifacts/part_b/README.md)
- [新版形态学验证与对照图](artifacts/part_b_morphology_v2/README.md)
- [A 交接说明](docs/PartA交接说明.md)
- [公共接口](docs/interfaces.md)、[分工文档](分工文档.md)、[C 原始交接说明](docs/C模块启动与交接.md)

## 安装

在仓库根目录运行。已验证 Python 3.12.14；完整依赖快照使用 OpenCV headless 4.14.0.94，与 A 保存的训练统计独立复核差异为 0。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
```

现有本地 `.venv` 已安装，无需重建。其他电脑也可用 `python -m venv .venv` 与该环境的 pip 安装同一快照。Windows 将 `.venv/bin/` 替换为 `.venv/Scripts/`，或激活环境后直接运行命令名。实际代码需要 Python 3.11 及以上。

## 数据与验证

A 的唯一源清单 `data/split_manifest.csv` 保留原始内容。公共评价读取机械派生视图，不重划数据：

```bash
.venv/bin/face-pixel-b prepare
.venv/bin/face-pixel-b validate --plan configs/morphology_plan.json --output results/validation-morphology-reproduction
```

prepare 生成 `data/processed/part_b/manifest.csv`、binary255 真值/有效区域和 `adapter.json`，校验 324 个源 PNG 的哈希与尺寸。本地已经生成，重复运行会拒绝覆盖；新输出目录需在后续命令中同步指定 `--manifest`。

validate 只读取 validation 图像与标签。新版计划为 `configs/morphology_plan.json`：固定 k=1.25，比较两种新增形态学流程的 5/9/15/21/31/41 像素核及三个参考方案；在相对 v1 全局 F1 下降不超过 0.01 的候选中选择 Recall 最高者。纯 F1 搜索仍以旧方案最好，这个取舍有完整记录。每个候选保留逐图计数。旧版 k 搜索计划保留为 `configs/validation_plan.json`，需显式区分两版结果。

源代码和实际选用的验证计划需已有 Git 提交，以记录可追溯版本。每次运行使用新的输出目录。

## 固定配置生成预测

```bash
.venv/bin/face-pixel-b predict --config configs/final_config.json \
  --split validation --output results/morphology-v2-validation-reproduction
.venv/bin/face-pixel-b predict --config configs/final_config.json \
  --split test --output results/morphology-v2-test-reproduction
```

预测入口只读取原图、模型和配置，不读取真值、不计算指标。输出为预测清单、二值 PNG、配置副本与记录。新版纯阈值参考使用 `configs/baseline_config.json`（固定 k=1.25），旧默认行为的同源码参考使用 `configs/morphology_reference.json`。本地新版预测为 `results/morphology-v2-validation/` 和 `results/morphology-v2-test/`。旧 `results/final-test/` 等目录属于 v1，不能配新版配置评价。

## C 接入新版

先在验证集核对新版文件交接；该命令已经执行，结果与验证搜索完全一致：

```bash
.venv/bin/face-pixel-evaluate \
  --manifest data/processed/part_b/manifest.csv \
  --predictions results/morphology-v2-validation/predictions.csv \
  --split validation --config-id part-b-v2-morphology-final \
  --run-config configs/final_config.json \
  --gt-encoding binary255 --pred-encoding binary255 --valid-encoding binary255 \
  --output results/evaluation-morphology-v2-validation-reproduction
```

若使用 reproduction 目录，同步修改预测路径。原测试集的新版回看可换成 `results/morphology-v2-test/predictions.csv` 与 `--split test`，输出目录应标记 diagnostic，报告按“原测试集开发后回看”解释。新的独立泛化评价需要另留数据。

B 已用同一批量入口复核保存的验证预测，基线和最终方案的混淆矩阵与验证搜索完全一致。完整命令见 B 交接说明。

## 标签、指标与范围

建模取样为皮肤/鼻子 `{1,2}`；脸部评价正类为 `{1,2,4,5,6,7,10,11,12}`；眼镜 `{3}` 忽略，其他类别为负类。推理不得利用真值。输入 BGR uint8，OpenCV YCrCb 通道顺序为 Y、Cr、Cb，阈值使用标准差。

指标采用 C 的实现：零分母为 null；全局指标先累计计数，逐图平均排除各指标未定义项并报告数量。最大连通区域不保证是人脸，仍存在大片漏检和背景/脖子误检。结论限定于 A 主观筛选的样本，不能推断官方族群属性或称为身份识别。

## 检查

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/verify_skin_statistics.py
```

42 项自动检查通过，新增测试覆盖开放缺口封口、边缘接触、空/满掩膜、核尺寸和 F1 损失上限。统计复核会更新 A 的 verification 文件；A 参数的复核副本仍为 `artifacts/part_b/training_verification.json`。

环境、派生掩膜、大量预测和完整案例图被 Git 忽略，可由命令复现。小型验证成果位于 `artifacts/part_b/`。旧的模拟准备代码保留在本地 `prep/b-bootstrap` 分支，不参与正式结果。课程提交时间按原通知核对：11 月 2 日（含）前。
