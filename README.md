# Face-pixel-recognize

基于颜色信息的脸部分割实验。B 集成分支已接入 main 公共约定、A 的正式数据与统计、C 的共用评价模块。

## 当前状态

A 提供 162 对图像/原始标签，划分为训练 98、验证 32、测试 32。B 完成 36 组验证比较并冻结方案；C 的独立测试和最终报告尚待完成。

最终方案：**Cb/Cr 双通道均值 ± 1.25 标准差 → 最大 8 连通域 → 填孔**。32 张验证图的全局 F1=0.727013、IoU=0.571108；纯阈值基线最优 k=1.5，F1=0.592240、IoU=0.420696。这些是验证结果，不是测试成绩。

- [B 正式交接说明](docs/PartB交接说明.md)
- [B 算法与验证报告章节](report/B算法与验证.md)
- [验证记录、曲线与案例](artifacts/part_b/README.md)
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
.venv/bin/face-pixel-b validate --output results/validation-reproduction
```

prepare 生成 `data/processed/part_b/manifest.csv`、binary255 真值/有效区域和 `adapter.json`，校验 324 个源 PNG 的哈希与尺寸。本地已经生成，重复运行会拒绝覆盖；新输出目录需在后续命令中同步指定 `--manifest`。

validate 只读取 validation 图像与标签，使用 `configs/validation_plan.json` 的固定比较范围，以 C 的全局 F1 选参。每个候选保留逐图计数与汇总。源代码和验证计划需已有 Git 提交，以记录可追溯的代码版本。每次运行使用新的输出目录。

## 固定配置生成预测

```bash
.venv/bin/face-pixel-b predict --config configs/final_config.json \
  --split validation --output results/final-validation-reproduction
.venv/bin/face-pixel-b predict --config configs/final_config.json \
  --split test --output results/final-test-reproduction
```

预测入口只读取原图、模型和配置，不读取真值、不计算指标。输出为 `predictions.csv`、二值 PNG、配置副本与运行记录。基线使用 `configs/baseline_config.json`。本地已生成 `results/final-test/` 和 `results/baseline-test/`，各有 32 张固定预测。

## C 独立测试

C 核查数据、标签与固定方案后运行；此命令尚未用于本次正式测试：

```bash
.venv/bin/face-pixel-evaluate \
  --manifest data/processed/part_b/manifest.csv \
  --predictions results/final-test/predictions.csv \
  --split test --config-id part-b-v1-final \
  --run-config configs/final_config.json \
  --gt-encoding binary255 --pred-encoding binary255 --valid-encoding binary255 \
  --output results/evaluation-final-test
```

若使用上面的 reproduction 预测目录，同步修改 `--predictions`。评价基线时，改为 `results/baseline-test/predictions.csv`、`configs/baseline_config.json`、`part-b-v1-baseline`，另设输出目录。

B 已用同一批量入口复核保存的验证预测，基线和最终方案的混淆矩阵与验证搜索完全一致。完整命令见 B 交接说明。

## 标签、指标与范围

建模取样为皮肤/鼻子 `{1,2}`；脸部评价正类为 `{1,2,4,5,6,7,10,11,12}`；眼镜 `{3}` 忽略，其他类别为负类。推理不得利用真值。输入 BGR uint8，OpenCV YCrCb 通道顺序为 Y、Cr、Cb，阈值使用标准差。

指标采用 C 的实现：零分母为 null；全局指标先累计计数，逐图平均排除各指标未定义项并报告数量。最大连通区域不保证是人脸，仍存在大片漏检和背景/脖子误检。结论限定于 A 主观筛选的样本，不能推断官方族群属性或称为身份识别。

## 检查

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/verify_skin_statistics.py
```

36 项自动检查通过。统计复核会更新 A 的 verification 文件；本次副本为 `artifacts/part_b/training_verification.json`。

环境、派生掩膜、大量预测和完整案例图被 Git 忽略，可由命令复现。小型验证成果位于 `artifacts/part_b/`。旧的模拟准备代码保留在本地 `prep/b-bootstrap` 分支，不参与正式结果。课程提交时间按原通知核对：11 月 2 日（含）前。
