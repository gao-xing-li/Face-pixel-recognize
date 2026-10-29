# Part B 正式验证与交接

日期：2026-10-29。本地工作分支 `feat/part-b`。接入 main `cd0cd19`、A `1389a77`、C `b944e52`；算法与验证计划冻结在 `dc87b4b`（完整提交号见配置）。当前只有验证结果；测试集已生成固定预测，尚未计算独立测试指标。

## 接口衔接与核验

A 的清单保留原始字节：`sample_id,image_path,label_path,split,...`，路径相对项目根目录，验证集名 val。B 生成 `data/processed/part_b/manifest.csv`，仅转换列名、路径基准、val→validation，并生成二值真值与有效区域。全部 162 个 ID、98/32/32 归属不变，324 个源 PNG 的哈希核对通过。映射证据见 `artifacts/part_b/adapter.json`。

建模取样 `{1,2}`；评价正类 `{1,2,4,5,6,7,10,11,12}`；忽略 `{3}`。派生 PNG 为 binary255。直接复用 C 的 evaluate/summarize；原准备代码“零分母返回 0”的临时口径已替换为 main/C 的 null 口径，不能混用旧模拟结果。

A 的六个统计值在本机 OpenCV 4.14.0 下独立复核差异均为 0。原始统计参数、建模代码和来源哈希保持一致；复核副本为 `artifacts/part_b/training_verification.json`。

## 算法与选参

Cb/Cr 同时满足均值 ± k×标准差，边界包含等号。无颜色预处理、不缩放。运行前提交 `configs/validation_plan.json`：k=1 到 3，步长 0.25，比较 none、fill、largest_fill、close_largest_fill，共 36 组。

填孔识别不与图像外部背景 4 连通的内部空洞。最大区域按 8 连通域像素面积选择，再填孔。闭运算候选使用 5×5 椭圆核、1 次、零边界，再取最大区域并填孔。所有操作只依赖预测；不能利用眼镜或五官真值。

依据 32 张验证图的全局 F1 选参；平局选更简单处理，再选较小 k。纯阈值最佳 k=1.5；最终选择 k=1.25 + largest_fill。仅表示本次候选中的最优，不声称所有方法中的最优。

| 验证方案 | Precision | Recall | F1 | IoU |
| --- | ---: | ---: | ---: | ---: |
| 基线：k=1.5，无后处理 | 0.494383 | 0.738395 | 0.592240 | 0.420696 |
| 最终：k=1.25，最大连通域＋填孔 | 0.648591 | 0.827008 | 0.727013 | 0.571108 |

全局 F1 提高约 13.48 个百分点，IoU 提高约 15.04 个百分点。最终逐图 F1 平均为 0.706746，另行标注。36 组候选及逐图记录见 `artifacts/part_b/`。

## 交给 C 的材料

| 内容 | 位置 |
| --- | --- |
| 最终冻结配置 | `configs/final_config.json`，ID `part-b-v1-final` |
| 基线冻结配置 | `configs/baseline_config.json`，ID `part-b-v1-baseline` |
| A 正式统计 | `configs/skin_params.json` |
| 唯一源清单 | `data/split_manifest.csv` |
| 派生公共视图 | `data/processed/part_b/manifest.csv`，由 prepare 重建 |
| 最终测试预测 | `results/final-test/predictions.csv`，32 张 |
| 基线测试预测 | `results/baseline-test/predictions.csv`，32 张 |
| B 报告章节 | `report/B算法与验证.md` |

配置记录代码提交、Python 源文件哈希、源/派生清单及统计哈希、阈值、后处理、标签编码、评价尺寸和选参依据。预测入口核对源代码与输入哈希；出现不一致应核查原因，不应修改哈希字段绕过检查。后续纯文档/成果提交不改变 Python 源文件哈希。

C 在其他目录安装后，先运行 prepare，再用固定配置生成 test 预测，最后独立评价。优先使用 `requirements-lock.txt` 中已验证的依赖。推理命令的 `run.json` 记录没有读取真值或计算指标。prepare 对测试标签只做预先确定的映射与数据完整性检查，未用于调参。

## 复核命令

安装、prepare、predict 和 C 的正式 test 命令见 README。复核 B 的保存验证预测：

```bash
.venv/bin/face-pixel-evaluate \
  --manifest data/processed/part_b/manifest.csv \
  --predictions results/final-validation/predictions.csv \
  --split validation --config-id part-b-v1-final \
  --run-config configs/final_config.json \
  --gt-encoding binary255 --pred-encoding binary255 --valid-encoding binary255 \
  --output results/evaluation-final-validation-rerun
```

这一批量入口已实际检查，基线和最终方案的 TP/FP/TN/FN、Precision/Recall/F1/IoU 与验证搜索完全相同。36 项自动检查全部通过，覆盖颜色、阈值、标签映射、填孔、连通域、零分母和验证不读取测试图像/标签。

## 尚待完成

C 独立核查测试配对、标签质量、身份与近重复限制，使用固定配置评价，再整理成功/失败案例并整合报告。B 可协助复现。若根据测试结果继续改算法，当前测试集已参与开发，需要重新安排独立测试。最大连通域方法未验证多人场景。
