# B 验证成果

所有分割指标来自 32 张 **validation** 图像，测试集尚未评价。

- `candidates.csv`：36 组全局指标。
- `validation_candidates/`：各组完整逐图计数与汇总。
- `selection.json`：预先确定的比较范围、基线与最终选择。
- `validation_curve.png`：四种处理的 F1/IoU 曲线。
- `baseline_validation/`、`final_validation/`：C 的保存预测批量入口复核结果，与 B 搜索完全一致。
- `final_validation/cases/`：按最终验证 F1 排序选出的最差两例及最佳一例，记录在 `selected_cases.json`。未用测试集选择案例。
- `adapter.json`：源/派生清单、标签定义、逐项哈希。
- `training_verification.json`：本机复核 A 训练统计的记录。

`per_image.csv` 中的 `case_path` 相对于完整评价输出目录 `results/evaluation-*-validation/`；这里只分享三张案例，其余图通过 README 命令重建。

![验证曲线](validation_curve.png)

最大连通域可能保留背景，颜色偏移仍会导致大片脸部漏检。详细分析见 [B 报告章节](../../report/B算法与验证.md)。数据来源、主观选样及身份未核验的限制沿用 A 的说明。
