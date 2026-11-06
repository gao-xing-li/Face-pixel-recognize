# 参数与固定配置

当前项目已包含真实训练参数和三套可运行配置，不需要外部检测副本。

| 文件 | 用途 |
| --- | --- |
| `skin_params.json` | 98 张训练图、6,387,552 个皮肤与鼻子像素的 Cr/Cb 均值、总体方差和标准差 |
| `baseline_config.json` | k=1.50，无后处理，ID `baseline` |
| `original_config.json` | k=1.25，最大区域与填孔，ID `original` |
| `final_config.json` | k=1.25，最大区域、41×41 膨胀、填孔、腐蚀，ID `final` |
| `validation_plan.json` | 原 36 组验证候选，按全局 F1 选择 |
| `morphology_plan.json` | 15 组补全候选，在原方法 F1 最多下降 0.01 的预算内最大化 Recall；记录测试样例反馈 |

颜色参数与检测配置分开保存。当前配置已精简，不含历史源码哈希门禁；标签映射由 `dataset.py` 实现。`original` 对应原独立测试的方法，`final` 对应开发后补全方法，名称不能替代结果范围说明。`configs/v1/` 和 `morphology_reference.json` 已不在当前目录，历史快照见 Git 提交或 `artifacts/`。

验证运行把选定配置写入新输出目录，不自动覆盖本目录。训练重算也建议显式指定新参数和输出路径。字段见 [实验接口](../docs/interfaces.md)，完整结果见 [运行逻辑](../docs/完整运行逻辑与独立测试.md)。
