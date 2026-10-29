# 参数与固定配置

当前不提供含虚构数值的正式模型参数或最终配置。

- A 已在 `feat/part-a` 固定版本提供 `skin_params.json`。C 适配时核查其训练名单、原始清单哈希、统计脚本哈希及训练肤色像素数，不改写参数、不选择 k；原文件仍保留在 A 分支。
- B 已在 `feat/part-b` 提供 `baseline_config.json` 和 `final_config.json`。C 保留它们的原字节，通过独立 B 副本运行；本分支不另外改写正式参数。原始配置快照及哈希随独立测试结果保存于 `artifacts/part_c/`。
- 正式测试时 C 将实际配置通过 `--run-config` 交给评价入口，保存原文件和哈希。程序核对 config_id，其余语义由交接时核查。

具体字段说明见 [公共接口](../docs/interfaces.md)。模拟配置由 demo 临时生成，明确标记 `synthetic_demo: true`。
