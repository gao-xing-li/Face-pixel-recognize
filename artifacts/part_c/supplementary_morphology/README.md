# 形态学补全附加测量

原检测版本为 19e1d740，新版本固定于 40e7b484。使用同一批 32 张验证图和 32 张原测试图，复现发布的验证计数后再进行配对评价。原方法的 64 张预测与历史输出逐像素一致。

更新设计参考过原测试图 00570 的反馈，因此这里是开发后附加对比。未根据本次测量结果修改参数。原独立测试记录保持不变。

- comparison.json 为总体计数、指标、逐图变化、边界均值和配对重采样区间。
- per_image_comparison.csv 为逐图计数与指标，semantic_comparison.csv 为原始类别的新增像素来源。
- predictions 保存 128 张预测掩膜、对应清单和固定配置。
- evaluation 保存逐图评价及输入哈希，input_provenance.json 汇总来源。路径中的 ${EXPERIMENT_ROOT} 表示本地实验根目录，未上传原始人像与标签文件。
- frozen_configs 和 remote_snapshot.json 保存冻结条件与经核验的远程文件 Git blob 哈希。

复现入口为 scripts/compare_frozen_methods.py，命令见 docs/结果诊断与报告复现.md。报告中的五张新增图展示空间差异与逐图分布，总体数值统一放在表格。
