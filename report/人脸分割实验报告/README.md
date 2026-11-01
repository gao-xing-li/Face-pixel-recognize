# 基于颜色的人脸分割实验报告

本稿使用 SYSU_Latex_Template 模板，包含分割算法、验证选参、独立测试、结果分析和实验总结。实验目的、相关原理与数据准备章节保留空白，封面姓名、学号、学院和专业留空。

## 阅读与编辑

- 阅读 main.pdf。
- 编辑入口为 main.tex，使用 XeLaTeX 编译，重复编译可更新目录和图表引用。
- sections/background.tex 保存待补充的背景章节。
- sections/method_validation.tex 为算法设计与验证选参。
- sections/test_analysis.tex 为独立测试与结果分析。
- sections/conclusion.tex 为实验总结。
- images 保存正文使用的图片及其 SVG 版本。

验证集最终 F1 为 72.70%，测试集最终 F1 为 70.69%，两者在正文中分别报告。00570 等四个案例保留完整对照，参数曲线、逐图 F1 分布、边界分布和重采样区间展示表格之外的信息。纯数值重复的图片已移出正文。

统计与图表依据记录于 sources。参考稿、模板版权和许可随源码保留。模板副本修正了过时的 hyperref 类选项和书签布尔值，版式沿用原模板。
