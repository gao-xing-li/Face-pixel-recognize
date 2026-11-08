# 基于颜色与双椭圆几何先验的人脸分割（新稿）

## 入口与编译

- 正文入口：`main_new.tex`（生成 `main_new.pdf`，并同步至仓库根目录 `报告.pdf`）。
- 在 `report/人脸分割实验报告/` 目录下可执行 `tectonic main_new.tex` 或 `xelatex main_new.tex` 两次，生成 `main_new.pdf`。
- 章节源码在 `sections_new/`，按 `01_intro.tex` 至 `07_lab4.tex` 及 `07_conclusion.tex` 组织；图片位于同目录 `images/`，通过 `\graphicspath{{images/}}` 解析。
- 对照稿 `main.tex`、`sections/`、`main.pdf` 亦已同步补入实验四章节 `sections/lab4_optimization.tex` 并重新编译。

## 结构

| 章 | 源文件 | 内容 |
| --- | --- | --- |
| 1 | `sections_new/01_intro.tex` | 任务定义与目标、四个核心问题与研究路线、章节安排与结论预览 |
| 2 | `sections_new/02_data.tex` | 数据来源与筛选、划分与去重检查、两套标签映射、评价指标体系 |
| 3 | `sections_new/03_model.tex` | YCrCb 转换、训练集肤色统计、阈值与后处理验证选参、设计取舍分析 |
| 4 | `sections_new/04_test.tex` | 实验三基础方案测试集评估（70.69% F1）、分阶段贡献、误检归因、三类典型失败 |
| 5 | `sections_new/05_completion.tex` | 形态学膨胀—填孔—腐蚀补全设计、核尺度分析、新增像素语义分解（70.70% F1） |
| 6 | `sections_new/06_boundary.tex` | 基础模型三项先验、P-R 工作点等值线轨迹、四层失效机理与解决途径 |
| 7 | `sections_new/07_lab4.tex` | 实验四：最大连通域诊断、轮廓拟合局限、双窗黑帽双眼定位、侧向椭圆与内外双椭圆约束、低支持放宽阈值重建、分区形态学连接与验证集消融（82.89% F1） |
| 8 | `sections_new/07_conclusion.tex` | 实验总结與未来展望 |

关键实现代码共 7 段：代码 2-1 标签映射、3-1 直方图统计、3-2 阈值与填孔、4-1 混淆计数与指标、5-1 补全流程、7-1 双窗黑帽眼睛定位与两级面积比回退、7-2 低支持放宽阈值重建与双椭圆分区形态学连接。

## 依赖与注意事项

- 需要 `tikz` 与 `pgfplots`（第 6 章与第 7 章精确率—召回率工作点轨迹图）。
- 封面姓名、学号、学院、专业保持留空，提交前按个人情况填写。
