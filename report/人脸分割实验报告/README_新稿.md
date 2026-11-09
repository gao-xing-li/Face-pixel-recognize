# 基于双眼定位与双椭圆形态学约束的人脸分割实验报告

## 入口与编译

- 正文入口：`main_new.tex` 与 `main.tex`（生成 `main_new.pdf`、`main.pdf`，并同步至仓库根目录 `报告.pdf`）。
- 在 `report/人脸分割实验报告/` 目录下执行 `tectonic main_new.tex` 或 `xelatex main_new.tex` 两次即可生成 PDF。
- 章节源码位于 `sections_new/`；图片位于同目录 `images/`，通过 `\graphicspath{{images/}}` 解析。

## 章节结构

| 章 | 源文件 | 内容 |
| --- | --- | --- |
| 1 | `sections_new/01_intro.tex` | 前期工作回顾与实验动机：任务与数据概述、实验三方法与结果简述、实验四改进方向 |
| 2 | `sections_new/02_data.tex` | 最大连通域分阶段检查与直接轮廓椭圆尝试：分阶段像素对照、弱开闭预操作与直接轮廓拟合局限 |
| 3 | `sections_new/03_model.tex` | 基于灰度黑帽变换与位置窗口的双眼定位：黑帽暗部提取、三轮定位对比与均值诊断、两级面积比回退 |
| 4 | `sections_new/04_test.tex` | 由双眼生成脸部椭圆与内外双椭圆空间约束：双眼局部坐标系、侧向姿态候选、内外双椭圆分工 |
| 5 | `sections_new/05_completion.tex` | 低支持度放宽肤色重选与分区形态学连接：内椭圆低支持判别、放宽阈值受限重建、内外分区形态学连接 |
| 6 | `sections_new/06_boundary.tex` | 验证集结果对照、消融分析与样本讨论：各轮消融结果、P-R 图、典型样本对照、当前局限与后续方向 |
| 7 | `sections_new/07_conclusion.tex` | 实验总结 |

## 注意事项

- 需要 `tikz` 与 `pgfplots`（第 6 章精确率—召回率工作点对比图）。
- 封面姓名、学号、学院、专业保持留空，提交前按个人情况填写。
