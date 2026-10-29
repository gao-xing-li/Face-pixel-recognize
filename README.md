# Face-pixel-recognize

数字图像处理实验：基于颜色信息的亚洲人脸分割。

本分支为 **Part A 独立成果分支**，从公共骨架引入前的共同提交 `4cbd57b` 开始，仅遵循目录布局，不引入 C 评价实现。包含全部 162 张原图和 162 张原始标签，无需另行下载数据。当前清单仍使用 `label_path`、`val` 和项目根目录相对路径；公共评价接口接入留到后续整合。

目录布局：`src/face_pixel/modeling.py` 为统计核心，`tools/` 为命令和准备工具；`data/split_manifest.csv` 为唯一划分，`data/part_a/` 为元数据，`data/raw/face_segment/` 为原图与标签，`configs/` 为参数，`results/skin_statistics/` 为小型正式统计成果。

`results/data_review/` 和 `results/split_review/` 为本地生成检查册，不入库。环境安装后可使用虚拟环境 Python 执行对应脚本重建（命令见交接说明，Windows 默认使用微软雅黑字体）。


B、C 接手可先阅读 [Part A 交接说明](docs/PartA交接说明.md)，其中集中说明成果入口、数据与标签、统计约定、复现方式和协作边界。

项目以肤色统计模型为基础，在多幅图像上完成像素分割、结果展示与评价。优先使用 YCbCr 颜色空间和肤色均值、方差统计，先完成基础版本，再根据验证效果选择辅助处理。主要使用 Python、OpenCV、NumPy、Matplotlib。

当前阶段为**A 部分肤色统计与初步分析已完成**：162 对样本按种子 `42` 划分为训练 98、验证 32、测试 32，人工检查已通过。已使用 98 张训练图的 6,387,552 个皮肤与鼻子像素计算 Cr/Cb 参数并生成分布图，独立复核通过。尚未实现分割、选定阈值系数或运行验证与测试评价。

## 肤色统计结果与运行

- [实现设计与决策记录](docs/PartA设计与决策记录.md)：第 6 节先记录设计，再记录实现结果。
- [统计结果初步分析](docs/PartA统计结果初步分析.md)：解释集中性、通道关系、图像权重与统计局限；规则和阈值由 B 验证确定。
- [统计结果与分布图](results/skin_statistics/README.md) · [正式参数](configs/skin_params.json) · [独立复核记录](results/skin_statistics/verification.json)
- [统计核心](src/face_pixel/modeling.py) · [计算入口](tools/compute_skin_statistics.py) · [复核程序](tools/verify_skin_statistics.py)

本次验证使用 Python 3.13.7。环境不入库，新机器先以 Python 3.13 创建环境并安装 [requirements-part-a.txt](requirements-part-a.txt)：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-part-a.txt
```

在项目根目录运行，无需激活环境：

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/compute_skin_statistics.py
.\.venv\Scripts\python.exe -X utf8 tools/verify_skin_statistics.py
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

首次检查已有成果也可以直接运行复核命令。预期训练图像数为 98、有效肤色像素数为 6,387,552，复核输出 `status: passed`，数值测试显示 `OK`。统计命令会重写正式统计文件；初步分析不会自动随重算更新，详情见交接说明。

参数采用 8 位 OpenCV YCrCb（输出 Y、Cr、Cb），不做颜色增强，全训练肤色像素等权，总体标准差 `ddof=0`。`k` 暂未选择；图中的均值 ± 1 标准差只作分布参考，不是正式分割阈值。

## 已选数据

- [数据说明与标签映射](data/README.md)
- [配对与划分清单](data/split_manifest.csv)：路径相对于项目根目录，`split` 为 `train`、`val` 或 `test`
- [划分记录](data/part_a/split_report.json) · [划分程序](tools/split_selected_data.py)；按集合检查页为本地生成的 `results/split_review/index.html`
- [来源与导入核查结果](data/part_a/metadata.json)
- [数据导入脚本](tools/prepare_selected_data.py)
- 人工检查册：本地生成的 `results/data_review/index.html`，不入库
- [检查图生成脚本](tools/build_label_review.py)：使用上述环境运行，生成本地检查册，命令见交接说明

人工检查册为本地静态网页，双击 `results/data_review/index.html` 可用浏览器打开，无须启动服务。详情页支持上一张、下一张及原始尺寸查看。页面不保存检查进度；请另行记录问题文件名和现象。生成检查图不代表人工检查已完成，也不会修改源图、标签或数据划分。

2026-10-29，成员 A 完成人工目视检查，反馈无明显问题；结合程序核查，数据准备与基本质量检查阶段通过。划分前已筛查相似候选并查看，未发现明显同照片裁剪或缩放重复；这不代表已核验人物身份或穷尽近重复。

**肤色建模取样类别为 `{1, 2}`；脸部评价正类为 `{1, 2, 4, 5, 6, 7, 10, 11, 12}`，眼镜 `{3}` 为忽略区域，其余类别为负类。** 眼镜既不计入 TP/FP/TN/FN，也不计入指标分母；推理和填孔不使用真值标签。原始多类别标签继续保留，尚未生成派生二值标签或实现分割。

## 文档入口

| 文档 | 用途 |
| --- | --- |
| [Part A 交接说明](docs/PartA交接说明.md) | B、C 使用 A 成果的统一入口 |
| [分工文档](分工文档.md) | 整合后的执行方案，明确职责、技术路线及本分支实际范围 |
| [Part A 设计与决策记录](docs/PartA设计与决策记录.md) | 当前已明确的数据选择、已有方向、实际准备状态与未定事项 |
| [分工方案对比与采纳说明](docs/分工方案对比与采纳说明.md) | 记录早期方案取舍；当前状态以 A 决策记录为准 |
| [补充分工方案原文](docs/参考方案/补充分工方案.md) | 保留本次提供的另一版分工，供追溯对照 |

## 分工主线与历史方案

- 以补充分工方案为执行主线：A 负责数据与建模，B 负责分割、验证调参和代码整合，C 负责独立测试、分析和报告整合；各负责人自主决定内部实现。
- 采用补充方案的推进方式：先跑通基础流程，再按验证效果选择形态学、连通区域等辅助处理；B 自行准备验证评价程序，不等待 C 才能调参。
- 保留仓库原方案的协作底线：颜色通道和参数含义一致、样本 ID 与标签一一对应、训练/验证/测试分开、指标定义一致、最终配置可复现。
- 将原方案的固定接口、文件名和阈值候选范围改为建议；将数据集选择、辅助处理和最终参数交给相应负责人。

## 任务范围

本实验以肤色统计作为基础，最终按已确定的**脸部组成区域正类与眼镜忽略区域**评价。基础肤色掩膜与最终脸部区域需区分，不能直接当作身份识别结果。“亚洲人脸”的样本依据及实际适用范围见数据说明；最新标签决定见 Part A 决策记录。

正式实验按“准备数据与划分 → 训练肤色统计 → 验证选参 → 固定配置 → 独立测试 → 报告整合”推进。完成代码后，由 B 补充安装、运行和复现命令。

补充方案记录的课程提交时间为 **11 月 2 日（含）前**，提交渠道为指定的 163 邮箱；具体收件地址和最终要求由小组按课程通知核对。
