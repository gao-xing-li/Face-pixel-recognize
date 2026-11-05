# 第三次 Project，基于颜色的人脸分割

报告为 李维政 24325143_郑立峰 24325341_李睿 24325142_第三次 project.pdf，本包包含实验源码、参数配置和数据划分清单。

## 复现步骤

先按 [数据获取说明](data/README.md) 放入原图和标签，再在本 README 所在目录执行。

```text
python -m pip install -e .
face-pixel-prepare
face-pixel --output results/reproduce
face-pixel-visualize --manifest data/processed/face/manifest.csv --run results/reproduce --before original --after final --output results/reproduce-visualization
```

默认比较三种方法，纯颜色阈值、最大连通域与填孔、形态学补全。预测、评价表、图像对照和补充统计会自动生成，附件不重复包含这些结果。完整分析与图表已收录于 PDF。

## 代码与参数

- src/face_pixel/detection.py 为颜色阈值与区域处理。
- src/face_pixel/evaluation.py 为混淆计数与五项指标。
- src/face_pixel/experiment.py 连接预测和评价，validation.py 实现验证选参。
- src/face_pixel/modeling.py 统计训练颜色，dataset.py 转换真值。
- src/face_pixel/visualization.py 与 assessment.py 展示效果、边界和逐图波动。
- configs/skin_params.json 保存训练统计，baseline、original、final 配置分别对应三种方法。

按需要重新选择参数和运行阶段诊断。

```text
python -m face_pixel.validation validate --plan configs/validation_plan.json --output results/reselect-original
python -m face_pixel.validation validate --plan configs/morphology_plan.json --output results/reselect-morphology
python scripts/analyze_error_stages.py --output results/stages
python -m unittest discover -s tests -v
```

输出使用新目录。原方法独立测试 F1 为 70.69%，补全后的开发后比较 F1 为 70.70%，报告中分别说明证据范围。
