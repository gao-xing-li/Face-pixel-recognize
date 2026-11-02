# 基于颜色的人脸分割实验

检测、验证、评价和可视化已经合并到同一个 Python 包。只需要一份项目和一个 Python 环境。

## 安装和运行

在项目根目录执行。

```text
python -m pip install -e .
face-pixel-prepare
face-pixel --output results/comparison
face-pixel-visualize --manifest data/processed/face/manifest.csv --run results/comparison --before original --after final --output results/visualization
```

`face-pixel-prepare` 按已有划分生成 0/1 脸部真值与有效区域，保留皮肤、鼻子、眼、眉、口腔和嘴唇，忽略眼镜类别。原图和标签在 `data/raw/face_segment/`。

`face-pixel` 分别运行 32 张验证图和 32 张测试图，对比三份配置。预测只读取原图和颜色参数，保存掩膜后才读取真值计算指标。

| 配置 | 颜色阈值 | 后处理 |
| --- | --- | --- |
| baseline_config.json | k=1.50 | 无 |
| original_config.json | k=1.25 | 最大连通域、填孔 |
| final_config.json | k=1.25 | 最大连通域、41×41 椭圆核膨胀、填孔、腐蚀 |

只比较原方法和补全方法时执行。

```text
face-pixel --configs configs/original_config.json configs/final_config.json --output results/morphology
```

## 原理与代码

| 文件 | 实验内容 |
| --- | --- |
| src/face_pixel/modeling.py | 用训练图的皮肤和鼻子像素计算 Cr/Cb 均值、总体方差和标准差 |
| src/face_pixel/dataset.py | 原始类别转脸部真值，保持 train、validation、test 划分 |
| src/face_pixel/detection.py | 均值加减 k 倍标准差，连通域、填孔和形态学处理 |
| src/face_pixel/validation.py | 在验证集比较参数，按配置生成预测 |
| src/face_pixel/evaluation.py | TP、FP、TN、FN，Accuracy、Precision、Recall、F1、IoU |
| src/face_pixel/batch.py | 按样本编号配对并评价保存的掩膜 |
| src/face_pixel/experiment.py | 串起检测、评价和方法比较 |
| src/face_pixel/visualization.py | 原图、真值、预测、误差图及逐图分布 |
| src/face_pixel/assessment.py | 边界 IoU 和按图像配对重采样 |

输入为 BGR，转换为 OpenCV 的 Y、Cr、Cb，检测仅使用 Cr、Cb。阈值中的缩放量是标准差。检测不缩放原图，真值不传入检测函数。

代码不再计算文件或源码哈希，不限制运行时必须匹配某个 Python、OpenCV 或 NumPy 精确版本。样本编号、掩膜编码、图像尺寸和指标定义的检查继续保留。

## 参数比较与训练统计

```text
python -m face_pixel.validation validate --plan configs/validation_plan.json --output results/validation-original
python -m face_pixel.validation validate --plan configs/morphology_plan.json --output results/validation-morphology
face-pixel-fit --params results/refit/skin_params.json --output results/refit
```

选参只使用验证集。原方案按全局 F1 选参，补全方案在 F1 最多下降 0.01 的条件下最大化 Recall。已有参数位于 configs/skin_params.json，重新统计时可以保存到新目录。

## 报告与实验记录

[实验报告 PDF](report/人脸分割实验报告/main.pdf) 和同目录 LaTeX 源码保留原来的独立测试，并加入形态学补全的附加实验。原测试图上的 F1 为 70.693% 和 70.703%，Recall 为 75.548% 和 79.549%。补全方法受测试案例反馈影响，按开发后比较报告。

`artifacts/` 保留历史实验记录，里面的版本号与哈希属于当时的原始记录，不参与现在的代码执行。报告结果不因代码整合而覆盖。

更多命令见 [运行说明](docs/结果诊断与报告复现.md)，测试执行 `python -m unittest discover -s tests -v`。每次运行使用新的输出目录，便于保留不同实验。
