# 实验接口

整个实验在同一包内运行，检测、验证和评价共用文件读取与指标定义。

```python
from face_pixel.detection import detect
from face_pixel.evaluation import evaluate, summarize

prediction = detect(image_bgr, skin_params, config)
metrics = evaluate(prediction, face_mask, valid_mask)
```

image_bgr 为 H×W×3 的 uint8 BGR 原图。三个掩膜均为 H×W 的整数 0/1，valid_mask 为 0 的像素不参与评价。预测掩膜保存成 PNG 时使用 0/255，读取评价时明确声明 binary255。检测函数不接收真值。

原始清单 data/split_manifest.csv 使用 sample_id、image_path、label_path、split，路径相对于项目根目录。准备后的 data/processed/face/manifest.csv 使用 sample_id、image_path、mask_path、valid_mask_path、source_label_path、split，路径相对于清单目录，val 统一为 validation。

预测清单包含 sample_id、pred_path。编号必须与所评价的整个 split 一致，不接受漏图、多图、重复编号和尺寸不一致。

配置只保存 config_id、threshold、preprocess、postprocess、evaluation_size。颜色参数独立放在 configs/skin_params.json，均值和标准差由训练图像统计。

run_experiment 输出 predictions、evaluation 和 experiment.json。evaluation.py 计算每张图的计数，summarize 先累加计数再计算 global，image_mean 则对逐图指标等权平均。分母为零时记为 null，不纳入该项逐图平均。
