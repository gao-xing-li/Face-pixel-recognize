# 数据获取

本提交包保留 split_manifest.csv，原始图像与标签按需获取，不重复放入附件。

1. 下载[实验所用的固定数据版本](https://github.com/gao-xing-li/Face-pixel-recognize/archive/1389a77158ab484dbdbbbb4e4e2f4d1c46a9625b.zip)，解压。
2. 将其中的 data/raw/face_segment 文件夹复制到本项目的 data/raw/face_segment。
3. 执行 face-pixel-prepare，生成二值脸部真值和有效区域。

最终目录应为以下结构。

```text
data/
  split_manifest.csv
  raw/face_segment/
    images/00526.png ...
    labels/00526.png ...
```

清单使用 162 对样本，训练 98 张、验证 32 张、测试 32 张。若已有课程原始数据，按清单 image_path 和 label_path 放置对应文件即可，保留原文件名与配对。
