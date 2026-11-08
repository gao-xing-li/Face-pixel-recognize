# 全图像形态学对照图册

双击 index.html 即可离线浏览。分享时复制整个目录。

覆盖 98 张训练图、32 张验证图、32 张原测试图；默认展示验证集。使用现有 original/final 配置重新生成固定预测，没有选参。验证/测试的 128 张预测及四组总体指标与历史归档一致。训练集只作描述，原测试集是开发后比较。

images/ 为可单独分享的六宫格原尺寸 PNG；per_image.csv 为逐图指标，summary.json 区分总体像素指标与逐图均值，provenance.json 保存配置及代码来源。

重新生成：在项目根目录执行 python tools/build_review_gallery.py --output results/新的目录名。
