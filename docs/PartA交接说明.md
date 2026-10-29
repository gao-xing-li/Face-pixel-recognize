# Part A 交接说明

更新日期：2026-10-29。面向 B、C，说明当前可用成果、数据与参数约定、复现方式和协作边界。文中路径均相对于项目根目录 `Face-pixel-recognize`，运行命令也以该目录为工作目录。

本分支是独立 Part A 成果，采用公共目录布局，但未接入 C 模块。原图与原始标签共 324 个 PNG 随分支纳入版本管理；清单字段 `label_path`、集合名 `val`、项目根目录相对路径和原始多类别标签语义暂时保留，后续集成再适配公共评价接口。

## 1. 成果入口与当前状态

A 已完成 162 对样本的数据整理与基本质量检查、98／32／32 划分及人工分布检查、训练肤色统计、分布图和初步分析。统计结果已通过数值测试与独立计算复核。尚未实现分割、验证选参、测试评价或正式派生二值标签；判别规则与阈值系数没有定案。

| 内容 | 文件或入口 | 用途 |
| --- | --- | --- |
| 统一配对与划分 | [split_manifest.csv](../data/split_manifest.csv) | 按样本定位图像、标签和所属集合 |
| 原图与标签 | [images](../data/raw/face_segment/images/) / [labels](../data/raw/face_segment/labels/) | 同名 PNG；均为 512 × 512，原图 RGB，标签为 L 模式类别编号 |
| 来源、标签定义、检查记录 | [metadata.json](../data/part_a/metadata.json)、[数据说明](../data/README.md) | 核查样本来源和当前约定 |
| 划分记录 | [split_report.json](../data/part_a/split_report.json) | 种子、数量、方法、相似候选记录及人工通过状态 |
| 人工检查材料（本地生成，不入库） | `results/data_review/index.html`、`results/split_review/index.html` | 运行对应生成脚本后，用浏览器打开 |
| 正式肤色统计参数 | [skin_params.json](../configs/skin_params.json) | 保存完整精度、颜色约定和训练来源 |
| 逐图计数与频数 | [per_image_counts.csv](../results/skin_statistics/per_image_counts.csv)、[histograms.npz](../results/skin_statistics/histograms.npz) | 核查像素贡献，供需要进一步分布分析时使用 |
| 图表与结果摘要 | [统计结果说明](../results/skin_statistics/README.md) | Cr/Cb 单通道分布和联合分布图 |
| 初步解释 | [PartA统计结果初步分析.md](PartA统计结果初步分析.md) | 集中性、通道相关性、图像权重及局限 |
| 统计、复核与数值测试 | [modeling.py](../src/face_pixel/modeling.py)、[计算入口](../tools/compute_skin_statistics.py)、[verify_skin_statistics.py](../tools/verify_skin_statistics.py)、[test_skin_statistics.py](../tests/test_skin_statistics.py) | 正式计算、独立复核与小数组检查 |
| 依赖与设计依据 | [requirements-part-a.txt](../requirements-part-a.txt)、[PartA设计与决策记录.md](PartA设计与决策记录.md) | 环境依赖、已确定选择及实施记录 |

拿到现有数据后，可直接读取清单与参数开始工作，不需要先重新下载、导入或划分。跨机器使用时保留上述项目相对结构；统计与复核依赖项目内数据，不依赖原作者的百度网盘下载路径。`.venv` 是本机环境，可在目标机器重新创建。

数据来自已下载的 Face_segment，选样是 A 按人物外观进行的主观筛查，并非官方族群标签。原来 163 对中的 `00584.png` 已排除，保留 `00583.png`。基本质量和划分的人工检查均反馈无明显问题；相似候选筛查未发现明显同照片裁剪或缩放重复，但没有完成人物身份核验，也不保证穷尽近重复。结论范围限于实际所选数据。

当前仅使用原始 `test` 选样；原始 `train` 可按需筛选补充，但 A 反馈其中部分图像含敏感元素、筛查更费力，目前不扩充。详情见[数据说明](../data/README.md#来源与选样说明)。

## 2. 数据清单与划分使用

| `split` 值 | 图像数量 | 用途 |
| --- | ---: | --- |
| `train` | 98 | A 的肤色统计，现有参数仅从此集合计算 |
| `val` | 32 | B 比较算法与参数 |
| `test` | 32 | C 对确定后的方案进行独立评价 |

划分约为 6∶2∶2，使用种子 `42`；按样本 ID 排序后随机打乱，一次分配。名单已由 A 人工检查通过，没有根据分割得分选择种子或重划分。原图、标签不分别移动到训练／验证／测试文件夹，集合归属以清单为准。

读取 CSV 使用 `encoding="utf-8-sig"`。关键字段为：

| 字段 | 含义 |
| --- | --- |
| `sample_id` | 稳定样本标识，用于配对、记录和追溯 |
| `image_path`、`label_path` | 相对于项目根目录的路径 |
| `split` | 本实验正式集合归属 |
| `source_split`、`source_group`、`source_filename` | 下载数据中的来源信息；其中 `source_split=test` 不是本实验测试集标志 |
| `width`、`height` | 图像及标签尺寸 |
| `label_ids_present` | 当前标签中实际出现的类别，以空格分隔 |
| `image_sha256`、`label_sha256` | 原图与标签的文件哈希 |

按清单指定路径读取，不依赖两份目录的遍历顺序。训练统计、验证选择和独立测试继续使用这套统一名单；测试集不用于选择规则或参数。如需调整样本或划分，应协调更新清单和关联记录，并检查现有统计是否需要重新计算。

## 3. 标签与评价约定

标签 PNG 保存的是 **0–18 的整数类别编号**。直接显示时接近黑色是正常现象；不能把标签当成 0/255 二值图，也不能用 `label > 0` 代表脸部。完整 19 类定义见[数据说明](../data/README.md#标签含义)。

| 用途 | 类别 ID | 含义 |
| --- | --- | --- |
| 肤色建模取样 | `{1, 2}` | 皮肤、鼻子，仅这些位置的原图颜色参与统计 |
| 最终脸部评价正类 | `{1, 2, 4, 5, 6, 7, 10, 11, 12}` | 皮肤、鼻子、双眼、双眉、口腔内部、上下唇 |
| 评价忽略区域 | `{3}` | 眼镜 |
| 最终脸部评价负类 | `{0, 8, 9, 13, 14, 15, 16, 17, 18}` | 背景、双耳、头发、帽子、耳饰、项链、脖子、衣服 |

眼镜位置既不加分也不扣分：评价时排除原标签为 `3` 的像素，不计入 TP、FP、TN、FN，也不计入相应指标的分母。不能把眼镜改成背景后继续参与评价。分割和填孔仍处理整张图，不能借助真值眼镜标签指导预测或补全。

对应的真值构造含义如下，假设已导入 NumPy 为 `np`，`label` 为读入的整数标签数组；这是标签约定示例，项目尚未提供完整评价模块：

```python
skin_sample_mask = np.isin(label, [1, 2])
face_gt = np.isin(label, [1, 2, 4, 5, 6, 7, 10, 11, 12])
valid = label != 3
# 评价时，在 valid 内比较预测与 face_gt；忽略区的预测值不影响计数。
```

评价时预测掩膜与标签应在同一尺寸、同一像素位置上比较，预测值需明确对应人脸／非人脸；不要将保存的 0/255 可视化值直接当类别编号使用。若自行变换尺寸，应保持离散标签语义与图像对齐。

原始标签保持不变，`skin_sample_mask` 是取样位置，不是已预测出的完整脸部。眼睛、嘴部等没有参与肤色统计，因此基础肤色输出与最终脸部评价目标的差别需要在结果解释中保留。B、C 可协商评价实现、跨图聚合和零分母处理，当前 A 没有为这些细节新增固定方案。

## 4. 颜色转换与统计参数

### 4.1 使用现有参数的输入条件

| 项目 | 当前口径 |
| --- | --- |
| 输入 | 8 位 `uint8`、BGR 顺序、原图尺寸 |
| 转换 | `cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)` |
| 转换输出顺序 | 通道 0 为 Y，通道 1 为 Cr，通道 2 为 Cb |
| 统计通道 | Cr、Cb；均为 0–255 编码值 |
| 颜色预处理 | 无；未做白平衡、亮度增强或平滑 |
| 采样与权重 | 全部训练肤色像素，每像素等权；未随机采样、未逐图等权 |
| 统计对象 | 98 张训练图，合计 6,387,552 个皮肤与鼻子像素 |
| 方差 | 总体方差，`ddof=0`，分母为总取样像素数 `N` |

若用 Pillow 读取 RGB，需先转换为 BGR，再使用上述转换；不能将 RGB 数组直接按 BGR 解释。也不要将通道值归一化到 0–1 后直接与当前参数比较。

在项目根目录运行时，下例按清单读取一张验证图，展示可处理中文路径的读图与通道访问方式，不包含判别规则：

```python
import cv2
import numpy as np
import csv
from pathlib import Path

root = Path.cwd()
with (root / "data/split_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
    row = next(r for r in csv.DictReader(f) if r["split"] == "val")
image_path = root / row["image_path"]
label_path = root / row["label_path"]

bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
cr = ycrcb[:, :, 1]
cb = ycrcb[:, :, 2]
label = cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
```

### 4.2 参数字段

使用 JSON 中的完整精度，以下数值仅供核对：

| 字段 | 近似值 | 含义 |
| --- | ---: | --- |
| `mean_cr` | 151.14949561 | Cr 均值 |
| `variance_cr` | 87.88095189 | Cr 总体方差 |
| `std_cr` | 9.37448409 | Cr 标准差 |
| `mean_cb` | 109.11018055 | Cb 均值 |
| `variance_cb` | 89.93800609 | Cb 总体方差 |
| `std_cb` | 9.48356505 | Cb 标准差 |

`threshold_k` 当前为 JSON `null`（Python 读取为 `None`），表示尚未选择，不能直接当作可用数值。矩形判别、双通道同时满足及 `k=2` 都没有被确定为正式算法。若 B 采用“均值 ± k 倍离散尺度”的规则，应使用 `std_*`，不能用 `variance_*` 代替。

JSON 还保存 `training_sample_ids`、`manifest_sha256`、`training_content_fingerprint`、统计脚本哈希和依赖版本，用于追溯参数来源。`histograms.npz` 包含 `cr[256]`、`cb[256]` 和 `joint_cr_cb[256,256]`，数组值为整数像素计数，联合数组轴顺序是 **Cr、Cb**；图上则是横轴 Cb、纵轴 Cr。

分布图中的均值 ± 1 标准差是参考线，初步分析中的示例范围覆盖率只描述训练皮肤与鼻子像素。它们不是最终脸部的评价指标，也不能证明某个阈值更好。当前统计存在通道负相关与低频尾部，详细解释见初步分析文档。

如果更改取样标签、训练样本、颜色空间、通道编码或颜色预处理，现有参数的适用条件也会变化，应与 A 协调重新统计。单纯对预测掩膜进行填孔等后处理不改变原始颜色统计。

## 5. 环境、运行与复核

### 5.1 环境准备

本次使用 Windows、Python 3.13.7。直接依赖锁定为 NumPy 2.5.4、OpenCV headless 5.0.0.93、Pillow 13.0.0、Matplotlib 3.11.2，见 `requirements-part-a.txt`；传递依赖未另做完整锁定。环境不入库，可在项目根目录创建：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-part-a.txt
```

创建环境前确认 `python` 对应 Python 3.13；命令直接调用虚拟环境解释器，无需激活。当前 OpenCV 为 headless 包，现有程序通过保存图片展示，不依赖 `cv2.imshow`。Python 补丁版本和各库实际版本记录在参数文件中。

### 5.2 复核现有成果

已有参数和数据可直接使用。若先确认当前保存的统计是否与输入一致：

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/verify_skin_statistics.py
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

前者使用另一条 RGB 读图与直接合并像素的计算路径复核已存结果，成功时输出 `status: passed`、`training_images: 98`、`selected_pixels: 6387552`。本次数值差为 0，允许的绝对差为 `1e-10`；它检查均值、方差、标准差、逐图和联合计数、训练名单与来源哈希，并更新 `results/skin_statistics/verification.json`。后者运行 5 项小数组数值测试，成功时显示 `OK`。

### 5.3 重新计算与其他脚本

需要重算时执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/compute_skin_statistics.py
.\.venv\Scripts\python.exe -X utf8 tools/verify_skin_statistics.py
```

统计脚本会重写参数、逐图计数、频数、两张分布图和统计结果摘要；不改图像、标签或划分。它读取已人工通过的划分，只处理 98 个训练条目；哈希、尺寸、标签或审核记录不匹配，以及某训练图没有有效肤色像素时会停止。复核脚本随后更新核查证据。初步分析文档和其补充统计不会由这两个脚本自动更新；输入或方法变化后需重新核对分析。

已有数据无需运行以下准备脚本，它们的作用是：

| 脚本 | 用途与依赖 |
| --- | --- |
| [prepare_selected_data.py](../tools/prepare_selected_data.py) | 可选的外部导入工具，显式传入 `--selected-dir`、`--original-dir`、`--label-dir`；同内容导入保留划分，不重新引入 `00584.png` |
| [split_selected_data.py](../tools/split_selected_data.py) | 重现固定 98／32／32 名单和划分浏览页；遇到不同的既有划分会停止，不静默替换 |
| [build_label_review.py](../tools/build_label_review.py) | 重建原图、彩色标签和叠加图检查册，不保存人工浏览进度 |

检查册脚本使用 Windows 微软雅黑字体，字体路径见脚本；它不影响读取现有统计参数。当前没有整合分割主程序或评价入口，不能把统计命令当作完整实验运行命令。

可选的本地检查册重建命令如下，输出目录由 Git 忽略；正常使用入库图表与参数无需重建：

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/build_label_review.py
.\.venv\Scripts\python.exe -X utf8 tools/split_selected_data.py
```

辅助脚本中的 `_bootstrap.py` 只将本项目 `src` 加入导入路径，无需安装 C 分支的 Python 包或依赖原作者下载目录。

## 6. 后续职责与协调事项

| 负责人 | 后续职责 |
| --- | --- |
| A | 维护数据与统计口径，解释参数和分布；发生相关输入变更时协助更新统计与说明 |
| B | 根据这些依据决定分割规则、验证选参及必要辅助处理，完成算法实现与整合 |
| C | 核对评价口径，使用确定后的方案进行独立测试与结果分析 |

A 的交付是数据、参数、计算程序及解释，B 自行决定判别形式、系数和辅助处理。双方需要协调的主要是数据、标签或颜色处理等会影响统计适用性的变更；评价实现的共享细节由 B、C 协商。测试数据不参与方案选择，训练覆盖率不作为测试成绩。

本说明不规定 B、C 的内部函数、参数搜索范围、具体后处理算法、逐项工作顺序或交付时限。当前也未编写面向老师的 A 部分报告初稿；已有分析和决策材料可在需要时作为素材。

目录迁移保留所有样本 ID、集合归属与原图／标签哈希。仅路径和关联文件哈希更新，当前统计数值与迁移前相同。Git 文件属性对清单、JSON 与被哈希的统计核心禁用自动换行转换，以保留跨机器核查所需的原始字节。
