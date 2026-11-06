# 数据说明与运行准备

当前仓库已包含 162 张原图与 162 张原始标签，位于 `raw/face_segment/images/`、`raw/face_segment/labels/`，无需另取 A 分支。它们已被 Git 跟踪，当前 `.gitignore` 也明确允许这两个目录中的 PNG；派生数据和其他临时文件仍按规则忽略。

## 来源与选样说明

实验者从 Face_segment 原始 `test/real_image/2` 中依据主观外观筛选，目标是外观接近亚洲人脸的样本，不能解释为官方族群属性标签。最初 163 对，图像完全重复的 00583/00584 保留 00583、排除 00584。原图均为 512×512 RGB，标签为单通道 0--18 类别编码。图像与标签人工检查、划分浏览均通过；未核验人物身份互斥，不保证穷尽近重复。

仅使用原数据 test 来源，原 train 可按需补充。实验者反馈 train 中部分图像含敏感元素、筛查成本较高，当前不扩充。来源目录的 test 与本实验重新划分的 test 不同。详细记录见 [metadata.json](part_a/metadata.json) 与 [split_report.json](part_a/split_report.json)。

## 清单与标签

- 源清单 [split_manifest.csv](split_manifest.csv) 使用 `sample_id,image_path,label_path,split,...`，路径相对**项目根目录**，固定 train=98、val=32、test=32，随机种子 42。
- 运行 `face-pixel-prepare` 生成 `processed/face/manifest.csv`，路径相对**该 CSV 目录**，val 改为 validation，不重新划分图像。
- 统计取 `{1,2}`；脸部真值正类 `{1,2,4,5,6,7,10,11,12}`；眼镜 `{3}` 不进入评价，其余为负类。派生真值与有效区域为 0/1，原始多类标签不变。
- 预测清单路径相对预测 CSV；预测 PNG 使用 0/255，计分时声明 `binary255`。

原图、真值与预测必须尺寸和像素位置一致。当前程序检查 ID、图像路径重复、标签范围和尺寸，不再进行文件内容哈希门禁。固定数据历史去重与来源哈希仍保留在原始记录中。`split_manifest.example.csv` 为虚构示例。

`processed/` 为可再生成的本地产物，默认不提交。已有目标目录时 prepare 拒绝覆盖：复用原清单，或用 `--output` 准备到新位置。字段见 [实验接口](../docs/interfaces.md)。
