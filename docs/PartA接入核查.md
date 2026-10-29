# Part A 接入核查与 B 接入方式

阶段说明：本文记录 A 接入时的数据核查；后续 B 已交付固定方案，C 已完成独立测试，最新状态见[完整运行逻辑与独立测试](完整运行逻辑与独立测试.md)。

本次完成 C 的数据准备与评价接线，不涉及 B 的算法选择、参数优化或正式测试性能。所有改动位于 C 分支；A 分支及 main 的内容未在本次接入中修改。

## 1. 固定来源与接入结论

来源为 [feat/part-a 固定提交](https://github.com/gao-xing-li/Face-pixel-recognize/tree/1389a77158ab484dbdbbbb4e4e2f4d1c46a9625b)，提交号 `1389a77158ab484dbdbbbb4e4e2f4d1c46a9625b`。原始清单 SHA-256 为 `006132d1c8f59804e6883337ea57aa2c07b56b0c64fb410be5d23455262fb918`。

已独立读取全部 162 对文件，核查结果为：

| 核查项 | 结果 |
| --- | --- |
| 原图及原始标签 | 324 个文件的哈希均与 A 清单一致，模式分别为 RGB 和 L |
| 尺寸及标签记录 | 每对均为 512 × 512；逐图实际类别集合与清单一致，未发现未知类别 |
| 样本及划分 | ID 唯一，训练 98、验证 32、测试 32；沿用原划分，不重新分配 |
| 精确重复 | 原图路径和文件哈希未重复；没有据此宣称身份或近重复检查完成 |
| 来源绑定 | 原清单与元数据、划分记录、模型参数中的清单哈希一致，内容指纹匹配审核记录 |
| 训练统计来源 | 98 张训练名单及内容指纹匹配，训练皮肤+鼻子像素数为 6,387,552，与 A 参数一致 |
| 统计实现来源 | A 的统计脚本文件哈希与参数记录一致；C 没有重新计算颜色均值或方差 |
| 评价接线 | 32 张验证图使用常量预测检查，计数符合独立统计，并通过批量配对、配置快照、输入哈希及案例输出 |

结构化证据见 [audit.json](../results/part_a_audit/audit.json)、[逐图数据计数](../results/part_a_audit/per_image_counts.csv) 和 [验证接线记录](../results/part_a_audit/validation_wiring.json)。原始图片不在 C 分支重复提交。

## 2. 已采用的标签语义

完全沿用 A 的 [交接说明](https://github.com/gao-xing-li/Face-pixel-recognize/blob/1389a77158ab484dbdbbbb4e4e2f4d1c46a9625b/docs/PartA交接说明.md)：

| 用途 | 原始类别 ID |
| --- | --- |
| 肤色建模采样 | 1、2：皮肤、鼻子 |
| 最终脸部真值正类 | 1、2、4、5、6、7、10、11、12：皮肤、鼻子、双眼、双眉、口腔、上下唇 |
| 评价忽略区域 | 3：眼镜；不进入任何混淆计数或指标分母 |
| 最终脸部真值负类 | 0、8、9、13、14、15、16、17、18 |

适配输出 `gt = isin(raw_label, face_positive_ids)`、`valid = ~isin(raw_label, ignore_ids)`，均保存为 uint8 0/1 PNG。原始多类标签保持不变。有效区域只交给评价程序，不能让检测算法读取真值或忽略掩膜来指导填孔、区域筛选。

本次转换后各集合的像素计数如下，属于标签描述，不是算法成绩：

| 集合 | 图像 | 有效像素 | 脸部正类 | 脸部负类 | 忽略像素 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 98 | 25,680,618 | 7,011,098 | 18,669,520 | 9,494 |
| validation | 32 | 8,388,608 | 2,228,122 | 6,160,486 | 0 |
| test | 32 | 8,388,608 | 2,150,722 | 6,237,886 | 0 |

验证/测试集中当前没有眼镜标签像素，训练集中有 9,494 个；忽略逻辑仍保留，并用显式包含眼镜的小数组检查。另已抽看一张验证样例和一张含眼镜的训练样例，脸部真值与忽略区域转换符合 A 的定义；这不替代全量人工逐像素审核。

## 3. 两种清单如何衔接

| A 原始格式 | C 派生格式 |
| --- | --- |
| `label_path` 指向 19 类标签 | `mask_path` 指向脸部 0/1 真值，另有 `valid_mask_path` |
| `split=val` | `split=validation`，只改名称，样本归属不变 |
| 路径相对于项目根目录 | 路径相对于派生 CSV 所在目录 |
| A 的 `source_split=test` 表示下载包来源 | 保留为 `source_dataset_split`；原本的实验集合名保留为 `part_a_split` |

不直接修改 A 的 CSV，以免使 A 的参数和审核记录中的哈希失效。`audit.json` 同时记录原清单哈希、派生清单哈希、标签映射哈希和与机器路径无关的规范化内容指纹。跨机器可重新运行适配，B 配置记录当次实际使用的派生清单版本即可。

## 4. B/C 的运行步骤

在 C 分支完成 `python -m pip install -r requirements.txt`。已有 A 副本时跳过下载；否则在 C 项目根目录运行：

```text
git -c core.autocrlf=false clone --branch feat/part-a --single-branch https://github.com/gao-xing-li/Face-pixel-recognize.git ../part-a-source
git -C ../part-a-source checkout --detach 1389a77158ab484dbdbbbb4e4e2f4d1c46a9625b
face-pixel-prepare-part-a --source-root ../part-a-source --source-revision 1389a77158ab484dbdbbbb4e4e2f4d1c46a9625b --output data/processed/part_a
```

适配程序检查固定 Git 版本及相关输入是否存在已跟踪修改，随后核查来源、生成掩膜。已有输出目录会拒绝覆盖。源目录和输出目录可以自选；跨 Windows 盘符时图片路径可能使用绝对路径，迁移后重新适配。

```text
data/processed/part_a/
├── split_manifest.csv    # 162 条公共格式记录
├── masks/                # 162 张脸部真值
├── valid/                # 162 张有效区域
├── label_mapping.json    # A 的完整映射与忽略规则
├── per_image_counts.csv  # 每图标签描述统计
├── audit.json            # 输入来源、哈希和核查结果
└── README.md
```

B 使用 A 的原图和正式肤色参数实现检测，在原有 32 张 `val`（公共名称 `validation`）上选参。输出该集合的 `sample_id,pred_path` 清单，然后运行：

```text
face-pixel-evaluate --manifest data/processed/part_a/split_manifest.csv --predictions results/b-validation/predictions.csv --split validation --config-id b-validation-v1 --run-config configs/b_validation.json --gt-encoding binary01 --pred-encoding binary255 --valid-encoding binary01 --output results/b-validation-evaluation
```

预测若保存为 0/1，改用 `--pred-encoding binary01`。配置文件至少包含与命令一致的 `config_id`；正式交接还应记录实际代码、A 参数及原始清单版本、派生清单/映射版本、阈值、全部处理步骤和验证选参依据。

固定方案交给 C 后，才用同一流程处理 32 张测试图，改用 `--split test` 及对应预测清单。A 参数中的 `threshold_k` 仍为 null，本次没有填入 k，也未形成任何测试性能结论。

## 5. 当前仍需 B 交付

1. 可运行检测代码或示例预测及其编码/尺寸约定，用真实预测替换接线检查中的常量掩膜。
2. 验证集选参记录和实际采用的完整处理流程。
3. 正式固定代码、配置及模型参数版本，供 C 复现独立测试。

C 已具备数据读取、脸部真值/有效区域、指标统计及批量输出条件。接线检查只验证程序，不作为检测基线、调参比较或报告中的性能成绩。
