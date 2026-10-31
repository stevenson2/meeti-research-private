# A/B PTQ 与室早证据捕获：分析代码

本目录保留 2026-10-29/30 实际使用的十个 PTQ、聚合与计算复核脚本。完整正负结果见 [研究报告](../../docs/reports/pvc_evidence_ptq_20261030/RESULTS.md)。

这些文件供审阅计算逻辑。它们来自本地实际运行文件，未在发布时重新执行实验；仅将机器路径改为环境变量，并统一 LF 换行。原文件及发布文件 SHA256、逐文件替换范围见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。没有改变门限、分数、候选规则、患者分母或停止规则。

## 文件与依赖

- snapshot/work/ptq_repair/：一次 fit-only MSE 范围修复、工作点评价、独立计数复算和整数图检查。
- snapshot/work/full_background/：160 段完整背景的网格、top3、复核负担和同池随机期望计算。
- snapshot/work/replay_repair/：五人修复前后对照与最终输入/名单复核。
- snapshot/work/ideal_r_diagnostic/：固定参考 R 替换、乐观保留和第三背景比较，以及另一路名单重构。

环境为 Python 3.12；量化使用 TensorFlow 2.21.0 的 TFLite/MLIR 内部接口及 flatbuffers，分析使用 NumPy。内部接口有版本依赖，不是通用 PTQ 软件包。Windows 与 WSL 的路径应分别配置。

使用到外部资产的脚本要求显式设置 ECG_INT8_SOURCE、ECG_SMALLMODEL_SOURCE、ECG_EXPAND200_SOURCE、ECG_CROSSLEVEL_SOURCE、ECG_DEV_SOURCE、ECG_PRODUCT_ROOT 中相关项。相对布局仍按 snapshot/work 与 snapshot/outputs 对应。不要直接在保存原结果的目录上运行这些会写出文件的脚本。

## 复现限制

模型、原始 ECG、逐拍标签、candidate_cache、scores、grid_scores、cases.json 和原始完整冻结文件没有上传。部分辅助模块也依赖原本地研究环境。因此仅 clone 本仓库不能端到端复现；这里提供实际分析逻辑、聚合表和可追踪的来源证明，不伪装成自含可运行数据包。

公开路径替换经过静态语法检查；本次没有运行训练、模型推理、下载或板测，也没有把后来的修复倒写成早期基线。
