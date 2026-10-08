#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate docs/reports/SESSION_REPORT_R16.md from the R16 result artifacts."""
import json
from pathlib import Path

ROOT = Path('/home/devcontainers/meeti')
R = ROOT / 'docs' / 'reports'


def j(p):
    return json.loads((R / p).read_text(encoding='utf-8'))


def main():
    a1 = j('r16_rr_shape/result.json')
    a2 = j('r16_rr_shape_gate/result.json')
    ph = j('r16_rr_shape_gate/posthoc_two_stage_sensitivity.json')
    aq = j('r16_rr_shape_gate/annot_qrs_sensitivity.json')
    a3_12 = j('r16_rr_seq/result_cfg12.json')
    a3_30 = j('r16_rr_seq/result_cfg30.json')
    a4 = j('r16_af_head_adapt/result_finetune.json')
    a5 = j('r16_theta_crossfit/result.json')
    a6 = j('r16_ltafdb_sensitivity/result.json')
    qc = j('r16_rr_shape/detector_qc.json')
    a4_dann3 = j('r16_af_head_adapt/result_dann_lam0.3.json')
    a4_dann1 = j('r16_af_head_adapt/result_dann_lam0.1.json')
    aug_path = R / 'r16_af_head_adapt/result_aug.json'
    a4_aug = json.loads(aug_path.read_text(encoding='utf-8')) \
        if aug_path.exists() else None

    def a3_table(d):
        lines = []
        for s, r in d['seeds'].items():
            lines.append(
                f"| {s} | {r['best_val_auc']} | {r['afdb_eval_auc']} | "
                f"{r['ltafdb_auc']} | "
                f"{r['thresholds']['0.9']['afdb_eval_op']['recall']} | "
                f"{r['thresholds']['0.9']['ltafdb_op']['recall']} | "
                f"{r['thresholds']['0.9']['device_052']['m4_alarms']} | "
                f"{r['thresholds']['0.95']['afdb_eval_op']['recall']} | "
                f"{r['thresholds']['0.95']['ltafdb_op']['recall']} | "
                f"{r['thresholds']['0.95']['device_052']['m4_alarms']} |")
        return '\n'.join(lines)

    def a4_rows(d):
        lines = []
        for s, r in d['seeds'].items():
            lines.append(
                f"| {s} | {r['best']['epoch']} | {r['ptb_fold10']['auc']} | "
                f"{r['afdb_eval']['auc']} | {r['ltafdb']['auc']} | "
                f"{r['go_afdb']} | {r['go_fold10']} | {r['go_pass']} |")
        return '\n'.join(lines)

    ph_top = ph['pass_cells'][:6]
    ph_lines = '\n'.join(
        f"| {x['window']} | {x['feature']} | {x['composite_target']} | "
        f"{x['theta2']} | {x['afdb_recall']} | {x['ltafdb_recall']} | "
        f"{x['052_events_deploy']} | {x['052_margin_deploy']} | "
        f"{x['rec_latest_events']} |" for x in ph_top)
    aq_lines = '\n'.join(
        f"| {k} | {v['theta2']} | {v['afdb_eval']['recall']} | "
        f"{v['ltafdb']['recall']} | {v['device_052']['m4_alarms']} | "
        f"{v['pass']} |" for k, v in aq['candidates'].items())
    a6_lines = '\n'.join(
        f"| {k} | {v['n']} | {v['pos']} | "
        f"{v['features']['rr_entropy']['auc']} | "
        f"{v['features']['cv']['auc']} | "
        f"{v['features']['mean_abs_drr_norm']['auc']} | "
        f"{v['features']['rr_entropy'].get('op_at_afdb_theta', {}).get('recall')} |"
        for k, v in a6['ltafdb'].items())
    a5_lines = '\n'.join(
        f"| {k} | {v['theta_summary']['range']} | "
        f"{v['pooled_operating_point']['theta_mean']} | "
        f"{v['pooled_operating_point']['afdb_eval_op']['recall']} | "
        f"{v['pooled_operating_point']['ltafdb_op']['recall']} | "
        f"{v['pooled_operating_point']['device_052']['m4_alarms']} | "
        f"{v['criterion_theta_range_le_0.30']} | "
        f"{v['pooled_operating_point']['pass_three_lines']} |"
        for k, v in a5['rr_gate']['features'].items())
    aug_line = '未运行/未完成'
    if a4_aug:
        r = list(a4_aug['seeds'].values())[0]
        aug_line = (f"seed42 fold10 {r['ptb_fold10']['auc']} / afdb "
                    f"{r['afdb_eval']['auc']} / ltafdb {r['ltafdb']['auc']} "
                    f"/ go {r['go_pass']}")

    txt = f"""# R16：Lane A 形状感知 RR 重建执行报告

日期：2026-09-10（会话跨日）。主题：按 `NEXT_SESSION_PROMPT_R16.md` 执行
A1-A6 + 事后门敏感性，回答“30 s 窗/形状特征能否把窦性心律不齐（052）
从 AF 里分出来”。烧录仍暂停；Lane B 与 `deploy_candidate_r11` 未动。
预注册见 `runs/R16_PREREG.md`（A1/A2 评测前冻结）。

## §0 执行摘要

- **A1：PASS（单特征级）**。30 s 窗 `drr_entropy` 在训练侧 θ_R90=2.9961 下
  ecg_real_052 事件级 FP=0（154 窗 0 报警），afdb 留出召回 **0.9064**≥0.90；
  10 s 窗无任何特征同时过 052 FP=0 与 afdb 召回≥0.90（0/34）。
  **关键机制**：10 s 窗的 Shannon 熵有 ~10 拍小样本上偏，把 052 的平滑呼吸性
  心律不齐推进 AF 阈值区间；30 s 窗（~34 拍）消除该上偏——052 的 30 s
  rr_entropy 分布 [p10,p50,p90]=[1.83,2.13,2.24] **完全低于** AF 留出
  [2.46,3.07,3.61]，而 10 s 时两者重叠（052 p90 2.12 vs AF p10 2.09）。
- **A2：FAIL（决定性，预注册）**。两段式 rr_entropy→形状门禁在两个变体、
  两个窗长下均未过三线。变体 A（θ1@train recall 0.90）30 s：052 已静默
  （0 事件）但 afdb 留出 0.8397 / ltafdb 0.8522 双双 <0.90；变体 B
  （θ1@0.97）30 s：Stage1 召回 afdb 0.9808 / ltafdb 0.9173 过线，但 052
  有 8 个 M4 事件，30 个 Stage2 候选在训练侧 θ2 下无一过三线。
  **Oracle 阈值诊断（事后）**：5/30（30 s_B）与 4/30（10 s_B）候选存在一个
  阈值使三线同时成立；训练侧 θ2 比 oracle 高约 0.05（cv 30 s：
  0.174 vs 0.120）。**阻塞=阈值跨域漂移，不是形状信息缺失。**
- **A2-PH（事后可行性，非预注册）**：同一两段式结构，Stage1 rr_entropy@
  train recall 0.99 + Stage2 归一化变异性@复合 train recall 0.95-0.99，
  **17/256 单元过三线**。最稳健单元=30 s `mean_abs_drr_norm`@复合 0.99：
  θ1=1.585 / θ2=0.1167，052 事件 **0**（deploy XQRS 与 PT100 双检测器，
  余量 0.0415）、afdb 留出 **0.9712**、ltafdb **0.9164**、rec_latest 0 事件。
  **标注 QRS 敏感性**（用数据库 .qrs 替代 PT100）：6/10 特征过三线
  （afdb 0.967-0.978 / ltafdb 0.932-0.944 / 052 0 事件）。
  **注意：单元选择为事后（32 特征 × 2 窗 × 4 目标），不得当部署结论。**
- **A3：FAIL**。RR 序列 1D CNN（12 RR 主口径 / 30 RR 变体，3 seeds）判别力强
  （afdb 留出 AUC 0.984-0.989、ltafdb 0.906-0.915），但 **0/3 seeds 过三线**：
  召回达标阈值下 052 总有 1-8 个 M4 事件。与 A2 同因（阈值漂移）。
- **A4：FAIL**。全网络低 LR 微调把 afdb 留出 AUC 从 R15-B3 的 0.7375-0.7404
  提升到 **0.8593/0.8690/0.8680**（fold-10 0.9736-0.9738），DANN λ=0.3
  seed42 到 **0.8890**（ltafdb 0.7328）、λ=0.1 0.8521；均 <0.90 go-line。
  按 R16 纪律 **形态学路线关闭**（不再重试）。
- **A5：FAIL（AF 头）**。B2 20% 留出概率按患者级 5 折：θ_youden 极差
  **0.577**、θ_R90tr 极差 **0.670**（判据 ≤0.30）；RR 门记录级 5 折阈值极稳
  （极差 0.002-0.243，全部 ≤0.30）但 pooled 工作点仍不过线
  （afdb 0.77-0.91 / ltafdb 0.81-0.86）→ **系统漂移定案**（不是折间抽样噪声）。
- **A6**：ltafdb **ECG1 一致优于 ECG2**（rr_entropy AUC 0.901/0.916/0.904
  vs 0.888/0.889/0.883，10/30/60 s）；30 s 最优、60 s 不再增益。
  **检测器 QC**：afdb PT100 vs .qrs 中位|差|0、within-1 84.6%；
  **ltafdb within-1 仅 65.2%、p90|差|25 拍、3.2% 窗 <3 拍**（18 窗 qrs 正常）
  → ltafdb 召回被检测器拖累（标注 QRS 复评后 ltafdb 召回升至 0.93-0.94）。
- **设备域边界不变**：仍只有 2 条正常录音、**无设备 AF 正样本**；052 静默是
  必要条件不是充分条件；灵敏度主张仍来自 afdb/ltafdb/PTB。A7（设备 AF 正样本）
  未获数据，登记跳过。
- **沿用 R15 修正**：afpdb 的 `p*` 是 PAF 前驱（记录内无 AF）**难负例**，不是
  AF 正类；`t*` 无标签；R16 未使用 afpdb/aftdb 训练（B5 池未消耗）。
  rec_latest 单独披露（post-hoc 门下 0 事件；非预注册验收）。
- **预处理假设**：单导联 lead II；afdb 用 ECG2、ltafdb 用 ECG2（A6 对比 ECG1）；
  100 Hz（afdb 250→100、ltafdb 128→100）；10 s 或 30 s 窗；RR 过滤
  0.30-2.00 s；RR 特征不做幅度重标定；AF 头用 p99 对齐到 PTB train 中位
  （0.5935 mV）；设备用 deploy-chain XQRS @250 Hz（PT100 作敏感性）。

## §A1 形状感知 RR 特征可行性（零训练）

- 特征 34 个：基线 6（rmssd/sdnn/rr_entropy/drr_entropy/cosen/sampen_m1）
  + 形状 28（滞后 1/2/3 自相关、RMSSD/SDNN、Poincaré SD1/SD2 及比值、
  pNN50/pNN20、TPR、ΔRR 符号熵、趋势斜率、SampEn(m2)、FuzzyEn(m1/m2)、
  DistEn(m2)、Katz/Higuchi 分形维、DFA α1、平滑度、归一化熵/range/IQR 等）。
- 三域统一：PTB folds1-9（19,634 条 10 s 窗，A4 检测器重算）；afdb 23 记录
  按记录奇偶切 fit/eval（fit 12,155 窗 / eval 9,643 窗，AF 3,643）；ltafdb
  84 记录采样 7,667 窗（AF 4,667）。30 s 窗仅 afdb/ltafdb/设备可提供
  （PTB 单条 10 s），30 s 阈值只用 afdb fit 拟合（披露）。
- 单特征 AF-vs-rest AUC（fit 池）与操作点见 `result.json`；
  **A1 目标（预注册）**：存在特征/LR 使 052 事件 FP=0 且 afdb 留出≥0.90。
  结果：**30 s 通过 1 个（drr_entropy，θ=2.9961，052 0 事件 / afdb 0.9064）**，
  10 s 通过 0 个；oracle 阈值下 10 s 4 个 / 30 s 11 个特征可通过。
- 不规则阴性子集（非 AF 且 rr_entropy≥1.5@10 s / ≥2.0@30 s）的
  AF-vs-不规则 AUC：形状特征整体未超过基线（详见 result.json），
  说明“把窦性心律不齐/房早等不规则阴性单独成列”后，10 s 形状维度信息不足；
  30 s 窗的归一化变异性才是有效维度。

## §A2 预注册两段式形状门禁（决定性）

| 变体/窗 | θ1 | Stage1 afdb 召回 | Stage1 ltafdb 召回 | Stage1 052 事件 | Stage2 训练侧选中 | 三线过 | oracle 可过候选 |
|---|---|---|---|---|---|---|---|
| 10s_A | 2.1175 | 0.8737 | 0.8543 | 4 | sampen_m2_r02 | 否 | 0/30 |
| 10s_B | 1.5850 | 0.9728 | 0.9314 | 8 | lr_shape | 否 | 4/30 |
| 30s_A | 2.5930 | 0.8397 | 0.8522 | **0** | lr_shape_plus_entropy | 否 | 0/30 |
| 30s_B | 1.9219 | 0.9808 | 0.9173 | 8 | lr_shape_plus_entropy | 否 | **5/30** |

- 变体 A（字面口径 θ1=训练侧 recall 0.90 的最大阈值；θ2=Stage1 阳性
  Youden）：30 s 时 052 已静默，但训练侧阈值在 afdb 留出/ltafdb 上召回不足
  （0.84/0.85）→ **Stage 2 只能降召回，无法补救**。
- 变体 B（主口径，预注册澄清：θ1=recall 0.97；θ2=复合 recall 0.90）：
  Stage1 过 afdb/ltafdb 召回线，但 052 仍有 8 个 M4 事件；30 个 Stage2
  候选（28 形状特征 + 2 LR）在训练侧 θ2 下全部失败。
- **Oracle 阈值分析（事后，必要条件下界）**：30 s_B 有 5/30 候选
  （cv/katz_fd/mean_abs_drr_norm/range_norm/sd1）存在阈值使三线同时成立；
  但训练侧 θ2 系统性高于 oracle θ2（cv：0.174 vs 0.120；mean_abs：
  0.181 vs 0.124）。**训练→评测的阈值漂移 = 本轮真实阻塞点。**

## §A2-PH 事后门敏感性（明确非预注册）

- 结构不变（Stage1 rr_entropy → Stage2 形状/归一化变异性），阈值仍只从
  训练侧推，但把 Stage1 目标放宽到 recall 0.99、Stage2 复合目标扫
  0.95-0.99。**17/256 单元过三线**（052 双检测器 0 事件 + afdb≥0.90 +
  ltafdb≥0.90）。前 6 名：

| 窗 | Stage2 特征 | 复合目标 | θ2 | afdb 召回 | ltafdb 召回 | 052 事件 | 052 余量 | rec_latest 事件 |
|---|---|---|---|---|---|---|---|---|
{ph_lines}

- 最优稳健单元：**30 s mean_abs_drr_norm**（θ1=1.585 / θ2=0.1167）：
  052 事件 0（XQRS 与 PT100）、afdb 0.9712、ltafdb 0.9164、rec_latest 0。
- **标注 QRS 敏感性**（数据库 .qrs 替代 PT100；设备 052 仍用 deploy XQRS）：

| Stage2 特征 | θ2 | afdb 召回 | ltafdb 召回 | 052 事件 | 过三线 |
|---|---|---|---|---|---|
{aq_lines}

- 解读：PT100 的 ltafdb 检测缺陷（见 §A6）压低了 ltafdb 召回；换用数据库
  标注后同一门在 6/10 特征上通过。**机制成立**：Stage 1 保召回，Stage 2 用
  归一化 RR 变异性（cv / mean|ΔRR| / RMSSD / SDNN / SD1 / Katz）拒绝平滑的
  呼吸性窦性心律不齐。
- **纪律声明**：该表是事后选择（32 特征 × 2 窗 × 4 目标），阈值数值本身来自训练侧，
  但“选哪个单元”看到了评测/设备结果；**不得当部署结论**。需要 R17 新预注册
  （固定单一特征+固定目标）并在新采集的设备数据（A7）上验证。

## §A3 RR 序列 1D CNN（小训练）

- 模型：3×Conv1D(48,k=3)→GAP→形状向量 MLP→Dense(16)→sigmoid（~15k 参数），
  类权重 balanced，Adam 1e-3，E≤30，早停（val=PTB fold9+3 afdb fit 记录），
  seeds 42/43/44。输入：因果 RR 序列 [rr/median−1, Δrr/median, ΔΔrr/median]
  + 8 形状标量（含缺失指示）。12 RR 主口径训练=PTB folds1-8 + afdb fit
  10 s 窗（26,597 窗）；30 RR 变体训练=afdb fit 30 s 窗（9,156 窗）。
- 12 RR（`result_cfg12.json`）：

| seed | val AUC | afdb AUC | ltafdb AUC | R@θ_R90 afdb | R@θ_R90 ltafdb | 052 事件@R90 | R@0.95 afdb | R@0.95 ltafdb | 052 事件@0.95 |
|---|---|---|---|---|---|---|---|---|---|
{a3_table(a3_12)}

- 30 RR（`result_cfg30.json`）：

| seed | val AUC | afdb AUC | ltafdb AUC | R@θ_R90 afdb | R@θ_R90 ltafdb | 052 事件@R90 | R@0.95 afdb | R@0.95 ltafdb | 052 事件@0.95 |
|---|---|---|---|---|---|---|---|---|---|
{a3_table(a3_30)}

- **判定：FAIL（0/3 seeds 三线）**。模型判别力（AUC）强于任何零训练特征，
  但 052 的分数分布与 AF 在召回所需阈值处重叠；换更宽松的训练侧目标
  （0.95-0.99）召回更高、052 事件更多。与 A2 同因：阈值跨域漂移。
- 负结果登记：RR 形状路线按提示词“收尾”（A3 未过，不再重试）。

## §A4 AF 头全网络修复（细调/DANN/增广）

- 数据：PTB folds1-8（17,441）+ afdb fit 9 记录（5k/5k）+ CinC2017 2,214
  + aftdb 400；验证 PTB fold9 + 3 afdb fit 记录；评测 PTB fold-10 / afdb
  留出 11 记录 / ltafdb 采样 / 设备（披露）。
- **全网络低 LR 微调**（trunk 1e-5 / head 1e-4，15 epochs，早停）：

| seed | best epoch | PTB fold-10 | afdb 留出 | ltafdb | go_afdb≥0.90 | go_fold10≥0.95 | go |
|---|---|---|---|---|---|---|---|
{a4_rows(a4)}

- **DANN**（GRL，域=PTB/afdb/CinC2017，λ ramp）：λ=0.3 seed42 afdb
  **0.8890** / fold10 0.9704 / ltafdb 0.7328；λ=0.1 seed42 afdb 0.8521 /
  fold10 0.9747 / ltafdb 0.7626。**域增广对照**：{aug_line}。
- **判定：FAIL**（afdb 留出 AUC 最高 0.889 <0.90；PTB fold-10 全部 ≥0.95）。
  全网络微调相对 R15-B3 冻结读出头（0.74）提升 +12~15pp，DANN λ=0.3 再
  +2pp，但 **未达 go-line**。按 R16 纪律 **形态学路线关闭**，Lane A 只保留
  RR 通道；设备域极性/幅度敏感性未变。

## §A5 阈值交叉拟合 + 设备域校准

- AF 头：B2 20% 留出概率（3,927 条/269 AF）按患者级 5 折（每折 ~54 AF）；
  15 个 θ_youden 极差 **0.5772**、θ_R90tr 极差 **0.6698**（判据 ≤0.30 →
  **FAIL**）。主因 seed 间差异（seed42 的 θ~0.58，seed43/44 的 θ~0.04-0.13）。
- RR 门：afdb fit 12 记录按记录 5 折，各特征 θ_R90 极差 0.0017-0.2431
  （全部 ≤0.30，**阈值稳定判据 PASS**），但 pooled 平均阈值工作点仍不过线：

| 特征 | θ 极差 | pooled θ | afdb 召回 | ltafdb 召回 | 052 事件 | 稳定判据 | 三线 |
|---|---|---|---|---|---|---|---|
{a5_lines}

- **结论**：RR 门阈值在记录级折间非常稳定，但 pooled 工作点仍失败 →
  失败来自 **fit 记录与 eval 记录的系统性分布漂移**（不是折间抽样噪声）。
  AF 头则 seed 间阈值失稳（θ 极差 0.577）。两者共同支持“spec@R90 主口径、
  点阈值只披露”。
- **设备域校准**：无用户提供的 ≥5 条设备正常录音（除 052 阴性对照与
  rec_latest），无设备 AF 正样本 → 无法构建设备校准集，登记披露。

## §A6 ltafdb 通道/长窗敏感性

| 域 | 窗数 | AF 窗 | rr_entropy AUC | cv AUC | mean|ΔRR| AUC | rr_entropy 召回@afdb 阈值 |
|---|---|---|---|---|---|---|
{a6_lines}

- **ECG1 一致优于 ECG2**（rr_entropy AUC +1.3~2.7pp；cv/mean|ΔRR| 同向）；
  30 s 最优，60 s 不再增益（N 段稀少导致 60 s 候选减少）。
- **检测器质量（`runs/r16_rr_shape/detector_qc.json`）**：
  afdb 23 记录/460 窗：中位|PT100−.qrs 拍数差| 0、p90 5、within-1 84.6%、
  within-2 88.0%、PT100<3 拍 2.6%（其中 11 窗 .qrs 正常）；
  ltafdb 30 记录/600 窗：中位 0、**p90 25**、within-1 **65.2%**、
  within-2 68.7%、PT100<3 拍 3.2%（18 窗 .qrs 正常）。
  → ltafdb 的 RR 召回数字被检测器拖累；标注 QRS 复评（§A2-PH）证实。
- 披露：ltafdb AF 富集、N 段稀少；60 s 窗要求 6 个连续同标签 10 s 窗，
  短 N 段被丢弃；本表用 20/class/record 采样，与 R15-B4 的 60/class 不同。

## §7 关键对比与教训

1. **30 s 窗是本轮决定性变量**：10 s 熵估计的小样本上偏把 052 推进 AF 区间；
   30 s 窗后 052 与 AF 的 rr_entropy 分布完全分离（052 max 2.285 <
   afdb 留出 p10 2.46）。这是 R15-B1 失败的直接技术根因之一。
2. **阻塞不是特征信息，而是阈值跨域漂移**：A2 oracle 阈值证明形状阶段
   可以同时满足三线（5/30 候选）；A5 记录级交叉拟合证明阈值在 fit 折间
   很稳，但 fit→eval 系统性漂移（afdb 留出比 fit 低、ltafdb 更低）。
   R15-B2/B3 的“校准失稳”在本轮被进一步定位为**域分布漂移**。
3. **两段式结构成立，但目标要放宽**：预注册的复合 recall 0.90 在漂移下
   过严；放宽到 0.95-0.99 后 17/136 单元过三线（含 30 s mean_abs_drr_norm
   0.9712/0.9164、052 0 事件）。这是 R16 的正信号，但属事后，需 R17 新预注册。
4. **形态学路线关闭**：全网络微调+DANN 把 afdb 从 0.74 提到 0.86-0.89，
   仍 <0.90；按纪律不再重试。Lane A 只保留 RR 通道。
5. **检测器质量是 ltafdb 判据的主要噪声源**：PT100 在 ltafdb 的 within-1
   只有 65%、p90 误差 25 拍；标注 QRS 复评把 ltafdb 召回从 0.90-0.92 提到
   0.93-0.94。后续应把“标注 QRS”作为 ltafdb 的敏感性口径并披露 PT100 局限。
6. **设备域边界**：2 条正常录音、无 AF 正样本；052 静默必要不充分。
   灵敏度主张仍来自外部域；A7 设备正样本采集是唯一解。

## §8 部署线

- `runs/deploy_candidate_r11/`、ESP-IDF 主线、Lane B（M4 门控）均未改动；
  烧录仍暂停等物理解锁（解锁后 `idf.py -p COM19 flash`；板上第一动作：
  设备 FP=0 → MI 26/26 → TFLM vs ESP-NN 计时）。
- Lane A 状态：RR 通道研究级正信号（§A2-PH），**未定案**；AF 头形态学路线
  关闭；设备域验收等 A7。

## §9 下一步（待用户决策）

1. **R17 新预注册**：固定“Stage1 rr_entropy@0.99 + Stage2 30 s
   mean_abs_drr_norm@0.99”单一单元（或 cv），在新设备采集上盲验；
   预注册判据同三线。
2. **A7 设备正样本**：≥5 条设备正常（含窦性心律不齐/房早）+ ≥5 条设备 AF
   录音；数据到位后用 A2-PH 胜出门做 PC 回放验收（设备 AF 召回≥0.90 且
   设备正常 FP=0）。
3. **ltafdb**：ECG1 + 标注 QRS 作为主口径；补足 N 段采样。
4. **阈值**：设备域校准集 + 域漂移校正（如按记录/HR 分层阈值）；
   spec@R90 主口径不变。
5. **AF 头**：按 A4 FAIL 关闭形态学路线；若用户要求重启，需新数据/新机制
   （不是再次微调）。

---
产物索引：`runs/R16_PREREG.md`（预注册）；`runs/r16_rr_shape/`（A1 +
detector_qc + 特征/RR 缓存）；`runs/r16_rr_shape_gate/`（A2 +
posthoc_two_stage_sensitivity.json + annot_qrs_sensitivity.json）；
`runs/r16_rr_seq/`（A3 cfg12/cfg30）；`runs/r16_af_head_adapt/`（A4
finetune/dann_lam0.3/dann_lam0.1/aug）；`runs/r16_theta_crossfit/`（A5）；
`runs/r16_ltafdb_sensitivity/`（A6）。
"""
    (R / 'SESSION_REPORT_R16.md').write_text(txt, encoding='utf-8')
    print('written', R / 'SESSION_REPORT_R16.md', len(txt))


if __name__ == '__main__':
    main()
