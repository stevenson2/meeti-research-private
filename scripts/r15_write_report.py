#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate docs/reports/SESSION_REPORT_R15.md and append the R15 revision to
runs/R14_TWO_LANE_GATE_SPEC.md from the round result JSONs."""
import json
from pathlib import Path

R = Path('/home/devcontainers/meeti/docs/reports')
b1 = json.load(open(R / 'r15_laneA_rr_gate/result.json'))
b1d = json.load(open(R / 'r15_laneA_rr_gate/diagnosis.json'))
b1s = json.load(open(R / 'r15_laneA_rr_gate/rr_shape.json'))
b2 = json.load(open(R / 'r15_af_calib20_summary.json'))
b3 = json.load(open(R / 'r15_af_adapt/result.json'))
b4 = json.load(open(R / 'r15_ltafdb_prep/result.json'))
b4e = json.load(open(R / 'r15_ltafdb_rr_eval/result.json'))
b5 = json.load(open(R / 'r15_af_pool/result.json'))

L = []
def p(s=''):
    L.append(s)

p('# R15：Lane A 重建（RR 优先 + AF 头修复）执行报告')
p()
p('日期：2026-09-09。主题：按 R15 提示词重建 Lane A（RR 主通道 + AF 头域适配），')
p('并纳入用户对 R14 复核后的四点修正（rec_latest 原始流、口径精确化、根因不过度归纳、')
p('“RR 唯一”改为“已测通道中唯一”）。烧录仍暂停；Lane B 与 deploy_candidate_r11 未动。')
p()
p('## §0 执行摘要')
p()
p('- **B1（决定性）：FAIL。** RR-only Lane A 在 ecg_real_052 上无法做到事件级 FP=0：')
p('  θ_R90=2.1418 时 19/174 窗报警、4 个 M4 事件；θ_youden=1.9808 时 5 个 M4 事件。')
p('  afdb 留出域召回 @θ_R90=0.8546（<0.90），@θ_youden=0.9076。**根因已定位**：')
p('  ecg_real_052 是呼吸性窦性心律不齐（RR 熵中位 1.82/最大 2.45、pNN50 0.47、')
p('  滞后 1 自相关 0.437、RMSSD/SDNN 1.05），与 AF 的“突变型”RR（自相关 ≈ −0.06、')
p('  RMSSD/SDNN 1.40）本质不同；不是 R 波检测错误（XQRS 与 PT100 一致）。')
p('  **全部 6 特征 + LR 组合均无法同时满足设备 FP=0 与 afdb 召回 ≥0.90**（诊断表）。')
p('- **B2：阈值稳定化判据 FAIL（极差 0.546 > 0.30），但工作点显著改善**：')
p('  20% 留出（3,927 条/~269 AF）后 fold-10 R@θ_tr **3/3 ≥0.90**（0.908/0.901/0.915，')
p('  R14 为 0.875/0.921/0.895，2/3 未过）；spec@R90 **0.961±0.003**（R14 0.945±0.013）；')
p('  fold-10 AUC 0.9698±0.0015（未退化）。spec@R90 定为 AF 头主口径。')
p('- **B3：AF 头域适配 FAIL**：冻结编码器 + 读出头在 PTB folds1-8 + afdb 5k+5k +')
p('  CinC2017 A/N/O 上训练，afdb 留出域 AUC 仅 **0.7375/0.7404/0.7401**（go-line ≥0.90）；')
p('  最佳 epoch 均为 1（即 R14 原头），说明扩训只过拟合、未带来跨域泛化；')
p('  fold-10 维持 0.966，CinC2017 留出 0.834，设备域极性敏感依旧。')
p('- **B4：ltafdb 第三外部域建成**：84 记录、**371,193 AF / 298,319 N 窗**，')
p('  对账 3/3 OK；RR 熵在该域 AUC **0.8737**、AP 0.8684（θ_youden 召回 0.928）。')
p('- **B5：扩训池登记完成**：CinC2017 2,214（A 738 + N/O 各 738）、afpdb 4,500')
p('  （n/p 30-min 难负例 2,000 + 续段/测试未标 2,500）、aftdb 400（全部 AF 段）。')
p('- **本轮最强证据（同一 afdb 留出 11 记录）**：零训练 RR 熵 AUC **0.9753**')
p('  vs 域适配 AF 头 **0.74**；配合 R14 的极性翻转 FP 6↔171，结论：')
p('  在已测通道中，稳健的 AF 信号是**节律**，不是形态学。')
p()
p('## §B1 RR-only Lane A 设备域门禁（零训练）')
p()
p('### B1.1 口径与修订（用户 2026-09-09 修正）')
p()
p('- 预注册验收修订为 **ecg_real_052 单录音**：事件级 FP=0（M4 协议 1-of-5、冷却 20）')
p('  + afdb 留出域 AF 召回 ≥0.90；rec_latest 从预注册验收中剔除并单独披露。')
p('- 更正 R14 说法：R 波索引**未持久化**（real_preprocess_exp7c.json 只有 n_peaks=211）；')
p('  本轮从原始 .ecgr 经固件部署链（comb+HP/LP+因果 HP0.5@250）重算 XQRS 峰位，')
p('  与固件/PC 驱动同检测器语义。')
p('- 关于 rec_latest 的原始流：仓库根目录存在 `rec_latest.ecgr`（31,400 B，64 s，')
p('  fs_eff=245.0625 Hz，`retest_ai_rec_latest.py` 即读取该文件），本轮 B1 已用它')
p('  重算真实 RR（102 个 XQRS 峰、中位 RR 0.608 s），**不是**恒定平均 RR 拼接。')
p('  其单独结果：θ_R90 下 0 个 M4 事件、θ_youden 下 1 个；若用户希望换一次采集，')
p('  可重新导出后补测。')
p()
p('### B1.2 结果（阈值来自 PTB folds1-9 + afdb fit 记录的联合训练侧分布）')
p()
p(f"- 拟合：{b1['fit']['n_fit']:,} 窗（正 {b1['fit']['pos_fit']:,}），"
  f"θ_youden={b1['thresholds']['theta_youden']}、θ_R90={b1['thresholds']['theta_R90']}；")
p(f"  拟合侧 R/spec：Youden {b1['fit']['fit_R_youden']}/{b1['fit']['fit_spec_youden']}、")
p(f"  R90 {b1['fit']['fit_R_R90']}/{b1['fit']['fit_spec_R90']}。")
p(f"- afdb 留出 11 记录（{b1['afdb_split']['n_eval_windows']:,} 窗，AF "
  f"{b1['afdb_split']['n_eval_af']:,}）：θ_R90 召回 {b1['afdb_eval_primary']['recall']} / "
  f"spec {b1['afdb_eval_primary']['spec']}；θ_youden 召回 "
  f"{b1['afdb_eval_youden']['recall']} / spec {b1['afdb_eval_youden']['spec']}。")
p(f"- ecg_real_052（部署链 XQRS）：θ_R90 **{b1['device']['ecg_real_052']['deploy_xqrs250']['primary']['n_flag']}/174 窗、"
  f"{b1['device']['ecg_real_052']['deploy_xqrs250']['primary']['m4_alarms']} 个 M4 事件**；"
  f"θ_youden {b1['device']['ecg_real_052']['deploy_xqrs250']['youden']['m4_alarms']} 个事件。")
p(f"- rec_latest（补充，非预注册）：θ_R90 {b1['device']['rec_latest']['deploy_xqrs250']['primary']['m4_alarms']} 个事件、"
  f"θ_youden {b1['device']['rec_latest']['deploy_xqrs250']['youden']['m4_alarms']} 个事件。")
p(f"- **判定：FAIL**（052 的 FP=0 与 afdb 召回 ≥0.90 两项均未过；见 acceptance 块）。")
p()
p('### B1.3 根因诊断（不调参，只定位）')
p()
p('| 通道 | afdb 留出 AUC | 052 事件@θ_R90 | rec_latest 事件@θ_R90 | 结论 |')
p('|---|---|---|---|---|')
for feat in ['rmssd', 'sdnn', 'rr_entropy', 'drr_entropy', 'sampen_m1',
             'cosen', 'lr_combo']:
    row = b1d['feature_device_afdb_table'][feat]
    a = row['afdb_eval'].get('R90', {}).get('recall')
    d052 = row['device'].get('R90', {}).get('ecg_real_052', {}).get(
        'deploy_xqrs250', {}).get('m4_alarms')
    drec = row['device'].get('R90', {}).get('rec_latest', {}).get(
        'deploy_xqrs250', {}).get('m4_alarms')
    p(f"| {feat} | {a} | {d052} | {drec} | "
      f"{'双过?' if (a and a >= 0.90 and d052 == 0 and drec == 0) else '至少一侧未过'} |")
p()
p(f"- RR 形态对比（中位 [p10,p90]）：ecg_real_052 全记录 RR 熵 "
  f"{b1s['device_whole_record']['ecg_real_052']['rr_entropy']}、"
  f"滞后 1 自相关 {b1s['device_whole_record']['ecg_real_052']['autocorr_lag1']}、"
  f"RMSSD/SDNN {b1s['device_whole_record']['ecg_real_052']['rmssd_over_sdnn']}、"
  f"pNN50 {b1s['device_whole_record']['ecg_real_052']['pnn50']}；")
p(f"  afdb AF 参照 RMSSD/SDNN "
  f"{b1s['afdb_eval']['AF']['rmssd_over_sdnn']['median']}、自相关 "
  f"{b1s['afdb_eval']['AF']['autocorr_lag1']['median']}；PTB 非 AF 自相关 "
  f"{b1s['ptb_train']['nonAF']['autocorr_lag1']['median']}。")
p('- 结论：052 的 RR 不规则是**平滑、自相关的呼吸性窦性心律不齐**，RR 熵与 AF 重叠；')
p('  RR 熵单通道无法区分。下一研究方向：形状感知特征（滞后 1 自相关、RMSSD/SDNN 比值）')
p('  作为新研究臂重新预注册，**不在 B1 上重调**。')
p()
p('## §B2 AF 头阈值重锚（20% 患者级留出，3 seeds）')
p()
p('| seed | fold-10 AUC | AP | spec@R90 | θ_tr | R@θ_tr | spec@θ_tr | best epoch |')
p('|---|---|---|---|---|---|---|---|')
for r in b2['rows']:
    p(f"| {r['seed']} | {r['auc']} | {r['ap']} | {r['spec_at_R90']} | "
      f"{r['theta_tr']} | {r['R_at_theta_tr']} | {r['spec_at_theta_tr']} | "
      f"{r['best_epoch']} |")
p()
p(f"- θ_tr 极差 **{b2['theta_tr_range']}**（判据 ≤0.30 → "
  f"**{'PASS' if b2['theta_tr_range_criterion_le_0.30'] else 'FAIL'}**；R14 5% 留出极差 "
  f"{b2['theta_tr_R14_5pct_range']}）。")
p(f"- fold-10 AUC {b2['auc_mean_sd'][0]}±{b2['auc_mean_sd'][1]}（R14 0.9674±0.0021）；")
p(f"  spec@R90 {b2['specR90_mean_sd'][0]}±{b2['specR90_mean_sd'][1]}（R14 0.9452±0.0125）；")
p(f"  R@θ_tr 3/3 ≥0.90（R14 2/3 未过）→ **工作点改善**。")
p('- 结论：阈值数值本身仍不稳（0.038–0.584），但 20% 留出把工作点拉回可用区；')
p('  spec@R90 定为 AF 头主口径；部署前仍需更大/设备域校准集。')
p()
p('## §B3 AF 头域适配（冻结编码器 + 扩训）')
p()
p(f"- 配方：冻结 R14 主干、缓存 128-d GAP 特征、读出头由 fc1/out 初始化；训练 "
  f"{b3['n_train']:,} 窗（PTB folds1-8 {b3['train_domains']['ptb']:,} / afdb "
  f"{b3['train_domains']['afdb']:,} / CinC2017 {b3['train_domains']['cinc2017']:,}）；")
p(f"  验证：PTB fold9 {b3['val']['ptb_fold9']:,} + afdb 留出 3 记录 2,000 窗 + CinC 留出 442。")
p()
p('| seed | 最佳 epoch | fold-10 AUC | fold-10 spec@R90 | afdb 留出 AUC | afdb AP | CinC 留出 AUC | go |')
p('|---|---|---|---|---|---|---|---|')
for s in ['42', '43', '44']:
    x = b3['seeds'][s]
    p(f"| {s} | {x['best']['epoch']} | {x['ptb_fold10']['auc']} | "
      f"{x['ptb_fold10']['spec_at_R90']['spec']} | {x['afdb_eval']['auc']} | "
      f"{x['afdb_eval']['ap']} | {x['cinc_val']['auc']} | {x['go_pass']} |")
p()
p('- **判定：FAIL**（go-line afdb AUC ≥0.90 未达；最佳 epoch=1 即 R14 原头，')
p('  扩训只过拟合训练域，afdb 留出反而略低于 R14 的 0.76–0.90）。')
p('- 设备域披露：052_chain 正极性 max 0.819 / 负极性 max 0.948（均值 0.224 vs 0.820）；')
p('  rec_latest max 0.91 —— 极性/幅度敏感性在适配后依然存在。')
p('- 结论：仅换读出头无法修复形态学跨域；需要全网络微调、域对抗或更强的域增广，')
p('  或接受“形态学分数跨域不可靠”的边界。')
p()
p('## §B4 ltafdb 第三外部域（prep + RR 评测）')
p()
p(f"- prep：{b4['n_records']} 记录、**AF {b4['n_windows_afib']:,} / N {b4['n_windows_n']:,} 窗**；")
p(f"  节律标签清单 {json.dumps(b4['rhythm_label_inventory'], ensure_ascii=False)}；")
p(f"  VT/AFL/AB/SVTA/SBR 等剔除；对账 {b4['reconcile']['records']} → "
  f"n_mismatch={b4['reconcile']['n_mismatch']}。")
p(f"- RR 熵评测：{b4e['n_windows']:,} 窗（AF {b4e['n_pos']:,}），AUC **{b4e['auc']}**、")
p(f"  AP {b4e['ap']}；θ_R90 召回 {b4e['op_points']['theta_R90']['recall']}/spec "
  f"{b4e['op_points']['theta_R90']['spec']}；θ_youden 召回 "
  f"{b4e['op_points']['theta_youden']['recall']}/spec {b4e['op_points']['theta_youden']['spec']}。")
p('- 披露：ltafdb AF 富集、N 段稀少且可能含其他不规则节律（spec 偏低）；通道用 ECG2；')
p('  该域主要作 RR 通道的召回/外部坐标，不作为部署判据。')
p()
p('## §B5 扩训池登记（修正语义后）')
p()
p('| 来源 | 窗数 | 正/负/未标 | 语义（已对 PhysioNet 文档核验） |')
p('|---|---|---|---|')
p(f"| CinC2017 | {b5['cinc2017']['n_windows']:,} | {b5['cinc2017']['n_pos']} / "
  f"{b5['cinc2017']['n_neg']} / 0 | A=AF 正；N/O 采样为负；~ 剔除 |")
p(f"| afpdb | {b5['afpdb']['n_windows']:,} | {b5['afpdb']['n_pos']} / "
  f"{b5['afpdb']['n_neg']} / {b5['afpdb']['n_unlabelled']} | n*/p* 30-min=难负例；"
  f"nc*/pc* 续段混合、t* 未标 → y=−1 |")
p(f"| aftdb | {b5['aftdb']['n_windows']:,} | {b5['aftdb']['n_pos']} / 0 / 0 | "
  f"全部为 AF 段（n/s/t=终止类别，a/b=挑战测试集）→ y=1 |")
p()
p('- 关键更正：afpdb 的 p* 是 **PAF 前驱（记录内无 AF）**，不能当 AF 正类；')
p('  t* 是无标签测试集。aftdb 的 n/s/t/a/b 只是终止类别，全部是 AF 段。')
p('- 产物：`runs/r15_af_pool/{cinc2017,afpdb,aftdb}.npz` + result.json（R13 家族，')
p('  1000 点@100 Hz raw mV，未做幅度对齐，供 B3/未来扩训复用）。')
p()
p('## §6 本轮关键对比与教训')
p()
p('1. **同一 afdb 留出 11 记录：RR 熵 AUC 0.9753 vs 域适配 AF 头 0.74**；')
p('   跨域时“节律”通道远强于“形态学”通道。')
p('2. **极性/幅度敏感性是形态学不可部署的直接证据**：R14 中同模型同录音仅极性翻转，')
p('   052 窗级 FP 从 6/174 摆到 171/174；R15 适配后 052 均值概率 0.224 vs 0.820。')
p('3. **阈值校准：** 20% 留出把 R@θ_tr 拉到 3/3 ≥0.90，但 θ_tr 数值极差仍 0.546；')
p('   口径应报 spec@R90，点阈值只作披露。')
p('4. **仅读出头域适配不够**：冻结主干 + 扩训在 afdb 留出无改善（0.74），')
p('   最佳 epoch=1 说明训练数据没有提供可泛化的形态特征。')
p('5. **RR 的边界**：窦性心律不齐/房早二联律同样抬高 RR 熵；形状感知特征')
p('   （自相关、RMSSD/SDNN）是下一步，不是 B1 重调。')
p('6. **按用户修正收敛表述**：R12（教师噪声迁移）与 R14（幅度/域失配+校准不稳）')
p('   根因不同，共同点只是“形态学分数跨域不可靠”；“RR 稳健”限定为“已测通道中”；')
p('   Apple/Kardia 的收敛证据是 IRNF 节律 + PhysioNet2017 + Tateno/Lake 文献线，')
p('   KardiaBand 本身是 ECG 深度学习。')
p()
p('## §7 部署线')
p()
p('- `runs/deploy_candidate_r11/`、ESP-IDF 主线、Lane B（M4 门控）均未改动；')
p('  烧录仍暂停等物理解锁（解锁后 `idf.py -p COM19 flash`；板上第一动作：')
p('  设备 FP=0 → MI 26/26 → TFLM vs ESP-NN 计时）。')
p()
p('## §8 下一步（待用户决策）')
p()
p('1. **形状感知 RR 通道**：在 rr_entropy 上加入滞后 1 自相关 / RMSSD-SDNN 比值，')
p('   重新预注册 B1 式设备门禁（目标：窦性心律不齐不报警、AF 召回 ≥0.90）。')
p('2. **AF 头**：若要继续，需全网络微调或域对抗 + 更真实的设备域数据；')
p('   仅读出头已被本轮否证。')
p('3. **ltafdb**：ECG1 通道敏感性 + 更大 N 采样；可作 RR 通道第二外部召回域。')
p('4. **阈值**：交叉拟合/设备域校准集（含设备 AF 正样本）；spec@R90 主口径。')
p('5. B5 池已就绪，可支持上述任一训练臂。')
p()
p('---')
p('产物索引：`runs/r15_laneA_rr_gate/`（B1 + diagnosis + rr_shape）、')
p('`runs/r15_af_calib20_summary.json` + `runs/r14_af_head/af_seed*_calib20/`（B2）、')
p('`runs/r15_af_adapt/`（B3）、`runs/r15_ltafdb_prep/` + `runs/r15_ltafdb_rr_eval/`（B4）、')
p('`runs/r15_af_pool/`（B5）。')
p()
(R / 'SESSION_REPORT_R15.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
print('report written', len(L), 'lines')

# ---- spec revision ----
spec = (R / 'R14_TWO_LANE_GATE_SPEC.md').read_text(encoding='utf-8')
rev = []
rev.append('')
rev.append('---')
rev.append('')
rev.append('## R15 修订（2026-09-09）：RR-only Lane A 门禁结果与主通道判定')
rev.append('')
rev.append('### R15.1 B1 RR-only 设备门禁（预注册修订：052 单录音）')
rev.append('')
rev.append('- **FAIL**：ecg_real_052 在 θ_R90=2.1418 下 19/174 窗报警、4 个 M4 事件；')
rev.append('  θ_youden=1.9808 下 5 个事件。afdb 留出召回 @θ_R90=0.8546（<0.90）、')
rev.append('  @θ_youden=0.9076。全部 6 特征 + LR 组合均无法同时过两条线。')
rev.append('- **根因**：ecg_real_052 为呼吸性窦性心律不齐（RR 熵中位 1.82/最大 2.45、')
rev.append('  pNN50 0.47、滞后 1 自相关 0.437、RMSSD/SDNN 1.05），与 AF 的突变型 RR')
rev.append('  （自相关 ≈ −0.06、RMSSD/SDNN 1.40）本质不同；R 波检测已交叉验证。')
rev.append('- rec_latest：预注册验收中剔除；补充结果（原始 `rec_latest.ecgr` 重算真实 RR）')
rev.append('  θ_R90 0 事件、θ_youden 1 事件。')
rev.append('')
rev.append('### R15.2 Lane A 主通道判定')
rev.append('')
rev.append('- **RR-only 主通道：BLOCKED**（B1 FAIL，原因=窦性心律不齐混淆，非检测器缺陷）。')
rev.append('- **AF 形态学头：未升级**（B3 域适配 FAIL：afdb 留出 AUC 0.74 vs go-line 0.90；')
rev.append('  最佳 epoch=1；设备域极性敏感依旧）。')
rev.append('- 同一 afdb 留出 11 记录：RR 熵 AUC 0.9753 vs 适配 AF 头 0.74 → 跨域稳健通道')
rev.append('  仍是节律；Lane A 维持“不部署”，Lane B 不变。')
rev.append('')
rev.append('### R15.3 下一步（新研究臂，需重新预注册）')
rev.append('')
rev.append('- 形状感知 RR 特征（滞后 1 自相关 / RMSSD-SDNN 比值）区分窦性心律不齐与 AF；')
rev.append('- AF 头需全网络微调/域对抗 + 设备域真实数据，而非仅读出头；')
rev.append('- 阈值口径：spec@R90 主口径（B2：20% 留出后 0.961±0.003；R@θ_tr 3/3 ≥0.90，')
rev.append('  但 θ_tr 数值极差 0.546 仍披露）。')
rev.append('- ltafdb 第三域（RR 熵 AUC 0.8737）作为外部召回坐标，不作部署判据。')
rev.append('')
(R / 'R14_TWO_LANE_GATE_SPEC.md').write_text(
    spec + '\n'.join(rev) + '\n', encoding='utf-8')
print('spec revision appended')
