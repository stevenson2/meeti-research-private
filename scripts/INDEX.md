# scripts/INDEX.md — 脚本编目（R17-CLEANUP P3 建立）

> 状态含义：**active** = 当前主线可复现链路；**round** = 某研究轮的实验脚本
>（保留原地，靠前缀定位）；**archived** = 一次性脚本，移入 `scripts/archive/`。
> 被 import 的模块不移动：`ecg_bench.py`、`r14_rr_baseline.py`、`r16_rr_features.py`、
> `r16_rr_data.py`、`models_patch/`。路径规则见 `check_paths.py`。

## 跨 shell 工具（AGENTS §1）

| 脚本 | 用途 | 状态 |
|---|---|---|
| `project_paths.json` | 跨 shell 路径唯一真值（canonical copy） | active |
| `project_paths.py` | Python 读取器：P / paths_for / shell_for | active |
| `project_paths.sh` | bash source 读取器（export 环境变量） | active |
| `project_paths.ps1` | PowerShell dot-source 读取器（env 变量） | active |
| `cross_shell.py` | Cross-shell task dispatcher (WSL bash <-> Windows PowerShell). | active |
| `check_paths.py` | Hardcoded-path audit for the meeti + electron repos (R17-CLEANUP P5). | active |
| `check_paths.json` | check_paths.py 的逐文件白名单（记录既有债务） | active |
| `task.sh` | 任务封装：train/build/flash/serial/check（WSL 侧入口） | active |
| `task.ps1` | 任务封装：build/flash/serial/train/check（Windows 侧入口） | active |

## 数据准备（MIMIC/MEETI/PTB/AF 库）

| 脚本 | 用途 | 状态 |
|---|---|---|
| `process_mimic_raw_beat_centered.py` | M3: process MIMIC raw .hea/.dat -> beat-centered 12-lead windows. | active |
| `r8_data_5s.py` | R8 data construction: single-lead 5 s windows for the record student. | active |
| `r9_data.py` | R9 data construction: normalization / form-alignment fixes for the 5s | active |
| `r10_ptbxl_fold10_prep.py` | R10: build PTB-XL fold-10 (strat_fold==10) validation data. | active |
| `r13_ptbxl_prep.py` | R13 phase-0 prep: PTB-XL single-lead lead-II 100 Hz dataset. | active |
| `r14_af_prep.py` | R14-A1: PTB-XL AF label prep on the R13 single-lead lead-II family. | active |
| `r14_afdb_prep.py` | R14-A7: MIT-BIH Atrial Fibrillation Database (afdb) -> Lane-A eval domain. | active |
| `r15_ltafdb_prep.py` | R15-B4: ltafdb window-level prep (third external AF domain). | active |
| `r16_rr_shape_prep.py` | R16-A1: shape-aware RR feature feasibility (zero training, pre-registered). | active |
| `r16_rr_data.py` | R16 data layer: RR sequences + extended RR features for all domains. | active |
| `scan_local_meeti_metadata.py` | scan_local_meeti_metadata.py — 扫描本地 MEETI 目录中已下载的 .mat 元数据。 | active |

## 训练主线（locked dual-head 谱系）

| 脚本 | 用途 | 状态 |
|---|---|---|
| `r4_train_label_fix.py` | Round 4 (autonomous loop): teacher-corrected weak labels for Head_R. | active/round |
| `train_dual_head_small.py` | M1: 共享骨干双头 MIL 学生 (Head_B 拍级 + Head_R 记录级) 小规模训练。 | active/round |
| `train_dual_head_v2_small.py` | M1 消融版: 双头共享骨干, 每个 epoch 拍级/记录级数据集各恰好扫一遍。 | active/round |
| `train_dual_head_v3_small.py` | M1 消融 v3: 单优化器 + 双头同时前向, 拍级/记录级数据集每个 epoch 各扫一遍。 | active/round |
| `train_dual_head_v3_fc1w_small.py` | Route-2 capacity ablation: dual-head v3 with wider fc1 (64 -> 128/256). | active/round |
| `train_dual_head_v3_init_small.py` | STEP 1: v3 dual-head training with clean-baseline initialization. | active/round |
| `train_dual_head_record_pls_kd_small.py` | C: record-level PLS distillation (MIMIC local teacher -> Head_R pooled emb). | active/round |
| `r9_train.py` | R9: normalization / form-alignment arms for the 5s single-lead student. | active/round |
| `train_meeti_weak_small.py` | 小规模实验: 公共库硬标签 + MEETI 记录级弱标签辅助训练(输出项目外)。 | active/round |
| `train_public_small.py` | 小规模公共库基线训练(输出全部写到项目外 /home/devcontainers/meeti/runs)。 | active/round |

## 蒸馏 / teacher 特征

| 脚本 | 用途 | 状态 |
|---|---|---|
| `extract_ecg_fm_local_features.py` | Extract ECG-FM center-time-step local features from beat-centered windows. | round |
| `extract_ecg_fm_teacher.py` | 用 ECG-FM pretrained encoder 对已保存的 MEETI 5s 段抽 768 维全局嵌入(CPU)。 | round |
| `m3_local_feature_probe.py` | M3 pre-registered probe gate. | round |
| `teacher_supervised_subspace.py` | A: supervised subspace (PLS) of ECG-FM local features. | round |
| `train_ecgfm_feature_kd_small.py` | 小规模 ECG-FM 特征蒸馏实验: MEETI beat -> student fc1 投影 -> teacher 768 嵌入(MSE), | round |
| `train_ecgfm_feature_probe_small.py` | 在 ECG-FM 特征蒸馏后的 fc1 特征上只训练分类头(冻结骨干), 输出项目外。 | round |
| `train_dual_head_kd_record_small.py` | M2: 在 M1 双头模型上, 仅给 Head_R 增加 ECG-FM 全局嵌入蒸馏 (MSE)。 | round |
| `train_dual_head_kd_record_v3_small.py` | 受控 KD A/B 的 KD 分支: 与 train_dual_head_v3_small.py 同配方 + Head_R 教师投影。 | round |
| `train_dual_head_local_kd_small.py` | M3 local feature distillation (only after probe gate passes). | round |
| `analyze_local_kd_failure.py` | Post-hoc diagnosis: why per-beat ECG-FM local feature distillation failed. | round |

## 评测 / 门控 / 事件

| 脚本 | 用途 | 状态 |
|---|---|---|
| `ecg_bench.py` | BENCHMARK_V1 unified evaluator (runs/BENCHMARK_V1.md).  Eval-only. | active/round |
| `m4_afe_event_eval.py` | M4: real AFE long-recording event-level FP/hour and MIT/INCART val recall. | active/round |
| `m4_diagnose_head_r.py` | M4 diagnosis: why does locked Head_R saturate (0.75-0.95) on real AFE normal beats? | active/round |
| `m4_event_specificity_sweep.py` | Direction 3: in-domain event-level specificity sweep. | active/round |
| `m4_postproc_eval.py` | M4 post-processing candidates for Head_R real-AFE false alarms. | active/round |
| `m4_ptbval_gate_check.py` | Report-only: PTB val event-level check of the V1 AND-gate (taub=0.2, taur=0.5). | active/round |
| `m4_threshold_sweep.py` | M4 real-AFE threshold sweep for the locked dual-head record head. | active/round |
| `r5_fullgate_eval.py` | Round 5 (autonomous loop): full-gate evaluation of the r4 teacher-label | active/round |
| `r8_eval_ptb_holdout.py` | R8: PTB record-level holdout eval for the 5 s single-lead student. | active/round |
| `r8a_caliber_bridge.py` | R8a caliber-bridge measurement (report-only, zero training). | active/round |
| `eval_record_head_ptb_holdout.py` | 用 PTB 记录级金标准验证已训练的 Head_R (record_probe.keras)。 | active/round |
| `eval_dual_head_probe_pair.py` | 评估一对 beat_probe / record_probe 模型。 | active/round |
| `eval_repo_manifest.py` | 按 curated manifest 评估仓库 pc_tools/ecg_dl 的前代模型 (同一测试标准)。 | active/round |
| `eval_tflite_selected.py` | TFLite 前代导出模型按同一测试评估 (每个谱系选最后一个导出)。 | active/round |
| `r14_meeti_af_scan.py` | R14-A1 optional: MEETI report-field AF scan (weak-label auxiliary domain). | active/round |
| `r14_afdb_af_eval.py` | R14 optional: external AF validation of the AF head on MIT-BIH afdb. | active/round |
| `r15_ltafdb_rr_eval.py` | R15-B4 eval: RR-entropy channel on ltafdb (third external domain). | active/round |
| `r15_laneA_rr_gate.py` | R15-B1: RR-only Lane A device-domain gate (zero training). | active/round |
| `r16_rr_shape_gate.py` | R16-A2: pre-registered two-stage shape gate (decisive). | active/round |
| `r16_detector_qc.py` | R16 detector QC: PT100 (A4 family) vs database QRS annotations. | active/round |
| `r16_annot_qrs_sensitivity.py` | R16 annotated-QRS sensitivity for the RR gate (detector-quality control). | active/round |
| `r16_ltafdb_sensitivity.py` | R16-A6: ltafdb channel / window-length sensitivity (low priority). | active/round |
| `r16_theta_crossfit.py` | R16-A5: threshold cross-fitting + device-domain calibration disclosure. | active/round |
| `r16_posthoc_gate.py` | R16 post-hoc gate sensitivity (clearly labelled NOT pre-registered). | active/round |

## 部署 / TFLite / QAT

| 脚本 | 用途 | 状态 |
|---|---|---|
| `m4_export_tflite.py` | Direction 1: export locked dual-head probes to TFLite + parity check. | active/round |
| `m4_tflite_e2e_gate.py` | Direction 1: end-to-end gated event replay with exported TFLite models. | active/round |
| `r1_qat_dual.py` | Round 1 (autonomous loop): int8 for the locked dual probes. | active/round |
| `r1b_qat_legacy.py` | Round 1b: true QAT int8 via TF_USE_LEGACY_KERAS=1 (tfmq needs Keras 2). | active/round |
| `r6_qat_gain_finetune.py` | Round 1b: true QAT int8 via TF_USE_LEGACY_KERAS=1 (tfmq needs Keras 2). | active/round |
| `r6b_ptq_gaincal_test.py` | R6b: decisive test - PTQ with GAIN-AUGMENTED calibration (input scale in | active/round |
| `r11_deploy_select.py` | R11 deployment candidate selection + complete model save. | active/round |
| `r11_pc_parity.py` | R11 PC-side parity + gate validation of the EXACT deployed artifacts. | active/round |
| `r16_device_replay_ref.py` | R16 device replay reference: PC-side replication of the ESP-IDF AI chain. | active/round |

## 基准 / 文献对照

| 脚本 | 用途 | 状态 |
|---|---|---|
| `bench_teacher_extract.py` | BENCHMARK_V1: ECG-FM teacher embeddings for the benchmark domains. | active/round |
| `plot_s_ranking.py` | Composite-score S ranking chart (Artificial-Analysis style). | active/round |
| `r10_lit_fm_linear_probe.py` | R10 lit row: ECG-FM linear probe on frozen embeddings (paper protocol). | active/round |
| `r10_lit_strodthoff.py` | R10 lit rows: official Strodthoff-lab xai4ecg checkpoints (resnet50-1D / | active/round |
| `r10_merge_lit_rows.py` | R10: merge lit_row_*.json into the existing leaderboard without | active/round |
| `r10_strodthoff_ext.py` | R10 extension (user challenge Q1/Q2): | active/round |
| `r10e_verify.py` | R10e verification round: data / method / ranking checks. | active/round |
| `r11_bench_bundle.py` | R11 on-board bench: build the replay bundle (deploy-chain beat windows). | active/round |
| `render_summary.py` | Render final benchmark v1.5 summary charts + tables as PNG images (v2). | active/round |
| `summarize_init_vs_scratch.py` | Aggregate eval_init_vs_scratch JSONs and apply the STEP 2.4 gates. | active/round |
| `r9_6seed_summary.py` | 6-seed aggregate for the r9_form-ptbbal candidate (42-47). | active/round |
| `r9_agg_sensitivity.py` | R9 eval-side probe: record-score aggregation sensitivity (zero training). | active/round |
| `r9_probe_scale_stats.py` | P1 probe: quantify per-window scale/offset stats across all domains. | active/round |
| `r9_resummarize.py` | Regenerate r9 arm summaries from per-seed result.json files. | active/round |

## R12-R16 研究轮

| 脚本 | 用途 | 状态 |
|---|---|---|
| `r12_t2_kd_beat.py` | R12-T2: ensemble -> single beat-head distillation (soft-label KD). | round |
| `r12_t3_dann.py` | R12-T3: DANN arm for the record head (gradient-level domain-invariance). | round |
| `r12_t4_temp.py` | R12-T4: train-side temperature scaling of the candidate record head. | round |
| `r12_t5_specr90.py` | R12: spec@R90 post-hoc eval for the T5 hardneg arm seeds 42-44 (PTB val | round |
| `r12_t6_beat_readout.py` | R12-T6 feasibility probe: per-beat readout from the frozen 5s encoder. | round |
| `r13_ptbxl_5cls_smoke.py` | R13 phase-0 smoke: PTB-XL single-lead 5-class vs binary NORM-vs-rest. | round |
| `r14_af_head.py` | R14-A2/A3/A5: AF-specialized record head on PTB-XL single-lead lead II. | round |
| `r14_af_opmetrics.py` | R14 post-hoc operating-point layer for the AF head (zero training). | round |
| `r14_af_seeds.py` | R14-A5: paired 3-seed verdict for the AF head vs the general binary head. | round |
| `r14_more_afdb_download.sh` | R14: 追加下载 afdb 补充记录（bash runner） | round |
| `r14_rr_baseline.py` | R14-A4: zero-training RR-irregularity baseline on fold-10 lead II. | round |
| `r14_two_lane_sim.py` | R14-A6: two-lane gate PC simulation (zero training). | round |
| `r15_af_adapt.py` | R15-B3: AF-head domain adaptation (frozen encoder + readout head). | round |
| `r15_af_pool.py` | R15-B5: external AF training/eval pool -> R13 family (1000 pts @100 Hz mV). | round |
| `r15_b1_diagnose.py` | R15-B1 diagnosis: why does the RR-only device gate fail on ecg_real_052? | round |
| `r15_b1_rr_shape.py` | R15-B1 evidence: RR-shape comparison (sinus arrhythmia vs AF vs ectopy). | round |
| `r16_af_head_adapt.py` | R16-A4: AF-head repair - full-network fine-tune / DANN / domain aug. | round |
| `r16_rr_features.py` | R16: extended RR features (irregularity + shape) for Lane A. | round |
| `r16_rr_seq.py` | R16-A3: causal RR-sequence 1D CNN (small, 3 seeds). | round |

## 报告 / 状态

| 脚本 | 用途 | 状态 |
|---|---|---|
| `r12_wrap_plan_state.py` | One-shot PLAN_STATE.json updater for the R12 wrap-up. | active |
| `r14_write_report.py` | Generate docs/reports/SESSION_REPORT_R14.md from the round's result JSONs. | active |
| `r15_write_report.py` | Generate docs/reports/SESSION_REPORT_R15.md and append the R15 revision to | active |
| `r16_write_report.py` | Generate docs/reports/SESSION_REPORT_R16.md from the R16 result artifacts. | active |

## 未分组（文件名无前缀的历史脚本）

| 脚本 | 用途 | 状态 |
|---|---|---|
| `download_worker.ps1` | Windows 侧下载 worker（PowerShell） | round |
| `push_sanitized.sh` | 研究树净化快照推送（旧流程，已由 Windows %TEMP%\meeti_push 流程取代） | round |
| `r10d_legacy_all.sh` | R10d: 前代模型全量同测试评估 runner | round |
| `r11_train_aug.py` | R11: PTB-val variance suppression arms on top of the r9_form-ptbbal | round |
| `r7_lead2_teacher_probe.py` | Round 7: Lead-II-only teacher probe (decisive test for the KD route). | round |
| `r8a_train_5s_student.py` | R8a/R8b-w5: standalone 5 s single-lead record student (no KD). | round |
| `run_legacy_evals.sh` | 前代模型同测试评估 runner | round |

## archived（`scripts/archive/`）

| 脚本 | 用途 | 状态 |
|---|---|---|
| `build_selected_manifest_from_archive.py` | 从已解压的 Zenodo MEETI 10k 元数据中选择小规模训练子集并生成 MIMIC 原始信号 URL。 | archived |
| `cleanup_mimic_raw.py` | cleanup_mimic_raw.py — 安全删除已处理/已校验的 MIMIC 原始 .hea/.dat。 | archived |
| `download_mimic_subset.sh` | set -euo pipefail | archived |
| `eval_legacy_models_same_test.py` | 把仓库 pc_tools/ecg_dl 里的前代 .h5 模型放到与双头/KD 完全相同的测试上。 | archived |
| `prepare_mimic_subset.py` | prepare_mimic_subset.py — 从 MIMIC-IV-ECG record_list.csv 生成 MEETI 蒸馏用下载清单。 | archived |
| `process_mimic_raw.py` | process_mimic_raw.py — MIMIC 原始 .hea/.dat → 学生 250 点拍窗 + 教师 5s 段缓存。 | archived |
| `validate_mimic_subset.py` | validate_mimic_subset.py — 校验已下载的 MIMIC-IV-ECG 子集。 | archived |

## models_patch/

被 import 的补丁模块（`ptbxl_builders.py`），不移动、不归档。
