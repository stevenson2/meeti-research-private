# N/V研究阶段：新增200患者、四格三种子与一次独立评价

2026-10-28阶段收官：12项训练及12模型的一次预先隔离患者评价均完成。此前27项及低通大模型3项只作历史背景，没有重跑。这里发布经检查的代码、聚合指标与研究说明，不是运行树完整备份。

- [完整阶段报告](RESULTS.md)
- [独立复核及冻结后审计修正说明](REVIEW.md)
- [固定协议摘要](PROTOCOL_SUMMARY.json)
- [模型身份（无权重）](MODEL_IDENTITIES.json)
- [代码与来源哈希](../../../experiments/nv_expand200_20261026/README.md)
- [独立评价12项汇总](independent_runs.csv)、[同seed差值](independent_same_seed_pairs.csv)、[描述性患者聚类区间](independent_paired_intervals.csv)
- [开发集旧/新/合并汇总](development_runs.csv)、[开发集配对差值](development_same_seed_pairs.csv)

核心判断：N位置稳健性有重复改善，原位N误判与V漏检仍有取舍，没有全面胜出的单一模型。当前独立40人已开启；未来重新选门限/模型时不能再次称为未见确认集。数据扩充同时改变了拍数、更新数、类权重与dev组成，不能作患者多样性单因素因果结论。

图表和CSV仅含聚合研究结果；没有原始ECG、逐拍真值/分数、患者清单、模型权重、训练缓存、日志、会话状态、私有密钥或完整冻结账本。正文中提及的本地证据文件并不表示它们包含在此仓库。真实AFE、设备/INT8及AF/VF不由本轮替代。
