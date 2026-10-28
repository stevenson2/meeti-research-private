"""Render only actual terminal outputs. Never starts training/evaluation."""
from common import *
import csv,datetime
def table(rows,cols):
    def f(v):
        if v is None:return '—'
        return f'{v:.5f}' if isinstance(v,float) else str(v)
    return '|'+ '|'.join(cols)+'|\n|'+'|'.join(['---']*len(cols))+'|\n'+'\n'.join('|'+ '|'.join(f(r.get(c)) for c in cols)+'|' for r in rows)+'\n'
def pp(x):return 100*x if x is not None else None
def local(t):return datetime.datetime.fromisoformat(t).astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def delta_text(rows,key,percentage=False):
    vals=[next(r[key] for r in rows if r['seed']==s) for s in [42,43,44]]
    return ' / '.join('未定义' if v is None else (f'{100*v:+.2f}pp' if percentage else f'{int(v):+d}') for v in vals)
def main():
    train=readj(ROOT/'status.json');test=readj(ROOT/'independent_status.json')
    assert all(r['status']=='completed' for r in train['runs']+test['runs']) and len(train['runs'])==len(test['runs'])==12
    assert readj(ROOT/'training_recovery_audit.json')['passed'] and readj(ROOT/'independent_prediction_audit.json')['passed']
    cmp=readj(ROOT/'independent_comparison.json');dev=readj(ROOT/'development_comparison.json');qc=readj(ROOT/'development_qc.json');pre=readj(ROOT/'independent_preparation.json');opened=readj(ROOT/'independent_opened.json');gate=readj(ROOT/'checkpoints_frozen_for_independent.json');acq=readj(ROOT/'acquisition_complete.json');P='C:/ecg_data/round_20261021/work/NV_EXPAND200_20261026/'
    pairs=cmp['same_seed_pairs'];body='# 新增200患者：四格三种子与一次独立评价\n\n'
    body+='本轮实际完成12项新数据训练及其开发评价，并在所有checkpoint、开发比较和恢复/预测审计冻结后，对预先隔离的40位新患者进行一次固定评价。以下判断只针对本次固定数据、网络、初始化和训练协议，不选择最好seed或强迫单一赢家。\n\n'
    for name,label in [('L_LP-L','相同固定窗口训练下，低通是否改善大模型'),('L_LP-S_LP','低通之后，扩大容量是否提供额外收益')]:
        rows=[r for r in pairs if r['pair']==name]
        body+=f"**{label}。**独立评价中，seed42/43/44原位N误判变化为 {delta_text(rows,'N_false_positive')}，原位V检出变化为 {delta_text(rows,'V_detected')}；N任一偏移误判变化为 {delta_text(rows,'N_any_wrong')}，V全部偏移检出变化为 {delta_text(rows,'V_all_detected')}。患者等权原位V检出差为 {delta_text(rows,'macro_V_detection_rate',True)}，患者等权V全部检出差为 {delta_text(rows,'macro_V_all_detected_rate',True)}。负的N误判差为改善，正的V检出差为改善；必须同时读N代价、V收益、患者支持及下方三次原始结果。\n\n"
    body+='上一轮旧数据下的四格结果只作历史背景，本轮没有重跑其冻结实验。新数据矩阵四格共享同一扩充数据、class-weight计算规则与训练/评价条件。相对上一轮，患者、总拍量、实际类权重、优化更新步数和dev患者都发生变化，因此本轮不把跨轮变化全部归因于患者多样性，也不是数据扩充单因素因果试验。\n\n'
    body+='## 实际数据与曝光链\n\n'
    body+=f"新增200人按120train/40dev/40independent冻结，每人最多4段；患者与段清单均先于新信号/标签内容获取。取得{acq['completed_payload_files']}/2400个源文件，累计应用payload {acq['received_payload_bytes']:,} B；唯一缺失train p09128_s49.atr在镜像和主站404，整段按统一完整性规则排除，患者及其余3段保留，不补人或补段。开发实际使用639个新增官方段加旧100段；独立集160段的480个文件先加密保管。\n\n"
    body+='来源为[Icentia11k v1.0官方数据](https://physionet.org/content/icentia11k-continuous-ecg/1.0/)，250Hz胸部固定单导联；只使用字面N/V，Q/S等不改标。N是拍类型，不能当作患者健康。旧40人维持30train/10dev。\n\n'
    data=[]
    for group in ['train','dev']:
        c=qc['counts'][group];data.append({'集合':group,'身份患者':c['patients'],'N/V可拟合患者':c['fitted_NV_patients'],'N':c['N'],'V':c['V'],'有V患者':c['V_patients'],'原位窗':c['windows'],'共同七偏移窗':c['avail7']})
    first=cmp['runs'][0];data.append({'集合':'independent','身份患者':40,'N/V可拟合患者':first['NV_patients'],'N':pre['N'],'V':pre['V'],'有V患者':pre['V_patients'],'原位窗':pre['windows'],'共同七偏移窗':pre['avail7']});body+=table(data,list(data[0]))
    body+='\n新增有V患者train75位、dev24位，均达到原覆盖建议；train四位无N/V身份保留但不拟合。没有1333/人机械上限，也未为了均衡删掉大量V。开发准备快照及完整患者清单见DATA_READINESS_REPORT.md与patient_roster_development_and_custody.csv；该快照的“未开封”指训练前时刻。\n\n'
    body+=f"独立集于 {local(opened['first_opened'])} 首次开启，晚于全部12项结束及冻结门。此后状态为CONFIRM_OPENED，不能在未来调参后把同一批人再次称为未见确认集。旧197个sealed atr始终未打开或哈希，历史EXPOSURE_UNVERIFIED不变。曝光链审计只覆盖已查本地记录，不能证明任何未记录外部历史访问都不存在；本机同账户下的加密、白名单与checkpoint门也不冒充第三方独立保管。\n\n"
    body+='## 独立评价：全部12次原位结果\n\n'
    raw=[{'run':r['id'],'N误判':r['N_false_positive'],'V检出':r['V_detected'],'患者等权N误判%':pp(r['macro_N_false_positive_rate']),'患者等权V检出%':pp(r['macro_V_detection_rate']),'AUC':r['auc'],'AP':r['ap']} for r in cmp['runs']];body+=table(raw,list(raw[0]))
    body+=f"\n零偏移分母：N={pre['N']:,}、V={pre['V']:,}，共{pre['windows']:,}窗。单类或无N/V患者保留身份，未定义指标留空；宏AUC/AP只汇总同时有两类者。AUC/AP描述排序，不能替代固定阈值计数，也不等于真实设备报警表现。\n\n"
    body+='## 独立评价：七偏移稳健性\n\n'
    robust=[{'run':r['id'],'N任一误判':r['N_any_wrong'],'V全部检出':r['V_all_detected'],'等权N任一%':pp(r['macro_N_any_wrong_rate']),'等权V全部%':pp(r['macro_V_all_detected_rate']),'最差统一N偏移':r['worst_N_delta'],'最差统一N误判':r['worst_N_false_positive'],'最差统一V偏移':r['worst_V_delta'],'最差统一V检出':r['worst_V_detected']} for r in cmp['runs']];body+=table(robust,list(robust[0]))
    body+=f"\n七偏移共同分母N={first['N_common']:,}、V={first['V_common']:,}；相对零偏移排除N={pre['N']-first['N_common']}、V={pre['V']-first['V_common']}。"
    body+='七偏移{-4,-2,-1,0,+1,+2,+4}使用预先固定几何共同集合；N任一误判与V全部检出按每拍七个位置联合统计，最差统一偏移则要求所有拍使用同一个delta，两者不可混称。分母、排除身份、每患者各delta与双向变化均有保存。\n\n'
    body+='## 同seed差值：左臂减右臂\n\n'
    pr=[{'pair':r['pair'],'seed':r['seed'],'ΔN原位误判':r['N_false_positive'],'ΔV原位检出':r['V_detected'],'ΔN任一误判':r['N_any_wrong'],'ΔV全部检出':r['V_all_detected'],'Δ等权N原位pp':pp(r['macro_N_false_positive_rate']),'Δ等权V原位pp':pp(r['macro_V_detection_rate'])} for r in pairs];body+=table(pr,list(pr[0]))
    body+='\n主配对为L_LP-L和L_LP-S_LP；S_LP-S与L-S为预先声明背景。没有挑选最好seed、阈值或看起来更好的单一指标。\n\n'
    body+='## 患者聚类区间与不确定性\n\n'
    ci=[{'pair':r['pair'],'endpoint':r['endpoint'],'三seed均值差pp':r['delta_percentage_points'],'95%下限pp':r['CI95_low_pp'],'95%上限pp':r['CI95_high_pp'],'支持患者':r['support_patients'],'有效重抽样次数':r['valid_bootstrap_draws']} for r in cmp['paired_intervals'] if r['seed_view']=='three_seed_mean' and r['pair'] in ['L_LP-L','L_LP-S_LP']];body+=table(ci,list(ci[0]))
    body+='\n按患者身份成簇配对重抽样2000次，所有拍和三个seed留在该患者簇内；seed固定，未把3×40当成120位独立患者。区间未做多重比较校正，仅为描述性区间，不据其中任一好看的区间宣布显著性赢家。逐seed区间及所有背景配对完整见independent_paired_intervals.csv。少量V患者/少量V拍会造成较大比例波动；数量目标不是统计充分性的保证。\n\n'
    body+='## 患者代价与完整原始表\n\n'
    changes=list(csv.DictReader((ROOT/'independent_same_seed_patient_changes.csv').open(encoding='utf-8-sig')))
    for pair in ['L_LP-L','L_LP-S_LP']:
        rr=[r for r in changes if r['pair']==pair];names=sorted({r['patient'] for r in rr});chosen=sorted(names,key=lambda p:sum(abs(int(r['delta_N_false_positive'])) for r in rr if r['patient']==p),reverse=True)[:3]
        chosen+=sorted(names,key=lambda p:sum(abs(int(r['delta_V_detected']))/int(r['V']) if int(r['V']) else 0 for r in rr if r['patient']==p),reverse=True)[:3];chosen=list(dict.fromkeys(chosen));examples=[]
        for p in chosen:
            row={'patient':p}
            for seed in [42,43,44]:
                r=next(r for r in rr if r['patient']==p and int(r['seed'])==seed);row[str(seed)+' ΔN/ΔV']=r['delta_N_false_positive']+' / '+r['delta_V_detected']
            examples.append(row)
        body+=f"\n{pair}：用于解释N计数差及V比例差较大的患者，非数据/模型选择规则。所有40人仍完整保留在CSV中。\n\n"+table(examples,list(examples[0]))
    body+='\n完整逐患者支持、N/V计数、最差/任一/全部条件和AUC/AP在independent_patients.csv；逐条件p、margin、身份在independent/<run>/eval_*/*.npz。个别差异不证明唯一生理机制。\n\n'
    body+='## 开发结果、实际预算与恢复\n\n'
    timing=[]
    for r in train['runs']:
        t=readj(ROOT/'runs'/r['id']/'training_summary.json');timing.append({'run':r['id'],'停止轮':t['epoch'],'最佳轮':t['best_epoch'],'保留状态更新':t['epoch']*STEPS,'最佳更新':t['best_epoch']*STEPS,'尝试次数':len(r['attempts']),'已记录尝试分钟':sum(a.get('seconds',a.get('observed_seconds_lower_bound',0)) for a in r['attempts'])/60,'尝试计时口径':'含中断，仅为下界' if any('seconds' not in a for a in r['attempts']) else '完整墙时','TF末次进程峰值MiB':t['resources'].get('tf_gpu_memory_bytes',{}).get('peak',0)/2**20,'RSS末次进程峰值MiB':t['resources']['rss_peak_mb']})
    body+=table(timing,list(timing[0]))
    body+=f"\n正式矩阵开始 {local(train['started'])}，训练/开发评价终态 {local(train['finished'])}，独立评价终态 {local(test['finished'])}，均为北京时间。每轮{STEPS:,}更新、最多80轮、patience12/min_delta1e-4；实际总样本与更新如表。TF allocator峰值不等于驱动总显存；gpu_observations.csv含桌面等驱动占用，且只是抽样而非连续峰值。\n\n"
    if (ROOT/'process_interruption_20261027.json').exists():
        recovery=readj(ROOT/'process_interruption_20261027.json')
        body+=f"进程中断恢复：S_LP_C_s43原进程丢失时，最后完整快照为第10轮，第11轮至少已执行{recovery['observed_discarded_partial_updates_lower_bound']:,}个batch更新但未形成完整快照。从第10轮的模型、Adam、Dropout及采样RNG原状恢复并重放第11轮；前六项未重跑，数据与冻结训练核心hash复核一致。恢复时只修正调度脚本跨重启的尝试计数并接回尚未执行的等待流程，数值训练实现未变。中断原因和原退出码未知，当前Linux运行时长重置已有记录。表中‘保留状态更新’及development_runs.csv完整轮次耗时不包含被丢弃的部分轮更新；实际执行总量更高，精确丢弃数量未知。中断尝试时长只能由最后遥测给出下界，不能当成完整精确运行时长。末次进程峰值不冒充多次进程的连续峰值，原第10轮恢复点另有字节一致备份。证据见process_interruption_20261027.json。\n\n"
    body+='旧10dev与新40dev均参与选模，不能称盲测。development_runs.csv、development_same_seed_pairs.csv分别提供旧/新/合并三种口径，development_comparison_frozen.json证明开发结果先于独立开启冻结。完整epoch快照保存模型、Adam、采样RNG、Dropout、epoch/best/wait；training_recovery_audit.json已加载核对12项完整状态和全部180份固定诊断预测（96份dev预测及84份旧train病例预测）。best_model.keras为推理模型，完整续训须用epoch快照加state.json。independent_prediction_audit.json重新核对96份独立预测、身份、计数、重复和BN不变。\n\n'
    body+='## 可以和不能推断的范围\n\n低通同时改变频率响应与有效感受野，不能从这四格认定混叠为唯一根因。更多患者和更分散标签支持不自动证明形态多样性；本轮也增加拍数、更新数并按同一规则重算实际类权重，跨轮数据收益不是单因素因果结论。三seed用于观察训练随机性，独立患者数仍是40。无事前临床容忍差，不强迫单一赢家。\n\n参考标注锚点绕过了检测器；1秒上下文窗不是已证明的纯QRS归因。N/V拍级误判不能换算成每小时报警，S/Q及AF/VF不混入两输出。本轮没有验证真实AFE保真、设备/INT8、AF/VF或部署，没有修改产品仓、GitHub PR/README、串口/刷写或其他agent的加速/ESP-DL工作。公共数据胸部单导联的新患者识别结果不能替代这些验证。\n\n固定范围到此结束，不自动增加患者、段数、模型、剂量、seed或下一轮训练。\n\n'
    body+=f"![四格三seed取舍]({P}independent_four_grid.png)\n\n![主配对患者聚类区间]({P}independent_primary_intervals.png)\n\n关键交付：patient_split_frozen.csv/json、segments_frozen.json、acquisition_ledger.json、development_qc.json、frozen_config.json、initializations.json、12个run最佳模型及恢复快照、开发与独立逐窗预测/统计、两类审计、全部同seed差值及描述性区间、delivery_manifest.json。\n"
    (ROOT/'FINAL_REPORT.md').write_text(body,encoding='utf-8')
    contracts={}
    for r in train['runs']:
        out=ROOT/'runs'/r['id'];snap=out/readj(out/'last.json')['snapshot']
        contracts[r['id']]=dict(selected_weights=dict(path=str(out/'best.weights.h5'),sha256=gate['checkpoints'][r['id']]['sha256']),inference_model=dict(path=str(out/'best_model.keras'),sha256=sha(out/'best_model.keras'),role='inference only'),continuation_snapshot=dict(directory=str(snap),model_sha256=sha(snap/'model.keras'),state_sha256=sha(snap/'state.json'),role='full continuation needs model.keras plus state.json; selected best weights may come from an earlier epoch'))
    writej(ROOT/'model_label_contract.json',dict(training_config_sha256=sha(ROOT/'frozen_config.json'),input=dict(shape=[250,1],dtype='float32',sampling_hz=250,chain=readj(ROOT/'prep_manifest.json')['chain'],anchor='[annotation_R-125,annotation_R+125)',normalization='frozen C float32 per-window z-score'),output={'0':'literal N','1':'literal V'},threshold='strict pV>0.50',scope='N/V-only annotation-anchored research classifier, not calibrated general-population risk; S/Q/AF/VF are not merged into these classes',models=contracts))
    print('FINAL CHINESE REPORT SAVED; all12+12 terminal and audited')
if __name__=='__main__':main()
