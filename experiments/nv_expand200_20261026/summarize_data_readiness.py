"""Development/transport facts only; no independent content access."""
from pathlib import Path
import json,csv,datetime
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    assert not (ROOT/'independent_opened.json').exists(),'Keep this pre-opening readiness snapshot unchanged; final test facts belong in the final report'
    f=read(ROOT/'patient_split_frozen.json');segments=read(ROOT/'segments_frozen.json');prep=read(ROOT/'prep_manifest.json');qc=read(ROOT/'development_qc.json');acq=read(ROOT/'acquisition_complete.json');audit=read(ROOT/'exposure_chain_audit.json');assign={p['patient']:p['group'] for p in f['patients']}
    roster=[]
    for group,patients in prep['patients'].items():
        for p in patients:
            old=p['patient'] in f['legacy_split'][group];usable=sum(s['patient']==p['patient'] for s in prep['segments']);roster.append(dict(patient=p['patient'],group=group,origin='legacy' if old else 'new',assigned_segments=usable if old else 4,usable_development_segments=usable,N=p['N'],V=p['V'],custody='DEVELOPMENT_EXPOSED'))
    for p in f['patients']:
        if p['group']=='independent':roster.append(dict(patient=p['patient'],group='independent',origin='new',assigned_segments=4,usable_development_segments='',N='',V='',custody='ENCRYPTED_NOT_OPENED'))
    with (ROOT/'patient_roster_development_and_custody.csv').open('w',newline='',encoding='utf-8-sig') as o:
        w=csv.DictWriter(o,fieldnames=list(roster[0]));w.writeheader();w.writerows(roster)
    body='# 数据获取、曝光与开发准备结果\n\n'
    body+=f"开发准备快照生成于 {datetime.datetime.now(datetime.timezone.utc).isoformat()}。下文‘未开封’指此快照时刻；最终训练及独立评价状态另见最终交付报告。\n\n"
    body+='新增200位患者的120/40/40身份分配、每人最多4段取段规则均已冻结并实际执行。已查历史内容获取/训练开发记录中无候选命中，原B方案名单未替换。该审计不证明任意未记录的外部历史访问不存在；同账户加密与程序隔离也不冒充第三方保管。旧sealed信号/atr未打开或哈希，其历史EXPOSURE_UNVERIFIED状态不变。\n\n'
    body+=f"实际获取 {acq['completed_payload_files']}/{acq['expected_payload_files']} 个源文件，应用读取的响应payload累计 {acq['received_payload_bytes']:,} B（包括200份RECORDS元数据；不含HTTP/TLS开销，不是全部网络实收认证）。唯一缺失为train患者p09128的p09128_s49.atr，镜像和主站均404。按统一完整性规则整段排除，不换人/补段，p09128保留其余3段；其余患者仍最多4段。\n\n"
    body+='| 开发集合 | 身份患者 | 可拟合N/V患者 | N | V | 有V患者 | 零偏移窗 | 七偏移共同窗 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for group in ['train','dev']:
        c=qc['counts'][group];body+=f"|{group}|{c['patients']}|{c['fitted_NV_patients']}|{c['N']:,}|{c['V']:,}|{c['V_patients']}|{c['windows']:,}|{c['avail7']:,}|\n"
    body+='\n新增有V患者为train75位、dev24位，达到原建议的40/12覆盖目标。没有机械每人1333拍上限，没有为凑类别平衡删除V；纯N患者与无N/V身份均保留。train中p02813、p05389、p07487、p08269没有字面N/V，因此不拟合，但身份不移出分组。\n\n'
    body+='独立评价40人/160段的480个源文件均已流式加密入独立vault，未解密、未解析信号/hea/atr、未统计N/V支持数、未运行模型。独立标签在清单中留空，不能当成0。用户已明确确认：全部12项训练/选模完成，最佳checkpoint、开发比较及恢复/预测审计冻结后一次评价。释放门已实测拒绝当前不完整矩阵，未发生解密。\n\n'
    body+=f"每轮 {qc['steps_per_epoch']:,} 次更新，最多80轮，总CosineDecay {qc['decay_steps']:,} 步；float32类权重 {qc['class_weights']}。训练只存零偏移缓存，dev保存7个真实重切窗口；按批只读mmap。旧缓存经hash及逐批复制核对，新流经同一冻结C前端与float32逐窗z-score。实际更新步数与各模型最佳/停止轮数另报告，更多更新不能全部归因于患者多样性。\n\n"
    for group in ['train','dev']:
        pp=sorted(prep['patients'][group],key=lambda p:p['V'],reverse=True);total=sum(p['V'] for p in pp);body+=f"{group}最大V贡献者{pp[0]['patient']}为{pp[0]['V']:,}/{total:,}（{pp[0]['V']/total:.2%}）；前三占{sum(p['V'] for p in pp[:3])/total:.2%}。阳性患者覆盖扩大不等于已证明形态多样性或泛化；仍需逐患者及独立评价结果。\n\n"
    body+='同源整段dat的已有SHA256检查覆盖新800个dat和旧100个允许开发dat，未发现跨组完全相同源文件；这不是全体局部窗近重复审查。一个NumPy私有header API兼容问题在正式训练前修正为公开reader，数据规则未改变。纯合成统计与隔离门测试不包含额外模型拟合。\n\n'
    body+='本文件仅交付数据获取/准备事实；不表示12项训练或独立评价已完成。训练真实进度见status.json，各run的progress.json与完整epoch快照。\n'
    (ROOT/'DATA_READINESS_REPORT.md').write_text(body,encoding='utf-8');print('DATA READINESS FACTS SAVED; independent labels left unknown')
if __name__=='__main__':main()
