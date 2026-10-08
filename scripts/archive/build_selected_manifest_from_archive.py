#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从已解压的 Zenodo MEETI 10k 元数据中选择小规模训练子集并生成 MIMIC 原始信号 URL。
注意: Zenodo 10k 包内 signal 为空(0,0), 需要从 MIMIC-IV-ECG 下载 .dat/.hea。
"""
import argparse, json, time, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

BASE_URL = "https://physionet.org/files/mimic-iv-ecg/1.0"
ABN = ["atrial fibrillation","atrial flutter","afib","a fib","supraventricular",
       "ventricular","infarct","infarction","ischemia","ischemic","block",
       "pacemaker","paced","prolonged qt","st elevation","st depression",
       "t wave inversion","premature"]
NOR = ["normal sinus rhythm","sinus rhythm","normal ecg","within normal limits"]

def cls(text):
    t = str(text).lower()
    if any(k in t for k in ABN): return "abnormal"
    if any(k in t for k in NOR): return "normal"
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/d/MEETI_10k/extracted/20260209_MEETI")
    ap.add_argument("--groups", default="p1000:p1014")
    ap.add_argument("--n-normal", type=int, default=4000)
    ap.add_argument("--n-abnormal", type=int, default=4000)
    ap.add_argument("--out-dir", default="/home/devcontainers/meeti/data/selected8k")
    ap.add_argument("--max-scan", type=int, default=0, help="0=不限制")
    args = ap.parse_args()
    root = Path(args.root)
    a,b = args.groups.split(":"); gi, gj = int(a[1:]), int(b[1:])
    groups = [f"p{i}" for i in range(gi, gj+1)]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    t0=time.time(); rows=[]; n=0; err=0
    for g in groups:
        gdir = root/g
        if not gdir.exists():
            print("skip", g, flush=True); continue
        for mat in sorted(gdir.rglob("*.mat")):
            n += 1
            if args.max_scan and n > args.max_scan: break
            try:
                d = loadmat(str(mat), squeeze_me=False)
                rid = str(np.asarray(d["id"]).ravel()[0]) if "id" in d else mat.stem
                report = str(np.asarray(d.get("report", [""])).ravel()[0])[:300]
                llm = str(np.asarray(d.get("LLM_Interpretation", [""])).ravel()[0])[:300]
                sig_shape = str(np.asarray(d.get("signal", np.zeros((0,0)))).shape)
                nf = sum(1 for k in d.keys() if str(k).startswith("featuredb_lead_"))
                # path relative like p1000/p10000032/s40689238/40689238
                rel = mat.relative_to(root).as_posix().replace(".mat","")
                rows.append({"record_path": "files/"+rel, "stem": "files/"+rel,
                             "study_id": rid, "subject_id": rel.split("/")[1],
                             "report": report, "llm": llm,
                             "report_class": cls(report),
                             "signal_shape": sig_shape, "n_featuredb": nf})
            except Exception as e:
                err += 1
            if n % 2000 == 0:
                print(f"scanned={n} err={err} elapsed={time.time()-t0:.0f}s", flush=True)
        if args.max_scan and n > args.max_scan: break
    df = pd.DataFrame(rows)
    df.to_csv(out/"scanned_manifest.csv", index=False)
    print("scanned", len(df), "err", err, flush=True)
    normal = df[df.report_class=="normal"]
    abnormal = df[df.report_class=="abnormal"]
    unk = df[df.report_class=="unknown"]
    pick=[]
    if len(normal): pick.append(normal.sample(min(args.n_normal,len(normal)), random_state=42))
    if len(abnormal): pick.append(abnormal.sample(min(args.n_abnormal,len(abnormal)), random_state=43))
    need = (args.n_normal+args.n_abnormal) - sum(len(x) for x in pick)
    if need>0 and len(unk): pick.append(unk.sample(min(need,len(unk)), random_state=44))
    sel=pd.concat(pick)
    urls=[]
    for _,r in sel.iterrows():
        base=f"{BASE_URL}/{r['record_path']}"
        urls += [base+".hea", base+".dat"]
    sel.to_csv(out/"selected_manifest.csv", index=False)
    (out/"urls.txt").write_text("\n".join(urls)+"\n", encoding="utf-8")
    summary={"scanned":len(df),"selected":len(sel),"urls":len(urls),
             "class_dist":sel.report_class.value_counts().to_dict(),
             "est_mb":round(len(sel)*0.117,1),
             "elapsed_s":round(time.time()-t0,1)}
    (out/"summary.json").write_text(json.dumps(summary,indent=2), encoding="utf-8")
    print(summary, flush=True)

if __name__ == "__main__":
    main()
