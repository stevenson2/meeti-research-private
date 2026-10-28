"""Only the development allowlist can be opened here. Bounded per-segment arrays."""
from pathlib import Path
import json,os,hashlib,ctypes as C,time,collections
import numpy as np
import wfdb
from scipy.signal import resample_poly
ROOT=Path(__file__).resolve().parent;R=ROOT.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,o):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
F=read(ROOT/'patient_split_frozen.json');STORE=Path(F['data_store']);DATA=STORE/'cache';DATA.mkdir(exist_ok=True)
ASSIGN={r['patient']:r['group'] for r in F['patients']};SEG=read(ROOT/'segments_frozen.json')['segments']
OLD=R/'NV_LONGRUN_20261024';BASE=read(R/'NV_BASELINE_R1/prep_manifest.json')
SCHEMA={'y':np.int8,'patient':np.int16,'segment':np.int32,'annotation_index':np.int64,'sample_orig':np.int64,'anchors_250':np.int64,'avail7':np.bool_}
SHIFTS=[-4,-2,-1,0,1,2,4]
FP,IP=C.POINTER(C.c_float),C.POINTER(C.c_int64)
SO=R/'V3A_H5_INT8_SAME_WINDOWS/frontend2.so';lib=C.CDLL(str(SO));lib.product_chain.argtypes=[FP,C.c_int,FP];lib.zscore_windows.argtypes=[FP,IP,C.c_int,FP]
def authorize(s):
    assert ASSIGN[s['patient']]==s['group'] and s['group'] in ['train','dev'],'INDEPENDENT_CONTENT_DENIED'
    p=STORE/'development'/s['group']/s['patient']/s['segment']
    assert p.resolve().is_relative_to((STORE/'development').resolve())
    return p
def attrs(s):
    p=authorize(s);header=wfdb.rdheader(str(p))
    assert header.file_name==[s['segment']+'.dat'],'Unexpected signal path in header'
    assert header.n_sig==1 and float(header.fs)==250.0 and header.units==['mV']
    assert header.adc_gain[0]>0 and header.sig_len>0 and header.fmt==['16']
    assert p.with_suffix('.dat').stat().st_size==header.sig_len*2
    ann=wfdb.rdann(str(p),'atr');symbol=np.asarray(ann.symbol);sample=np.asarray(ann.sample,np.int64)
    want=np.flatnonzero((symbol=='N')|(symbol=='V'));anchors=np.rint(sample[want]*250/float(header.fs)).astype(np.int64)
    valid=(sample[want]>=0)&(sample[want]<header.sig_len)&(anchors>=125)&(anchors+125<=header.sig_len)
    selected=want[valid];anchor=anchors[valid];y=(symbol[selected]=='V').astype(np.int8)
    # Annotation index is the immutable beat identity; exact duplicate sample+label
    # or contradictory N/V anchors stop for review, never silently relabel/drop.
    pairs=np.stack([sample[selected],y],1)
    assert len(np.unique(pairs,axis=0))==len(pairs),f'Duplicate NV sample+label {s["segment"]}'
    assert len(np.unique(sample[selected]))==len(selected),f'Conflicting NV anchor {s["segment"]}'
    a7=(anchor-4>=125)&(anchor+4+125<=header.sig_len)
    return header,ann,selected,anchor,y,a7,dict(all_annotation_counts=dict(collections.Counter(ann.symbol)),raw_N=int((symbol=='N').sum()),raw_V=int((symbol=='V').sum()),excluded_NV_boundary_or_source=int((~valid).sum()),N=int((y==0).sum()),V=int((y==1).sum()),windows=len(y),avail7=int(a7.sum()),samples=int(header.sig_len),fs=float(header.fs),gain=float(header.adc_gain[0]),baseline=int(header.baseline[0]))
def windows(stream,anchors):
    anchors=np.ascontiguousarray(anchors,np.int64);x=np.empty((len(anchors),250),np.float32)
    if len(anchors):lib.zscore_windows(stream.ctypes.data_as(FP),anchors.ctypes.data_as(IP),len(anchors),x.ctypes.data_as(FP))
    return x
def signal(s,header):
    p=authorize(s);rec=wfdb.rdrecord(str(p),physical=False,channels=[0]);v=rec.d_signal[:,0]
    assert len(v)==header.sig_len
    raw=(np.asarray(v,np.float64)-rec.baseline[0])/rec.adc_gain[0];assert np.isfinite(raw).all()
    raw500=np.ascontiguousarray(resample_poly(raw,2,1),np.float32);stream=np.empty((len(raw500)+1)//2,np.float32)
    lib.product_chain(raw500.ctypes.data_as(FP),len(raw500),stream.ctypes.data_as(FP));assert len(stream)==len(raw) and np.isfinite(stream).all()
    return stream,dict(raw_min=int(v.min()),raw_max=int(v.max()),raw_int16_limit_samples=int(((v==-32768)|(v==32767)).sum()))
def dataoffset(path):
    with path.open('rb') as f:
        version=np.lib.format.read_magic(f)
        reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        shape,fortran,dtype=reader(f);return f.tell()
def mapped_rows(path,n,k,start,count):
    return np.memmap(path,mode='r+',dtype=np.float32,offset=dataoffset(path)+start*k*250*4,shape=(count,k,250))
def main():
    acquisition=read(ROOT/'acquisition_complete.json')
    availability=read(ROOT/'source_availability.json')
    missing=set(availability['unavailable_segment_bases'])
    assert all('HTTP 404' in f['error'] and f['path'].rsplit('.',1)[0] in missing for f in acquisition['failures'])
    assert not (ROOT/'development_qc.json').exists(),'Do not overwrite a frozen development dataset'
    t=time.monotonic();ledger=read(ROOT/'acquisition_ledger.json')
    devseg=[s for s in SEG if s['group'] in ['train','dev'] and s['relative_base'] not in missing]
    # Prove guard rejection without opening a holdout file.
    try:authorize(next(s for s in SEG if s['group']=='independent'))
    except AssertionError as e:assert 'INDEPENDENT_CONTENT_DENIED' in str(e)
    else:raise AssertionError('Guard failed')
    names={g:sorted(F['legacy_split'][g]+[r['patient'] for r in F['patients'] if r['group']==g]) for g in ['train','dev']}
    maps={g:{p:i for i,p in enumerate(names[g])} for g in names}
    segments=[dict(group=s['group'],patient=s['patient'],segment=s['segment'],source='legacy') for s in BASE['segments']]
    scan=[];newmeta={g:{k:[] for k in SCHEMA} for g in names}
    for i,s in enumerate(devseg):
        p=authorize(s)
        for ext in ['.hea','.atr','.dat']:
            item=ledger['files'][s['relative_base']+ext]
            assert not item['encrypted'] and sha(p.with_suffix(ext))==item['plaintext_sha256']
        h,ann,ix,anchor,y,a7,q=attrs(s);si=len(segments);segments.append(dict(group=s['group'],patient=s['patient'],segment=s['segment'],source='new'))
        g=s['group'];data=dict(y=y,patient=np.full(len(y),maps[g][s['patient']],np.int16),segment=np.full(len(y),si,np.int32),annotation_index=ix,sample_orig=np.asarray(ann.sample,np.int64)[ix],anchors_250=anchor,avail7=a7)
        for k in SCHEMA:newmeta[g][k].append(data[k].astype(SCHEMA[k]))
        scan.append(dict(**s,**q,segment_index=si))
        if (i+1)%40==0:print('DEVELOPMENT HEADER/LABEL QC',i+1,'/',len(devseg),flush=True)
    metas={};oldmetas={};counts={}
    oldeq=read(OLD/'input_equivalence.json')
    for g in names:
        with np.load(OLD/f'{g}_meta.npz') as f:oldmeta={k:f[k] for k in SCHEMA}
        assert sha(OLD/f'{g}_meta.npz')==oldeq['datasets'][g]['metadata_sha256']
        pmap=np.array([maps[g][p['patient']] for p in BASE['patients'][g]],np.int16);oldmeta['patient']=pmap[oldmeta['patient']]
        oldmetas[g]=oldmeta
        metas[g]={k:np.concatenate([oldmeta[k],*newmeta[g][k]]).astype(SCHEMA[k]) for k in SCHEMA}
        y=metas[g]['y'];counts[g]=dict(windows=len(y),N=int((y==0).sum()),V=int((y==1).sum()),patients=len(names[g]),fitted_NV_patients=len(np.unique(metas[g]['patient'])),V_patients=len(np.unique(metas[g]['patient'][y==1])),legacy_windows=len(oldmeta['y']),new_windows=len(y)-len(oldmeta['y']),avail7=int(metas[g]['avail7'].sum()))
        np.savez(DATA/f'{g}_meta.npz',**metas[g])
    import shutil
    needed=sum(counts[g]['windows']*(1 if g=='train' else 7)*250*4+128 for g in names)
    assert shutil.disk_usage(STORE).free-needed>8*1024**3,'Insufficient disk for complete caches with headroom'
    for g in names:
        k=1 if g=='train' else 7;n=counts[g]['windows'];dest=DATA/f'{g}_X.npy';mm=np.lib.format.open_memmap(dest,mode='w+',dtype=np.float32,shape=(n,k,250));del mm
        source=np.load(OLD/f'{g}_X.npy',mmap_mode='r')
        assert sha(OLD/f'{g}_X.npy')==oldeq['datasets'][g]['mmap_sha256']
        nold=counts[g]['legacy_windows']
        for start in range(0,nold,4096):
            end=min(start+4096,nold);mm=mapped_rows(dest,n,k,start,end-start)
            if g=='train':mm[:,0,:]=source[start:end,4,:]
            else:mm[:]=source[start:end]
            assert np.array_equal(mm[:,0,:],source[start:end,4,:]) if g=='train' else np.array_equal(mm,source[start:end])
            mm.flush();del mm
        del source
    offsets={g:counts[g]['legacy_windows'] for g in names};extra=[]
    for i,s in enumerate(scan):
        h,ann,ix,anchor,y,a7,q=attrs(s);stream,rawstats=signal(s,h);g=s['group'];n=len(y);k=1 if g=='train' else 7
        flat=0
        if n:
            mm=mapped_rows(DATA/f'{g}_X.npy',counts[g]['windows'],k,offsets[g],n)
            for col,d in enumerate([0] if g=='train' else SHIFTS):
                valid=(anchor+d>=125)&(anchor+d+125<=len(stream));mm[:,col,:]=0
                xf=windows(stream,anchor[valid]+d);assert np.isfinite(xf).all();mm[valid,col,:]=xf
                if d==0:flat=int((np.std(xf,axis=1)<1e-6).sum())
            mm.flush();del mm
        offsets[g]+=n;extra.append(dict(segment=s['segment'],group=g,patient=s['patient'],**rawstats,zero_flat_windows=flat))
        if (i+1)%20==0:print('DEVELOPMENT FLOAT CACHE',i+1,'/',len(devseg),dict(offsets),flush=True)
    assert all(offsets[g]==counts[g]['windows'] for g in names)
    per={}
    for g in names:
        meta=metas[g];per[g]=[dict(patient=p,N=int(((meta['patient']==pi)&(meta['y']==0)).sum()),V=int(((meta['patient']==pi)&(meta['y']==1)).sum())) for pi,p in enumerate(names[g])]
    prep=dict(patients=per,segments=segments,counts=counts,scan=scan,excluded_unavailable_segments=sorted(missing),signal_flags=extra,chain=BASE['chain'],anchor=BASE['anchor'],labels='literal N/V only',frontend_sha256=sha(SO),old_inputs_copied_byte_identical=True,new_independent_content_opened=False,cache_train_offsets=[0],cache_dev_offsets=SHIFTS,seconds=time.monotonic()-t)
    write(ROOT/'prep_manifest.json',prep)
    n=counts['train']['windows'];steps=(n+255)//256;cw=np.array([n/(2*counts['train']['N']),n/(2*counts['train']['V'])],np.float32)
    manifest=dict(finished=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),counts=counts,steps_per_epoch=steps,decay_steps=steps*80,class_weights=cw.astype(float).tolist(),cache_files={f'{g}_{suffix}':{'path':str(DATA/f'{g}_{suffix}'),'sha256':sha(DATA/f'{g}_{suffix}'),'bytes':(DATA/f'{g}_{suffix}').stat().st_size} for g in names for suffix in ['X.npy','meta.npz']},all_used_dev_files_hash_match=True,excluded_unavailable_segments=sorted(missing),old_input_equivalence=True,independent_guard_rejection_test=True,duplicate_or_conflicting_NV_anchors=False,independent_content_parsed=False,targets={'new_train_V_patients':40,'new_dev_V_patients':12},actual_new_V_patients={g:sum(p['V']>0 and p['patient'] in ASSIGN for p in per[g]) for g in names})
    write(ROOT/'development_qc.json',manifest)
    print('DEVELOPMENT QC COMPLETE',json.dumps(counts),flush=True)
if __name__=='__main__':main()
