"""Only six synthetic saved scores; never opens independent data or a model."""
from common import *
from independent_evaluate import summarize_predictions

def main():
    assert not (ROOT/'independent_opened.json').exists(),'This check is designed to precede release'
    names=['synthetic_mixed','synthetic_N_only','synthetic_V_only','synthetic_boundary_only','synthetic_empty']
    meta=dict(y=np.array([0,1,0,1,0,1],np.int8),patient=np.array([0,0,1,2,3,3],np.int16),avail7=np.array([1,1,1,1,0,0],bool))
    base=np.array([.9,.8,.2,.8,.2,.8],np.float32);pred={}
    for delta in SHIFTS:
        p=base.copy()
        if delta==-4:p[2]=.8
        if delta==4:p[1]=.2
        pred[delta]=(p,p-.5)
    stats,shifts,changes=summarize_predictions(meta,names,pred)
    assert stats['identity_patients']==5 and stats['NV_patients']==4
    assert stats['zero']['N_false_positive']==1 and stats['zero']['V_detected']==3
    assert stats['robust']['N_common']==stats['robust']['V_common']==2
    assert stats['robust']['N_any_wrong']==2 and stats['robust']['V_all_detected']==1 and stats['robust']['any_flip']==2
    assert stats['macro_N_any_wrong_rate']==1 and stats['macro_V_all_detected_rate']==.5
    assert stats['macro_AUC']==.5 and stats['macro_AP']==.75
    pp={p['patient']:p for p in stats['per_patient']}
    for key in ['synthetic_boundary_only','synthetic_empty']:
        row=pp[key]
        assert row['worst_N_delta'] is None and row['worst_V_delta'] is None
        assert row['N_any_wrong_rate'] is None and row['V_all_detected_rate'] is None and row['any_flip_rate'] is None
    assert pp['synthetic_V_only']['ap']==1 and pp['synthetic_V_only']['auc'] is None
    assert pp['synthetic_N_only']['ap'] is None
    # JSON must preserve undefined values as null, with no non-standard NaN tokens.
    json.dumps(dict(stats=stats,shifts=shifts,changes=changes),allow_nan=False)
    writej(ROOT/'synthetic_independent_statistics_check.json',dict(passed=True,finished=now(),synthetic_rows=6,synthetic_identities=5,no_model_inference=True,no_training_fit=True,no_independent_content_access=True,implementation_sha256=sha(ROOT/'independent_evaluate.py'),checks=['pooled and patient-equal counts','common-set exclusion','N-any and V-all joint conditions','undefined worst shift when only boundary samples exist','single-class and empty patients','macro AUC/AP excludes positive-only AP','strict JSON nulls instead of NaN']))
    print('PURE SYNTHETIC INDEPENDENT STATISTICS CHECK PASSED; no signal/model/independent access',flush=True)

if __name__=='__main__':main()
