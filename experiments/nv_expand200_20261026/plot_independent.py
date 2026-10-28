"""Publication-style all-seed fixed-protocol plots; no fitting or selection."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS=['S','L','S_LP','L_LP']
PANELS=[('macro_N_false_positive_rate','Zero-offset N false-positive rate','lower'),
        ('macro_V_detection_rate','Zero-offset V sensitivity','higher'),
        ('macro_N_any_wrong_rate','N error at any of seven offsets','lower'),
        ('macro_V_all_detected_rate','V detected at all seven offsets','higher')]

def main():
    status=readj(ROOT/'independent_status.json')
    assert len(status['runs'])==12 and all(r['status']=='completed' for r in status['runs'])
    assert readj(ROOT/'independent_prediction_audit.json')['passed']
    cmp=readj(ROOT/'independent_comparison.json');runs=cmp['runs']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(12.4,9.2),layout='constrained')
    colours=['#276FBF','#B85450','#37855B'];markers=['o','s','^']
    for ax,(key,title,direction) in zip(axes.ravel(),PANELS):
        yy=[]
        for seed,col,mark in zip([42,43,44],colours,markers):
            vals=[next(r[key] for r in runs if r['model']==model and r['seed']==seed) for model in MODELS]
            y=np.array([100*v if v is not None else np.nan for v in vals]);yy.append(y)
            ax.plot(range(4),y,color=col,marker=mark,markersize=6,linewidth=1,alpha=.85,label=f'Seed {seed}')
        mean=np.nanmean(np.stack(yy),0)
        ax.scatter(range(4),mean,marker='_',s=310,color='#161616',linewidth=2.5,zorder=5,label='Fixed 3-seed mean')
        ax.set(title=title+' ('+direction+' is better)',ylabel='Patient-equal rate (%)',xticks=range(4),xticklabels=MODELS,xlim=(-.3,3.3))
        ax.grid(axis='y',alpha=.20);ax.set_axisbelow(True)
        finite=np.concatenate(yy);finite=finite[np.isfinite(finite)]
        if finite.size:
            lo,hi=float(finite.min()),float(finite.max());pad=max((hi-lo)*.18,.2)
            ax.set_ylim(max(0,lo-pad),min(100,hi+pad))
        ax.text(.01,.02,'Same seed linked across models; no best-seed selection',transform=ax.transAxes,fontsize=8,color='#555555')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle('Prospective 40-patient evaluation: all four models and all three seeds\nFixed threshold p(V) > 0.50; patient-equal endpoints',fontsize=15)
    fig.savefig(ROOT/'independent_four_grid.png',dpi=180)
    fig.savefig(ROOT/'independent_four_grid.svg')
    plt.close(fig)
    ci=[r for r in cmp['paired_intervals'] if r['pair'] in ['L_LP-L','L_LP-S_LP'] and r['seed_view']=='three_seed_mean']
    fig,axes=plt.subplots(1,2,figsize=(13.0,4.7),layout='constrained',sharey=True)
    for ax,pair in zip(axes,['L_LP-L','L_LP-S_LP']):
        for j,(key,title,direction) in enumerate(PANELS):
            r=next(r for r in ci if r['pair']==pair and r['endpoint']==key)
            if r['delta_percentage_points'] is not None:
                point,lo,hi=[r[k] for k in ['delta_percentage_points','CI95_low_pp','CI95_high_pp']]
                ax.plot([lo,hi],[j,j],color='#276FBF',linewidth=2)
                ax.scatter([point],[j],color='#161616',s=35,zorder=4)
        ax.axvline(0,color='#999999',linewidth=1,linestyle='--')
        ax.set(title=pair+' (left minus right)',xlabel='Paired difference (percentage points)',yticks=range(4),yticklabels=['Zero N error (lower better)','Zero V sensitivity (higher better)','N any-offset error (lower better)','V all-offset sensitivity (higher better)'])
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle('Fixed 3-seed mean; paired patient-cluster bootstrap (2,000 draws)\n95% descriptive intervals; no multiplicity correction',fontsize=13)
    fig.savefig(ROOT/'independent_primary_intervals.png',dpi=180)
    fig.savefig(ROOT/'independent_primary_intervals.svg');plt.close(fig)
    writej(ROOT/'figure_provenance.json',dict(created=now(),source_sha256=sha(ROOT/'independent_comparison.json'),script_sha256=sha(Path(__file__)),all_models=MODELS,all_seeds=[42,43,44],threshold='pV>0.50',endpoints=[x[0] for x in PANELS],figures={n:sha(ROOT/n) for n in ['independent_four_grid.png','independent_four_grid.svg','independent_primary_intervals.png','independent_primary_intervals.svg']}))
    print('ALL-SEED FOUR-GRID AND PRIMARY INTERVAL FIGURES SAVED',flush=True)

if __name__=='__main__':main()
