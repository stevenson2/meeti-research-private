#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render final benchmark v1.5 summary charts + tables as PNG images (v2)."""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import zlib  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path('/home/devcontainers/meeti')
LB = ROOT / 'runs/benchmark_v1/leaderboard.json'
WS = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261')
DESK = Path('/mnt/c/Users/cai/OneDrive/Desktop')

W = {'ptb_val': 0.35, 'ptb_test': 0.40, 'meeti_val': 0.25}
CANDIDATE = '#7b5ea7'
LOCKED = '#1a1a1a'
TEACHER = '#85b8d0'
LIT = '#b0413e'
PALETTE = ['#3fa34d', '#2f6fdb', '#f28e2b', '#4c9f9f', '#d45d9c',
           '#9acd32', '#b0705c', '#8e44ad']
DISPLAY = {
    'r9_form_ptbbal': 'R9 form+ptbbal\n(6 seeds)',
    'r9_form': 'R9 form', 'r9_norm': 'R9 norm',
    'locked_dual_head_v3_seeded_ptb': 'LOCKED\ndual-head v3',
    'legacy_repo_exp7c_v4': 'legacy exp7c_v4\n(leak-flagged)',
    'r8a_raw': 'R8a raw', 'r9_norm_w5': 'R9 norm + w5',
    'legacy_repo_baseline_v5': 'legacy\nbaseline v5',
    'legacy_repo_baseline_v4': 'legacy\nbaseline v4',
    'legacy_repo_baseline_v3': 'legacy\nbaseline v3',
    'legacy_repo_v6a5': 'legacy\nv6a5',
    'legacy_repo_capacity_w2_seed42': 'legacy\ncapacity_w2',
    'legacy_repo_distill_pilot_seed42': 'legacy\ndistill pilot',
    'legacy_repo_distill_pilot_v2_seed42': 'legacy\ndistill v2',
    'fm_teacher_lead2fill': 'FM teacher\n(lead2 fill)',
    'legacy_repo_original_cnn_v2': 'legacy\ncnn_v2 (orig)',
    'legacy_repo_archived_cnn_v2': 'legacy\ncnn_v2 (arch)',
    'legacy_repo_resnet_medium': 'legacy\nresnet-M',
    'legacy_repo_resnet_small': 'legacy\nresnet-S',
    'legacy_repo_cnn_m': 'legacy\ncnn-M',
    'legacy_new_m1_beat': 'legacy\nM1 beat head',
    'legacy_new_v3_noptb_beat': 'legacy v3 beat\n(noPTB)',
    'legacy_new_v3_ptb_beat': 'v3 beat head\n(+PTB, locked run)',
    'legacy_ensemble': 'Ensemble\n(soft-vote x3)',
    'lit_ecgfm-linear_lead2fill_mimichead': 'LIT ECG-FM linear probe\n'
                                            '(lead2fill, mimichead)',
    'lit_resnet_lead2fill_3beat': 'LIT Strodthoff resnet50-1D\n'
                                  '(PTB-XL ckpt, lead2fill)',
    'lit_xresnet_lead2fill_3beat': 'LIT Strodthoff xresnet1d50\n'
                                   '(PTB-XL ckpt, lead2fill)',
    'lit_lenet_lead2fill_3beat': 'LIT Strodthoff lab LeNet\n'
                                 '(PTB-XL ckpt, lead2fill)',
}
TNAME = {
    'locked_dual_head_v3_seeded_ptb': 'locked dual-head v3',
    'r9_form_ptbbal': 'r9_form_ptbbal [CANDIDATE]',
    'lit_ecgfm-linear_lead2fill_mimichead': 'LIT ECG-FM linear probe',
    'lit_resnet_lead2fill_3beat': 'LIT Strodthoff resnet50-1D',
    'lit_xresnet_lead2fill_3beat': 'LIT Strodthoff xresnet1d50',
    'lit_lenet_lead2fill_3beat': 'LIT Strodthoff lab LeNet',
    'fm_teacher_lead2fill': 'FM teacher (lead2fill)',
}


def s_score(row):
    s = 0.0
    for k, w in W.items():
        v = row.get(k)
        v = v['mean'] if isinstance(v, dict) else v
        if v is None:
            return None
        s += w * (v - 0.5) / 0.5
    return 100 * s


def s_beat_score(row):
    v, t = row.get('beat_mi_val'), row.get('beat_mi_test')
    v = v['mean'] if isinstance(v, dict) else v
    t = t['mean'] if isinstance(t, dict) else t
    if v is None or t is None:
        return None
    return 100 * ((v - 0.5) / 0.5 + (t - 0.5) / 0.5) / 2


def color_for(name):
    if name == 'r9_form_ptbbal':
        return CANDIDATE
    if name.startswith('lit_'):
        return LIT
    if 'locked' in name:
        return LOCKED
    if 'teacher' in name:
        return TEACHER
    return PALETTE[zlib.crc32(name.encode()) % len(PALETTE)]


def bar_chart(rows, title, sub, out):
    names = [DISPLAY.get(n, n) for _, n in rows]
    vals = [s for s, _ in rows]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(19, 7), dpi=150)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    x = list(range(n))
    ymin = min(-18.0, min(vals) * 1.5)
    ymax = max(vals) * 1.18
    ax.set_xlim(-1.05, n - 0.3)
    ax.set_ylim(ymin, ymax)
    span = ymax - ymin
    for gy in range(int(ymin) - 10, int(ymax) + 10, 10):
        ax.axhline(gy, color='#e8e8ec', lw=1.1, zorder=0)
    ax.axhline(0, color='#cccccc', lw=1.0, zorder=1)
    ax_pos = ax.get_position()
    x_unit_in = (ax_pos.width * fig.get_figwidth()) / (n + 0.4)
    y_unit_in = (ax_pos.height * fig.get_figheight()) / span
    aspect = y_unit_in / x_unit_in
    for xi, v in enumerate(vals):
        c = color_for(rows[xi][1])
        p = FancyBboxPatch(
            (xi - 0.36, min(0, v)), 0.72, abs(v) if abs(v) > 0.5 else 0.5,
            boxstyle='round,pad=0,rounding_size=0.3',
            mutation_aspect=aspect, fc=c, ec='none', zorder=3)
        ax.add_patch(p)
        if v >= 12:
            ax.text(xi, v - 2.2, f'{v:.0f}', ha='center', va='top',
                    color='white', fontsize=12, fontweight='bold', zorder=4)
        elif v >= 0:
            ax.text(xi, v + 1.6, f'{v:.0f}', ha='center', va='bottom',
                    color='#333333', fontsize=11, fontweight='bold', zorder=4)
        else:
            ax.text(xi, v - 1.6, f'{v:.0f}', ha='center', va='top',
                    color=LIT, fontsize=11, fontweight='bold', zorder=4)
    locked_s = next((s for s, nm in rows if 'locked' in nm), None)
    if locked_s is not None:
        ax.axhline(locked_s, color='#1a1a1a', lw=1.2, ls=(0, (4, 3)),
                   alpha=0.55, zorder=2)
        ax.text(n - 0.35, locked_s + 1.2, 'LOCKED reference', ha='right',
                fontsize=9, color='#1a1a1a', alpha=0.7)
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.set_yticks([])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha='right', fontsize=9.5,
                       color='#333333')
    ax.tick_params(axis='x', length=0, pad=6)
    ax.set_title(title, loc='left', fontsize=15, fontweight='bold',
                 color='#1a1a1a', pad=26)
    ax.text(0.0, 1.035, sub, transform=ax.transAxes, fontsize=8.8,
            color='#777777')
    ax.text(0.995, 1.045, '▲ ESP32-ECG Bench v1.5', transform=ax.transAxes,
            ha='right', fontsize=11, color='#5a5a5a', fontweight='bold')
    fig.subplots_adjust(left=0.045, right=0.995, top=0.82, bottom=0.25)
    fig.savefig(out, facecolor='white')
    plt.close(fig)
    print('saved', out)


def table_image(header, rows, out, width=19.0, row_h=0.42, fsize=9,
                title=None):
    n = len(rows) + 1
    fig_h = max(3.0, n * row_h + 1.6)
    fig, ax = plt.subplots(figsize=(width, fig_h), dpi=150)
    ax.axis('off')
    y = 1.0
    xs = np.linspace(0.0, 1.0, len(header) + 1)
    if title:
        ax.text(0.0, 1.06, title, transform=ax.transAxes, fontsize=13,
                fontweight='bold', color='#1a1a1a')
    ax.add_patch(plt.Rectangle((0, y - row_h / n * 2), 1, row_h / n * 2,
                               fc='#2c3e50', ec='none',
                               transform=ax.transAxes))
    for j, h in enumerate(header):
        ax.text((xs[j] + xs[j + 1]) / 2, y - row_h / n, str(h),
                ha='center', va='center', fontsize=fsize, color='white',
                fontweight='bold', transform=ax.transAxes)
    y -= row_h / n * 2
    for i, row in enumerate(rows):
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0, y - row_h / n), 1, row_h / n,
                                       fc='#f2f4f7', ec='none',
                                       transform=ax.transAxes))
        is_lit = 'LIT' in str(row[0])
        for j, cell in enumerate(row):
            bold = j == 1 and ('CANDIDATE' in str(cell)
                               or 'locked' in str(cell))
            color = LIT if is_lit else '#1a1a1a'
            last = j == len(header) - 1
            ax.text((xs[j] + xs[j + 1]) / 2, y - row_h / (2 * n),
                    str(cell), ha='center', va='center', fontsize=fsize,
                    color=color,
                    fontweight='bold' if bold or (last and cell != '—')
                    else 'normal', transform=ax.transAxes)
        y -= row_h / n
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(out, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print('saved', out)


def om_of(m):
    if 'aggregate' not in m:
        return m.get('op_metrics')
    oms = [s.get('op_metrics') for s in m.get('per_seed', [])
           if s.get('op_metrics')]
    if not oms:
        return None
    out = {}
    for k in oms[0]:
        v0 = oms[0][k]
        if not isinstance(v0, dict):
            vals = [o[k] for o in oms if k in o]
            out[k] = float(np.mean(vals)) if k.startswith('theta') else v0
            continue
        out[k] = {}
        for th in v0:
            st = v0[th]
            if not isinstance(st, dict):
                out[k][th] = st
                continue
            out[k][th] = {}
            for kk, vv in st.items():
                if kk == 'tp_fp_fn_tn':
                    continue
                try:
                    out[k][th][kk] = float(np.mean(
                        [o[k][th][kk] for o in oms
                         if th in o.get(k, {})]))
                except (KeyError, TypeError):
                    out[k][th][kk] = vv
    return out


def spec_of(m, dom, thk):
    if 'aggregate' not in m:
        c = (m.get('op_metrics', {}).get(dom, {})
             .get(thk, {}).get('tp_fp_fn_tn'))
        if c:
            return c[3] / max(c[3] + c[1], 1)
        return None
    vals = [s['op_metrics'][dom][thk] for s in m.get('per_seed', [])
            if s.get('op_metrics') and thk in s['op_metrics'].get(dom, {})]
    sp = [o['tp_fp_fn_tn'][3] / max(o['tp_fp_fn_tn'][3]
                                    + o['tp_fp_fn_tn'][1], 1)
          for o in vals if o.get('tp_fp_fn_tn')]
    return float(np.mean(sp)) if sp else None


def g(om, dom, th, key):
    try:
        return om[dom][th][key]
    except (KeyError, TypeError):
        return None


def main():
    board = json.load(open(LB))
    rows = []
    for name, m in board['models'].items():
        s = s_score(m['aggregate'] if 'aggregate' in m else m)
        if s is not None:
            rows.append((s, name))
    rows.sort(reverse=True)
    order = {n: i for i, (s, n) in enumerate(rows)}
    WS.mkdir(parents=True, exist_ok=True)

    bar_chart(
        rows,
        'ESP32-ECG Unified Benchmark v1.5 - Composite Score S (record level)',
        'S = 100·Σ w·(AUC−0.5)/0.5 | w: ptb_val 0.35 · ptb_test 0.40 · '
        'meeti_val 0.25 | 50 = chance | lit_ rows = frozen literature '
        'checkpoints (amplitude-matched, sec 11.2 v1.3.1)',
        WS / 'BENCHMARK_v1.5_S_ranking.png')

    rows_b = [(s_beat_score(m['aggregate'] if 'aggregate' in m else m), n)
              for n, m in board['models'].items()
              if s_beat_score(m['aggregate'] if 'aggregate' in m else m)
              is not None]
    rows_b.sort(reverse=True)
    bar_chart(
        rows_b,
        'ESP32-ECG Unified Benchmark v1.5 - Beat-level Sub-score S_beat',
        'S_beat = 100·[(AUC_mi_val−0.5)/0.5 + (AUC_mi_test−0.5)/0.5]/2 on '
        'the leak-free MIT+INCART patient-level split (beat-capable '
        'families only)',
        WS / 'BENCHMARK_v1.5_S_beat_ranking.png')

    # op table (include lit rows with em-dash op cells)
    op_rows = []
    for name in sorted([n for n in board['models']
                        if (om_of(board['models'][n]) or
                            n.startswith('lit_'))],
                       key=lambda n: order.get(n, 999)):
        m = board['models'][name]
        om = om_of(m)
        s = s_score(m['aggregate'] if 'aggregate' in m else m)
        if om and 'ptb_test' in om:
            r5 = g(om, 'ptb_test', '0.500', 'recall')
            sp5 = spec_of(m, 'ptb_test', '0.500')
            f5 = g(om, 'ptb_test', '0.500', 'f1')
            b5 = g(om, 'ptb_test', '0.500', 'bacc')
            ap = om['ptb_test'].get('avg_precision')
            th = om.get('theta_train_youden_ptb')
            rtr = g(om, 'ptb_test', 'theta_tr', 'recall')
            sptr = spec_of(m, 'ptb_test', 'theta_tr')
            cells = [f'{r5:.3f}' if r5 is not None else '—',
                     f'{sp5:.3f}' if sp5 is not None else '—',
                     f'{f5:.3f}' if f5 is not None else '—',
                     f'{b5:.3f}' if b5 is not None else '—',
                     f'{ap:.3f}' if isinstance(ap, float) else '—',
                     f'{th:.2f}' if isinstance(th, float) else '—',
                     f'{rtr:.3f}' if rtr is not None else '—',
                     f'{sptr:.3f}' if sptr is not None else '—']
        else:
            cells = ['—'] * 8
        nm = TNAME.get(name, name.replace('legacy_', 'L. ')
                       .replace('_seed42', ''))
        op_rows.append([f'{len(op_rows) + 1}', nm,
                        f'{s:.1f}' if s is not None else '—', *cells])
    table_image(
        ['#', 'Model', 'S', 'R@.5', 'spec@.5', 'F1@.5', 'bAcc@.5', 'AP',
         'theta_tr', 'R@tr', 'spec@tr'],
        op_rows,
        WS / 'BENCHMARK_v1.5_op_metrics_table.png',
        width=19.0, row_h=0.5, fsize=8.6,
        title='Operating-point layer v1.5 - PTB test record level '
              '(theta=0.5 fixed + theta_tr = PTB TRAIN-side Youden; '
              'negative class n=20 -> spec/bAcc carry 5% per record)')

    # detail table
    det_rows = []
    for name in sorted([n for n, m in board['models'].items()
                        if s_score(m['aggregate'] if 'aggregate' in m
                                   else m) is not None],
                       key=lambda n: order.get(n, 999)):
        m = board['models'][name]
        row = m['aggregate'] if 'aggregate' in m else m
        cells = [TNAME.get(name, name.replace('legacy_', 'L. ')
                           .replace('_seed42', ''))]
        for k in ['meeti_val', 'ptb_val', 'ptb_test', 'beat_val',
                  'beat_mi_val', 'beat_mi_test']:
            v = row.get(k)
            if isinstance(v, dict):
                cells.append(f"{v['mean']:.4f}±{v['sd']:.2f}")
            elif isinstance(v, float):
                cells.append(f'{v:.4f}')
            else:
                cells.append('—')
        s = s_score(row)
        sb = s_beat_score(row)
        cells.append(f'{s:.1f}' if s is not None else '—')
        cells.append(f'{sb:.1f}' if sb is not None else '—')
        det_rows.append(cells)
    table_image(
        ['Model', 'MEETI val', 'PTB val', 'PTB test', 'beat val',
         'MI beat val', 'MI beat test', 'S', 'S_beat'],
        det_rows,
        WS / 'BENCHMARK_v1.5_detail_table.png',
        width=21.0, row_h=0.5, fsize=9.2,
        title='BENCHMARK v1.5 full detail - 28 models (record score '
              'top2mean; multi-seed = mean±sd; LIT rows = literature '
              'checkpoints, amplitude-matched caliber)')

    for f in ['BENCHMARK_v1.5_S_ranking.png',
              'BENCHMARK_v1.5_S_beat_ranking.png',
              'BENCHMARK_v1.5_op_metrics_table.png',
              'BENCHMARK_v1.5_detail_table.png']:
        try:
            shutil.copyfile(WS / f, DESK / f)
        except Exception as e:
            print('desktop copy skip', f, e)
    print('ALL RENDERED')


if __name__ == '__main__':
    main()
