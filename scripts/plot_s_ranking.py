#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Composite-score S ranking chart (Artificial-Analysis style).

Reads runs/benchmark_v1/leaderboard.json, recomputes S per model with the
frozen formula, and renders rounded vertical bars with in-bar scores and
rotated model names.  Output: runs/benchmark_v1/S_ranking.png (+ copy on
the Windows desktop for quick viewing).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path('/home/devcontainers/meeti')
LB = ROOT / 'runs/benchmark_v1/leaderboard.json'
OUT = ROOT / 'runs/benchmark_v1/S_ranking.png'
DESKTOP = Path('/mnt/c/Users/cai/OneDrive/Desktop/S_ranking_benchmark.png')

W = {'ptb_val': 0.35, 'ptb_test': 0.40, 'meeti_val': 0.25}

DISPLAY = {
    'r9_form_ptbbal': 'R9 form+ptbbal\n(6 seeds)',
    'r9_form': 'R9 form',
    'r9_norm': 'R9 norm',
    'locked_dual_head_v3_seeded_ptb': 'LOCKED\ndual-head v3',
    'legacy_repo_exp7c_v4': 'legacy exp7c_v4\n(leak-flagged)',
    'r8a_raw': 'R8a raw',
    'r9_norm_w5': 'R9 norm + w5',
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

CANDIDATE = '#7b5ea7'      # purple - our #1
LOCKED = '#1a1a1a'         # black  - incumbent
TEACHER = '#85b8d0'        # light blue - ceiling probe
LIT = '#b0413e'            # brick red - literature reference rows (sec 11)
PALETTE = ['#3fa34d', '#2f6fdb', '#f28e2b', '#4c9f9f', '#d45d9c',
           '#9acd32', '#b0705c', '#8e44ad']


def s_score(row):
    s = 0.0
    for k, w in W.items():
        v = row.get(k)
        v = v['mean'] if isinstance(v, dict) else v
        if v is None:
            return None
        s += w * (v - 0.5) / 0.5
    return 100 * s


def main():
    board = json.load(open(LB))
    rows = []
    for name, m in board['models'].items():
        row = m['aggregate'] if 'aggregate' in m else m
        s = s_score(row)
        if s is not None:
            rows.append((s, name))
    rows.sort(reverse=True)
    names = [DISPLAY.get(n, n) for _, n in rows]
    vals = [s for s, _ in rows]
    n = len(rows)

    def color_for(name):
        if name == 'r9_form_ptbbal':
            return CANDIDATE
        if name.startswith('lit_'):
            return LIT
        if 'locked' in name:
            return LOCKED
        if 'teacher' in name:
            return TEACHER
        return PALETTE[hash(name) % len(PALETTE)]

    fig, ax = plt.subplots(figsize=(19, 7), dpi=150)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    x = list(range(n))
    ax.set_xlim(-1.05, n - 0.3)
    ymax = max(vals) * 1.18
    ax.set_ylim(0, ymax)

    # light horizontal gridlines
    for gy in range(0, int(ymax) + 1, 10):
        if gy:
            ax.axhline(gy, color='#e8e8ec', lw=1.1, zorder=0)

    # rounded bars
    ax_pos = ax.get_position()
    x_unit_in = (ax_pos.width * fig.get_figwidth()) / (n + 0.4)
    y_unit_in = (ax_pos.height * fig.get_figheight()) / ymax
    aspect = y_unit_in / x_unit_in
    r = 0.34
    for xi, v in enumerate(vals):
        c = color_for(rows[xi][1])
        p = FancyBboxPatch(
            (xi - 0.36, 0), 0.72, v,
            boxstyle=f'round,pad=0,rounding_size={r}',
            mutation_aspect=aspect, fc=c, ec='none', zorder=3)
        ax.add_patch(p)
        ax.text(xi, v - 3.2, f'{v:.0f}', ha='center', va='top',
                color='white', fontsize=12.5, fontweight='bold', zorder=4)

    # locked reference line
    locked_s = next(s for s, nm in rows if 'locked' in nm)
    ax.axhline(locked_s, color='#1a1a1a', lw=1.2, ls=(0, (4, 3)),
               alpha=0.55, zorder=2)
    ax.text(n - 0.35, locked_s + 0.8, 'LOCKED reference', ha='right',
            fontsize=9, color='#1a1a1a', alpha=0.7)

    # axes cosmetics
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.set_yticks([])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha='right', fontsize=9.5,
                       color='#333333')
    ax.tick_params(axis='x', length=0, pad=6)

    # titles + brand
    ax.set_title('ESP32-ECG Unified Benchmark v1.3 - Composite Score S '
                 '(record level)', loc='left', fontsize=15,
                 fontweight='bold', color='#1a1a1a', pad=26)
    ax.text(0.0, 1.035, 'S = 100·Σ w·(AUC−0.5)/0.5   |   w: ptb_val 0.35 · '
            'ptb_test 0.40 · meeti_val 0.25   |   50 = chance, 100 = '
            'perfect   |   training models = multi-seed mean; lit_ rows = '
            'frozen literature checkpoints (sec 11)',
            transform=ax.transAxes, fontsize=8.8, color='#777777')
    ax.text(0.995, 1.045, '▲ ESP32-ECG Bench', transform=ax.transAxes,
            ha='right', fontsize=11, color='#5a5a5a', fontweight='bold')

    fig.subplots_adjust(left=0.045, right=0.995, top=0.82, bottom=0.25)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor='white')
    try:
        import shutil
        shutil.copyfile(OUT, DESKTOP)
        print('copied ->', DESKTOP)
    except Exception as e:
        print('desktop copy skipped:', e)
    print('saved ->', OUT)


if __name__ == '__main__':
    sys.exit(main())
