#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R10: merge lit_row_*.json into the existing leaderboard without
re-evaluating frozen rows (eval-only merge; ecg_bench.write_md reused)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/devcontainers/meeti/scripts')

ROOT = Path('/home/devcontainers/meeti')
OUT = ROOT / 'runs/benchmark_v1'


def main():
    board = json.load(open(OUT / 'leaderboard.json'))
    board['spec'] = 'BENCHMARK_V1 v1.3 (runs/BENCHMARK_V1.md)'
    for row_file in sorted(OUT.glob('lit_row_*.json')):
        row = json.load(open(row_file))
        name = row_file.stem[len('lit_row_'):]
        board['models'][name] = row
        board['record_score'][name] = row.get('record_score', 'top2mean')
        print('merged', name, 'S =', row.get('S'), flush=True)
    with open(OUT / 'leaderboard.json', 'w') as f:
        json.dump(board, f, indent=2, ensure_ascii=False)

    import ecg_bench
    ecg_bench.OUT = OUT
    ecg_bench.write_md(board)
    print('leaderboard + LEADERBOARD.md updated', flush=True)


if __name__ == '__main__':
    main()
