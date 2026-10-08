"""Regenerate r9 arm summaries from per-seed result.json files."""
import json
import sys

import numpy as np

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from r9_train import summarize  # noqa: E402

tag = sys.argv[1]
seeds = [42, 43, 44]
results = [json.load(open(
    f'/home/devcontainers/meeti/runs/{tag}_seed{s}/result.json'))
    for s in seeds]
summarize(tag, seeds, results)
