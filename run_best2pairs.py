"""
run_best2pairs.py
-----------------
Find the best 2-pair merge configuration for each chosen task by running
run_exp_ft.py with every possible combination of two non-consecutive layer
index pairs chosen from {0, 2, 4, 6, 8, 10}.

Edit the top section to select tasks, sv_portion, and optional flags.
"""

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

# ── User-configurable section ────────────────────────────────────────────────

TASKS = ["MNIST"]   # tasks to evaluate

SV_PORTION  = 2       # singular-value denominator passed to run_exp_ft.py
WHITENING   = True    # --whitening flag
LOOP        = False   # --loop flag

RESULTS_BASE = Path("results")  # base results directory

# ── Pair generation ──────────────────────────────────────────────────────────

# Possible first-layer indices for a merge pair:
#   index i  →  merges layers (i, i+1)
# Valid range is 0..10 (model has 12 layers, so last merge is (10,11)).
# Pairs where j == i+1 are excluded because run_exp_ft.py rejects groups
# that contain consecutive integers.
LAYER_INDICES = list(range(11))  # 0 to 10 inclusive

# All combinations of exactly 2 distinct indices, excluding consecutive ones
ALL_2_COMBOS = [
    list(combo)
    for combo in itertools.combinations(LAYER_INDICES, 2)
    if combo[1] != combo[0] + 1   # drop (i, i+1) pairs
]

print(f"Total pair combinations to evaluate: {len(ALL_2_COMBOS)}")
print("Combinations:", ALL_2_COMBOS)
print()

# ── Runner ───────────────────────────────────────────────────────────────────

for task in TASKS:
    out_dir = RESULTS_BASE / task / "best_2_pairs"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{task}_best_2_pairs.json"

    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"Results will be saved to: {results_path}")
    print(f"{'='*60}")

    # Encode all combinations as the list_of_pairs argument (JSON string)
    list_of_pairs_arg = json.dumps(ALL_2_COMBOS)

    cmd = [
        sys.executable, "run_exp_ft.py",
        task,
        str(SV_PORTION),
        list_of_pairs_arg,
        "--results_path", str(results_path),
        "--whitening", str(WHITENING),
        "--loop",      str(LOOP),
    ]

    print("Running:", " ".join(cmd))
    print()

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"[WARNING] run_exp_ft.py exited with code {result.returncode} for task '{task}'")
    else:
        print(f"\n[OK] Task '{task}' completed. Results saved to {results_path}")

    print()

# ── Summary ──────────────────────────────────────────────────────────────────

print("=" * 60)
print("All tasks finished. Summary of best pairs per task:")
print("=" * 60)

for task in TASKS:
    results_path = RESULTS_BASE / task / "best_2_pairs" / f"{task}_best_2_pairs.json"
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        best_key = max(data, key=data.get)
        best_acc = data[best_key]
        print(f"  {task:12s}: best pair = {best_key}  (acc = {best_acc:.5f})")
    else:
        print(f"  {task:12s}: no results file found")
