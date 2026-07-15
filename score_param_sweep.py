#!/usr/bin/env python3
"""
score_param_sweep.py
────────────────────
Reads all  eval/param_sweep/<label>.json  files produced by run_map_sweep.sh,
computes per-combo metrics relative to the BRRT and BG-BRRT baselines, then
prints a ranked leaderboard of the top parameter combinations.

Metrics (per trial, averaged across all trials):
  · nodes       — total node count (lower is better)
  · iters       — total iteration count (lower is better)
  · search_time — wall-clock search time in seconds (lower is better)
  · path_len    — path length (lower is better)
  · success     — success rate (higher is better)

Each metric is compared against both BRRT and BG-BRRT.
A combo "wins" on a metric if Case3 is strictly better than BOTH baselines.

Composite score = number of metrics where Case3 beats both baselines.
Tie-break: weighted average of normalised relative improvements.

Usage:
  python3 score_param_sweep.py --sweep-dir eval/param_sweep [--top 10]
"""

import argparse
import json
import os
import glob
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── Metric weights for the composite tie-break score ─────────────────────────
METRIC_WEIGHTS = {
    "search_time": 0.35,
    "nodes":       0.25,
    "iters":       0.20,
    "path_len":    0.10,
    "success":     0.10,
}

LOWER_IS_BETTER = {"nodes", "iters", "search_time", "path_len"}
HIGHER_IS_BETTER = {"success"}


@dataclass
class ComboResult:
    label: str
    params: Dict
    # Averages across all trials for each algorithm
    case3: Dict[str, float] = field(default_factory=dict)
    brrt:  Dict[str, float] = field(default_factory=dict)
    bg:    Dict[str, float] = field(default_factory=dict)
    num_trials: int = 0
    # Derived
    wins: int = 0
    composite: float = 0.0


def extract_metrics(run_list: List[Dict]) -> Tuple[Dict, Dict, Dict]:
    """Return (case3_metrics, brrt_metrics, bg_metrics) averaged over run_list."""
    def empty(): return {"nodes": [], "iters": [], "search_time": [], "path_len": [], "success": []}
    c3, br, bg = empty(), empty(), empty()

    for run in run_list:
        algos = run.get("algorithms", {})
        for algo_key, store in (("BRRT_Case3", c3), ("BRRT", br), ("BG_BRRT", bg)):
            a = algos.get(algo_key)
            if a is None:
                continue
            store["nodes"].append(a.get("node_count", float("nan")))
            store["iters"].append(a.get("num_iterations", float("nan")))
            store["search_time"].append(a.get("search_time", float("nan")))
            store["path_len"].append(a.get("path_length", float("nan")))
            store["success"].append(1.0 if a.get("success", False) else 0.0)

    def avg(lst):
        valid = [x for x in lst if x == x]  # drop NaN
        return statistics.mean(valid) if valid else float("nan")

    def summarise(d):
        return {k: avg(v) for k, v in d.items()}

    return summarise(c3), summarise(br), summarise(bg)


def relative_improvement(case3_val: float, baseline_val: float, lower_better: bool) -> float:
    """Return fractional improvement of case3 over baseline (positive = better)."""
    if baseline_val == 0 or baseline_val != baseline_val:
        return 0.0
    if lower_better:
        return (baseline_val - case3_val) / abs(baseline_val)
    else:
        return (case3_val - baseline_val) / abs(baseline_val)


def score_combo(cr: ComboResult) -> None:
    """Populate cr.wins and cr.composite in-place."""
    wins = 0
    weighted_sum = 0.0
    weight_total = 0.0

    for metric, weight in METRIC_WEIGHTS.items():
        c3_val  = cr.case3.get(metric, float("nan"))
        br_val  = cr.brrt.get(metric, float("nan"))
        bg_val  = cr.bg.get(metric, float("nan"))

        if any(v != v for v in (c3_val, br_val, bg_val)):
            continue  # skip metrics with NaN data

        lower_better = metric in LOWER_IS_BETTER

        beats_brrt = (c3_val < br_val) if lower_better else (c3_val > br_val)
        beats_bg   = (c3_val < bg_val) if lower_better else (c3_val > bg_val)

        if beats_brrt and beats_bg:
            wins += 1

        imp_brrt = relative_improvement(c3_val, br_val, lower_better)
        imp_bg   = relative_improvement(c3_val, bg_val, lower_better)
        avg_imp  = (imp_brrt + imp_bg) / 2.0

        weighted_sum  += weight * avg_imp
        weight_total  += weight

    cr.wins      = wins
    cr.composite = weighted_sum / weight_total if weight_total > 0 else 0.0


def load_combo(path: str) -> Optional[ComboResult]:
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [warn] Cannot read {path}: {e}")
        return None

    label  = data.get("label", os.path.basename(path).replace(".json", ""))
    params = data.get("params", {})
    runs   = data.get("results", [])

    if not runs:
        return None

    c3, br, bg = extract_metrics(runs)

    cr = ComboResult(label=label, params=params,
                     case3=c3, brrt=br, bg=bg, num_trials=len(runs))
    score_combo(cr)
    return cr


def format_delta(case3_val: float, baseline_val: float, lower_better: bool) -> str:
    if case3_val != case3_val or baseline_val != baseline_val:
        return "  N/A "
    imp = relative_improvement(case3_val, baseline_val, lower_better)
    sign = "↓" if (lower_better and case3_val < baseline_val) else \
           "↑" if (not lower_better and case3_val > baseline_val) else \
           ("↑" if imp > 0 else "↓")
    return f"{sign}{abs(imp)*100:5.1f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dir", default="eval/param_sweep",
                    help="Directory containing <label>.json combo files")
    ap.add_argument("--top", type=int, default=10,
                    help="Number of top combos to print")
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.sweep_dir, "*.json"))
    if not files:
        print(f"No JSON files found in {args.sweep_dir}")
        return

    combos: List[ComboResult] = []
    for f in sorted(files):
        cr = load_combo(f)
        if cr is not None:
            combos.append(cr)

    if not combos:
        print("No valid combo results found.")
        return

    # Sort: most wins first, then highest composite score
    combos.sort(key=lambda c: (-c.wins, -c.composite))

    METRICS_ORDER = ["nodes", "iters", "search_time", "path_len", "success"]
    LOWER_BETTER  = LOWER_IS_BETTER

    print("\n" + "=" * 90)
    print(f"  PARAM SWEEP LEADERBOARD  (top {args.top} of {len(combos)} combos)")
    print("=" * 90)
    print(f"  {'Rank':<5} {'Label':<42} {'Wins':>4} {'Score':>7}  Trials")
    print("-" * 90)

    top_n = combos[:args.top]
    for rank, cr in enumerate(top_n, 1):
        print(f"  {rank:<5} {cr.label:<42} {cr.wins:>4}/{len(METRICS_ORDER)}  "
              f"{cr.composite:>+6.3f}   {cr.num_trials}")

    print()

    # Detailed breakdown for the best combo
    best = combos[0]
    print(f"  BEST COMBO: {best.label}")
    print(f"  Parameters:")
    for k, v in best.params.items():
        print(f"    {k:32s} = {v}")
    print()
    print(f"  Metric breakdown (Case3 vs BRRT / BG-BRRT):")
    print(f"  {'Metric':<14} {'Case3':>10} {'BRRT':>10} {'BG-BRRT':>10}   "
          f"{'vs BRRT':>9} {'vs BG-BRRT':>11}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10}   {'-'*9} {'-'*11}")
    for m in METRICS_ORDER:
        c3 = best.case3.get(m, float("nan"))
        br = best.brrt.get(m, float("nan"))
        bg = best.bg.get(m, float("nan"))
        lb = m in LOWER_BETTER
        print(f"  {m:<14} {c3:>10.4f} {br:>10.4f} {bg:>10.4f}   "
              f"{format_delta(c3, br, lb):>9} {format_delta(c3, bg, lb):>11}")

    print()

    # Print top-10 params in launch-override format for easy re-use
    print("  ── roslaunch override for best combo ──────────────────────────")
    p = best.params
    print(f"  roslaunch path_finder test_planners.launch \\")
    print(f"    brrt_eta_stagn:={p.get('eta_stagn','')} \\")
    print(f"    brrt_cylinder_radius:={p.get('cylinder_radius_factor','')} \\")
    print(f"    brrt_epsilon_h:={p.get('epsilon_h','')} \\")
    print(f"    brrt_alpha:={p.get('brrt_alpha','')} \\")
    print(f"    ...")
    print("=" * 90)
    print()

    # Write a summary CSV for further analysis
    csv_path = os.path.join(args.sweep_dir, "leaderboard.csv")
    with open(csv_path, "w") as f:
        f.write("rank,label,wins,composite,num_trials,"
                "eta_stagn,cylinder_radius_factor,epsilon_h,brrt_alpha,"
                "c3_nodes,c3_iters,c3_time,c3_pathlen,c3_success,"
                "brrt_nodes,brrt_iters,brrt_time,brrt_pathlen,brrt_success,"
                "bg_nodes,bg_iters,bg_time,bg_pathlen,bg_success\n")
        for rank, cr in enumerate(combos, 1):
            p = cr.params
            row = [
                rank, cr.label, cr.wins, f"{cr.composite:.4f}", cr.num_trials,
                p.get("eta_stagn",""), p.get("cylinder_radius_factor",""),
                p.get("epsilon_h",""), p.get("brrt_alpha",""),
                cr.case3.get("nodes",""), cr.case3.get("iters",""),
                cr.case3.get("search_time",""), cr.case3.get("path_len",""),
                cr.case3.get("success",""),
                cr.brrt.get("nodes",""), cr.brrt.get("iters",""),
                cr.brrt.get("search_time",""), cr.brrt.get("path_len",""),
                cr.brrt.get("success",""),
                cr.bg.get("nodes",""), cr.bg.get("iters",""),
                cr.bg.get("search_time",""), cr.bg.get("path_len",""),
                cr.bg.get("success",""),
            ]
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"  Full CSV leaderboard: {csv_path}")


if __name__ == "__main__":
    main()
