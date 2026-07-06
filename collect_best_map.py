#!/usr/bin/env python3
"""
collect_best_map.py
====================
Post-processes the output of run_map_sweep.sh.

For each per-map JSON produced by the ROS node (100 start/goal pairs),
computes per-map aggregate statistics for BRRT, BG_BRRT, and BRRT_Case3,
then identifies the *best* map where BRRT_Case3 outperforms both baselines
across all three metrics:
  • mean path length  (lower is better)
  • mean node count   (lower is better)
  • mean iterations   (lower is better)

Score = number of metrics (0–6) where BRRT_Case3 beats a baseline.
A perfect score is 6 (wins all 3 metrics vs both BRRT and BG_BRRT).

Usage
-----
  python3 collect_best_map.py --sweep-dir eval/map_sweep/blob_map_obs20_sz500
  python3 collect_best_map.py --sweep-dir eval/map_sweep/blob_map_obs20_sz500 \
                              --output best_map_summary.json

The script also prints a ranked table of all maps to stdout.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Algorithm keys exactly as they appear in the result JSON files
ALGO_BRRT   = "BRRT"
ALGO_BG     = "BG_BRRT"
ALGO_CASE3  = "BRRT_Case3"

BASELINES   = [ALGO_BRRT, ALGO_BG]
ALGOS_ALL   = [ALGO_BRRT, ALGO_BG, ALGO_CASE3]

# Maximum possible score = 3 metrics × 2 baselines
MAX_SCORE   = 6


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_map_stats(json_path: str) -> Optional[dict]:
    """
    Load one per-map JSON and return a dict with aggregate stats for
    BRRT, BG_BRRT, and BRRT_Case3.  Returns None if the file is unreadable
    or lacks results.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [WARN] Cannot read {json_path}: {e}", file=sys.stderr)
        return None

    results = data.get("results", [])
    if not results:
        print(f"  [WARN] No results in {json_path}", file=sys.stderr)
        return None

    stats: dict[str, dict] = {}

    for algo in ALGOS_ALL:
        lengths, nodes, iters = [], [], []
        total, success = 0, 0
        for run in results:
            algo_data = run.get("algorithms", {}).get(algo)
            if algo_data is None:
                continue
            total += 1
            if algo_data.get("success", False):
                success += 1
                pl = algo_data.get("path_length", None)
                nc = algo_data.get("node_count", None)
                ni = algo_data.get("num_iterations", None)
                if pl is not None and pl < 1e15:   # filter DBL_MAX failures
                    lengths.append(pl)
                if nc is not None:
                    nodes.append(nc)
                if ni is not None:
                    iters.append(ni)

        stats[algo] = {
            "total": total,
            "success": success,
            "success_rate": (success / total * 100) if total > 0 else 0.0,
            "mean_length": float(np.mean(lengths)) if lengths else None,
            "mean_nodes":  float(np.mean(nodes))   if nodes  else None,
            "mean_iters":  float(np.mean(iters))   if iters  else None,
            "std_length":  float(np.std(lengths))  if lengths else None,
            "std_nodes":   float(np.std(nodes))    if nodes  else None,
            "std_iters":   float(np.std(iters))    if iters  else None,
        }

    return {
        "file": json_path,
        "total_runs": len(results),
        "stats": stats,
    }


def score_map(map_stat: dict) -> Tuple[int, float]:
    """
    Score = number of metrics (0–6) where BRRT_Case3 beats each baseline.
    Tiebreaker = combined improvement margin (summed % improvements vs both
    baselines).
    Returns (score, margin).
    """
    c3 = map_stat["stats"].get(ALGO_CASE3, {})

    score  = 0
    margin = 0.0

    for baseline_key in BASELINES:
        bval_dict = map_stat["stats"].get(baseline_key, {})
        for key in ("mean_length", "mean_nodes", "mean_iters"):
            bval  = bval_dict.get(key)
            c3val = c3.get(key)
            if bval is None or c3val is None or bval == 0:
                continue
            if c3val < bval:
                score  += 1
                margin += (bval - c3val) / bval * 100.0   # % improvement

    return score, margin


def fmt(val, fmt_str=".2f"):
    if val is None:
        return "N/A"
    return f"{val:{fmt_str}}"


def pct(bv, cv):
    """Percent improvement of cv over bv (positive = cv is better/lower)."""
    if bv is not None and cv is not None and bv != 0:
        return f"{(bv - cv) / bv * 100:+.1f}%"
    return "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find the best map where BRRT_Case3 beats both BRRT and BG_BRRT.\n"
            "Algorithms compared: BRRT  |  BG-BRRT  |  BRRT_Case3 (proposed)"
        )
    )
    parser.add_argument(
        "--sweep-dir",
        required=True,
        help="Directory containing per-map JSON files (output of run_map_sweep.sh).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save a JSON summary of the best map.",
    )
    parser.add_argument(
        "--min-success",
        type=int,
        default=50,
        help=(
            "Minimum number of successful runs required for EACH algorithm "
            "for a map to be considered (default: 50)."
        ),
    )
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"ERROR: sweep directory not found: {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(glob.glob(str(sweep_dir / "*.json")))
    if not json_files:
        print(f"ERROR: no JSON files found in {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*78}")
    print(f"  BEST-MAP ANALYSIS")
    print(f"  Algorithms : BRRT  |  BG-BRRT  |  BRRT_Case3 (proposed)")
    print(f"  Sweep dir  : {sweep_dir}")
    print(f"  Maps found : {len(json_files)}")
    print(f"  Min success: {args.min_success} per algorithm")
    print(f"{'='*78}\n")

    # ── Load all maps ────────────────────────────────────────────────────────
    all_maps = []
    for jf in json_files:
        ms = load_map_stats(jf)
        if ms is None:
            continue

        # Filter: require minimum success count for ALL three algos
        skip = False
        for algo in ALGOS_ALL:
            suc = ms["stats"].get(algo, {}).get("success", 0)
            if suc < args.min_success:
                print(
                    f"  [SKIP] {os.path.basename(jf)} – {algo} only "
                    f"{suc} successes (need {args.min_success})"
                )
                skip = True
                break
        if skip:
            continue

        sc, mg = score_map(ms)
        ms["score"]  = sc
        ms["margin"] = mg
        all_maps.append(ms)

    if not all_maps:
        print(
            "No maps passed the minimum-success filter. "
            "Try lowering --min-success."
        )
        sys.exit(1)

    # ── Sort: score DESC, then margin DESC ──────────────────────────────────
    all_maps.sort(key=lambda m: (m["score"], m["margin"]), reverse=True)

    # ── Print ranked table ──────────────────────────────────────────────────
    # Columns: rank | file | score | BRRT_len | BG_len | C3_len | Δbrrt% | Δbg%
    #                              | BRRT_itr | BG_itr | C3_itr | Δbrrt% | Δbg%
    W   = 10   # numeric column width
    SEP = "  "

    hdr = (
        f"{'Rank':<5} {'File':<28} {'Sc':>3}{SEP}"
        f"{'BRRT_len':>{W}} {'BG_len':>{W}} {'C3_len':>{W}} "
        f"{'Δbrrt%':>7} {'Δbg%':>7}{SEP}"
        f"{'BRRT_itr':>{W}} {'BG_itr':>{W}} {'C3_itr':>{W}} "
        f"{'Δbrrt%':>7} {'Δbg%':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    for rank, ms in enumerate(all_maps, 1):
        b   = ms["stats"][ALGO_BRRT]
        bg  = ms["stats"][ALGO_BG]
        c3  = ms["stats"][ALGO_CASE3]
        fname = os.path.basename(ms["file"])
        if len(fname) > 28:
            fname = fname[:25] + "..."

        row = (
            f"{rank:<5} {fname:<28} {ms['score']:>3}{SEP}"
            f"{fmt(b['mean_length']):>{W}} "
            f"{fmt(bg['mean_length']):>{W}} "
            f"{fmt(c3['mean_length']):>{W}} "
            f"{pct(b['mean_length'],  c3['mean_length']):>7} "
            f"{pct(bg['mean_length'], c3['mean_length']):>7}{SEP}"
            f"{fmt(b['mean_iters'], ',.0f'):>{W}} "
            f"{fmt(bg['mean_iters'], ',.0f'):>{W}} "
            f"{fmt(c3['mean_iters'], ',.0f'):>{W}} "
            f"{pct(b['mean_iters'],  c3['mean_iters']):>7} "
            f"{pct(bg['mean_iters'], c3['mean_iters']):>7}"
        )
        print(row)

    # ── Best map detail ──────────────────────────────────────────────────────
    best = all_maps[0]
    b    = best["stats"][ALGO_BRRT]
    bg   = best["stats"][ALGO_BG]
    c3   = best["stats"][ALGO_CASE3]

    print(f"\n{'='*78}")
    print(f"  BEST MAP  =>  {best['file']}")
    print(f"  Score    : {best['score']} / {MAX_SCORE}   (margin: {best['margin']:.1f}%)")
    print(f"{'='*78}")

    print(f"\n  {'Metric':<22} {'BRRT':>14} {'BG-BRRT':>14} {'BRRT_Case3':>14}")
    print(f"  {'-'*66}")

    def detail_row(label, key, fmt_str=".2f"):
        bval  = b.get(key)
        bgval = bg.get(key)
        c3val = c3.get(key)
        print(
            f"  {label:<22} "
            f"{fmt(bval,  fmt_str):>14} "
            f"{fmt(bgval, fmt_str):>14} "
            f"{fmt(c3val, fmt_str):>14}"
        )

    detail_row("Success rate (%)",  "success_rate")
    detail_row("Mean path length",  "mean_length")
    detail_row("Std path length",   "std_length")
    detail_row("Mean node count",   "mean_nodes",  ",.0f")
    detail_row("Std node count",    "std_nodes",   ",.0f")
    detail_row("Mean iterations",   "mean_iters",  ",.0f")
    detail_row("Std iterations",    "std_iters",   ",.0f")

    print(f"\n  Runs in map       : {best['total_runs']}")
    print(f"  BRRT successes    : {b['success']} / {b['total']}")
    print(f"  BG-BRRT successes : {bg['success']} / {bg['total']}")
    print(f"  Case3 successes   : {c3['success']} / {c3['total']}")

    # ── Improvement summary vs each baseline ────────────────────────────────
    print(f"\n  BRRT_Case3 improvement over baselines (best map):")
    for bkey, blabel in [(ALGO_BRRT, "BRRT"), (ALGO_BG, "BG-BRRT")]:
        bd = best["stats"][bkey]
        print(f"    vs {blabel}:")
        for metric, mlabel in [
            ("mean_length", "path len"),
            ("mean_nodes",  "nodes   "),
            ("mean_iters",  "iters   "),
        ]:
            bv = bd.get(metric)
            cv = c3.get(metric)
            print(f"      {mlabel}:  {pct(bv, cv)}")

    # ── Optional JSON output ────────────────────────────────────────────────
    if args.output:
        summary = {
            "best_map_file":  best["file"],
            "score":          best["score"],
            "max_score":      MAX_SCORE,
            "margin_pct":     best["margin"],
            "total_runs":     best["total_runs"],
            ALGO_BRRT:        {k: v for k, v in b.items()},
            ALGO_BG:          {k: v for k, v in bg.items()},
            ALGO_CASE3:       {k: v for k, v in c3.items()},
            "all_maps_ranked": [
                {
                    "file":     m["file"],
                    "score":    m["score"],
                    "margin":   m["margin"],
                    ALGO_BRRT:  m["stats"][ALGO_BRRT],
                    ALGO_BG:    m["stats"][ALGO_BG],
                    ALGO_CASE3: m["stats"][ALGO_CASE3],
                }
                for m in all_maps
            ],
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"\n  Summary saved to: {out_path}")

    print("")


if __name__ == "__main__":
    main()
