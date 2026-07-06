#!/usr/bin/env python3
"""
Plot the p_bias trajectory tracked during BRRT_Case3 runs.

CSV format (no header):  run_index, iteration, p_bias
Output CSV path is derived from the output_result path in test_planners.launch
by replacing .json with _pbias.csv, e.g.:
  /home/xuanloc/DACN/brrt_optimize/eval/random_blobs/50_pbias.csv
"""

import argparse
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ── Constants that mirror the C++ adaptive-pbias parameters ──────────────────
P_MIN  = 0.05
P_MAX  = 0.90
P_INIT = 0.50
# ─────────────────────────────────────────────────────────────────────────────


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"[ERROR] File not found: {path}")
    df = pd.read_csv(path, header=None, names=["run", "iteration", "pbias"])
    print(f"Loaded {len(df):,} rows  |  runs: {sorted(df['run'].unique())[:5]}{'...' if df['run'].nunique() > 5 else ''}")
    return df


def plot_all_runs(df: pd.DataFrame, output_path=None):
    """One subplot per run, pbias vs iteration."""
    runs = sorted(df["run"].unique())
    n = len(runs)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             sharex=False, sharey=True)
    axes = np.array(axes).flatten()

    colors = cm.tab10.colors
    for ax_idx, run_id in enumerate(runs):
        ax = axes[ax_idx]
        sub = df[df["run"] == run_id]
        ax.plot(sub["iteration"], sub["pbias"],
                color=colors[ax_idx % len(colors)], linewidth=0.9)
        ax.axhline(P_MIN,  color="red",   linestyle="--", linewidth=0.7, label="p_min")
        ax.axhline(P_MAX,  color="green", linestyle="--", linewidth=0.7, label="p_max")
        ax.axhline(P_INIT, color="gray",  linestyle=":",  linewidth=0.7, label="p_init")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Run {run_id}", fontsize=8)
        ax.set_xlabel("Iteration", fontsize=7)
        ax.set_ylabel("p_bias", fontsize=7)
        ax.tick_params(labelsize=6)

    for ax in axes[n:]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=8)
    fig.suptitle("BRRT_Case3 – p_bias per Iteration (all runs)", fontsize=12, y=1.01)
    plt.tight_layout()
    _save_or_show(fig, output_path, suffix="_per_run")


def plot_aggregate(df: pd.DataFrame, output_path=None):
    """Aggregate: mean ± std and median across runs, bucketed by iteration bins."""
    n_bins = 200
    df = df.copy()
    df["iter_bin"] = pd.cut(df["iteration"], bins=n_bins, labels=False)

    agg = df.groupby("iter_bin")["pbias"].agg(["mean", "std", "median"]).dropna()
    iter_max = df["iteration"].max()
    bin_width = iter_max / n_bins
    agg["x"] = (agg.index + 0.5) * bin_width

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(agg["x"],
                    (agg["mean"] - agg["std"]).clip(P_MIN, P_MAX),
                    (agg["mean"] + agg["std"]).clip(P_MIN, P_MAX),
                    alpha=0.25, color="steelblue", label="mean ± std")
    ax.plot(agg["x"], agg["mean"],   color="steelblue", linewidth=1.5, label="mean")
    ax.plot(agg["x"], agg["median"], color="darkorange", linewidth=1.2,
            linestyle="--", label="median")
    ax.axhline(P_MIN,  color="red",   linestyle="--", linewidth=1, label=f"p_min={P_MIN}")
    ax.axhline(P_MAX,  color="green", linestyle="--", linewidth=1, label=f"p_max={P_MAX}")
    ax.axhline(P_INIT, color="gray",  linestyle=":",  linewidth=1, label=f"p_init={P_INIT}")
    ax.set_xlabel("Iteration (binned)", fontsize=11)
    ax.set_ylabel("p_bias", fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("BRRT_Case3 – p_bias Aggregate across All Runs", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, output_path, suffix="_aggregate")


def plot_heatmap(df: pd.DataFrame, output_path=None):
    """Heatmap: rows = runs, columns = iteration bins, colour = p_bias."""
    n_bins = 100
    runs = sorted(df["run"].unique())
    matrix = np.full((len(runs), n_bins), np.nan)

    iter_max = df["iteration"].max()
    bin_width = max(iter_max / n_bins, 1)

    for r_idx, run_id in enumerate(runs):
        sub = df[df["run"] == run_id].copy()
        sub["bin"] = (sub["iteration"] / bin_width).astype(int).clip(0, n_bins - 1)
        binned = sub.groupby("bin")["pbias"].mean()
        for b, v in binned.items():
            matrix[r_idx, b] = v

    fig, ax = plt.subplots(figsize=(12, max(4, len(runs) * 0.35 + 1)))
    im = ax.imshow(matrix, aspect="auto", vmin=P_MIN, vmax=P_MAX,
                   cmap="RdYlGn", origin="upper",
                   extent=[0, iter_max, len(runs) + 0.5, 0.5])
    plt.colorbar(im, ax=ax, label="p_bias")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Run", fontsize=11)
    ax.set_yticks(range(1, len(runs) + 1))
    ax.set_yticklabels(runs, fontsize=max(5, 8 - len(runs) // 10))
    ax.set_title("BRRT_Case3 – p_bias Heatmap (run × iteration)", fontsize=13)
    plt.tight_layout()
    _save_or_show(fig, output_path, suffix="_heatmap")


def _save_or_show(fig, output_path, suffix=""):
    if output_path:
        base, ext = os.path.splitext(output_path)
        ext = ext or ".png"
        path = base + suffix + ext
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot p_bias tracking from BRRT_Case3 runs.")
    parser.add_argument(
        "csv",
        nargs="?",
        default="/home/xuanloc/DACN/brrt_optimize/eval/random_blobs/50_pbias.csv",
        help="Path to the _pbias.csv file generated by test_planner.")
    parser.add_argument(
        "--save", "-s", metavar="OUTPUT",
        help="Save plots to this base path (e.g. ./pbias_plot.png). "
             "If omitted, displays interactively.")
    parser.add_argument(
        "--runs", "-r", metavar="N", type=int, default=0,
        help="Only plot the first N runs (0 = all).")
    args = parser.parse_args()

    df = load_csv(args.csv)

    if args.runs > 0:
        runs = sorted(df["run"].unique())[: args.runs]
        df = df[df["run"].isin(runs)]
        print(f"Filtered to first {args.runs} runs.")

    plot_aggregate(df, args.save)
    plot_heatmap(df, args.save)
    if df["run"].nunique() <= 20:
        plot_all_runs(df, args.save)
    else:
        print("Skipping per-run grid (>20 runs). Use --runs N to limit.")


if __name__ == "__main__":
    main()
