#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(path: Path):
    with path.open() as f:
        data = json.load(f)
    return data


def plot_metrics(data, out_path: Path):
    path_series = defaultdict(list)
    time_series = defaultdict(list)
    iter_series = defaultdict(list)

    for entry in data:
        algo = entry.get("planner", "unknown")
        path_series[algo].append(entry.get("path_length", None))
        time_series[algo].append(entry.get("search_time", None))
        iter_series[algo].append(entry.get("num_iterations", None))

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    metrics = [
        ("Path length", path_series),
        ("Search time", time_series),
        ("Iterations", iter_series),
    ]

    for ax, (label, series) in zip(axes, metrics):
        for algo, values in sorted(series.items()):
            if all(v is None for v in values):
                continue
            ax.plot(range(1, len(values) + 1), values, marker="o", label=algo)
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[0].set_title("Per-run metrics")
    axes[-1].set_xlabel("Run index")
    axes[0].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    fig.savefig(out_path)
    print(f"Wrote plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot path lengths from result.json")
    parser.add_argument("--data", type=Path, required=True, help="Path to result.json from log_to_json.py")
    parser.add_argument("--out", type=Path, default=Path("plot.png"), help="Output image path")
    args = parser.parse_args()

    data = load_results(args.data)
    plot_metrics(data, args.out)


if __name__ == "__main__":
    main()
