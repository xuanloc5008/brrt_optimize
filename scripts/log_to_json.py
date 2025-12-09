#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


PATTERNS = [
    r"\[(RRT#)\]\s+final path len:\s+([0-9.+-eE]+)",
    r"\[(RRT\*)\]\s+final path len:\s+([0-9.+-eE]+)",
    r"\[(RRT)\]\s+final path len:\s+([0-9.+-eE]+)",
    r"\[(BRRT\*)\]\s+final path len:\s+([0-9.+-eE]+)",
    r"\[(BRRT)\]\s+final path len:\s+([0-9.+-eE]+)",
    r"\[(BRRTOpitmizeCase1)\]\s+final path len:\s+([0-9.+-eE]+)",
    r"\[(BRRT_LOG)\]\s+time:\s+([0-9.+-eE]+)\s+iter:\s+([0-9.+-eE]+)\s+len:\s+([0-9.+-eE]+)",
    r"\[(BRRTOpitmizeCase1_LOG)\]\s+time:\s+([0-9.+-eE]+)\s+iter:\s+([0-9.+-eE]+)\s+len:\s+([0-9.+-eE]+)",
]
COMPILED = [re.compile(p) for p in PATTERNS]


def parse_log(log_path: Path):
    results = []
    counts = {}
    with log_path.open("r", errors="ignore") as f:
        for line in f:
            for regex in COMPILED:
                m = regex.search(line)
                if not m:
                    continue
                planner = m.group(1)
                # Handle extended log with time/iter/len
                if planner.endswith("_LOG"):
                    search_time = float(m.group(2))
                    iterations = float(m.group(3))
                    path_len = float(m.group(4))
                else:
                    search_time = None
                    iterations = None
                    path_len = float(m.group(2))
                # Timestamp is first token if present (ROS log format)
                tokens = line.strip().split()
                log_time = None
                if tokens:
                    try:
                        log_time = float(tokens[0])
                    except ValueError:
                        log_time = None
                counts[planner] = counts.get(planner, 0) + 1
                results.append(
                    {
                        "run_index": counts[planner],
                        "planner": planner,
                        "path_length": path_len,
                        "search_time": search_time,
                        "num_iterations": iterations,
                        "log_time": log_time,
                        "line": line.strip(),
                    }
                )
                break
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract planner results from ROS logs.")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path.home() / ".ros" / "log" / "latest" / "rosout.log",
        help="Path to rosout.log (default: ~/.ros/log/latest/rosout.log)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("result.json"),
        help="Where to write the JSON summary (default: result.json in cwd)",
    )
    args = parser.parse_args()

    if not args.log.exists():
        raise SystemExit(f"Log file not found: {args.log}")

    results = parse_log(args.log)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} entries to {args.out}")


if __name__ == "__main__":
    main()
