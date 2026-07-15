#!/bin/bash
# =============================================================================
# run_map_sweep.sh  —  Fixed-map parameter tuning sweep for BRRT_Optimize_Case3
#
# Strategy
# ────────
# 1. Build the workspace once.
# 2. Sweep over the Cartesian product of
#      eta_stagn  x  cylinder_radius_factor  x  epsilon_h  x  brrt_alpha
#    on ONE fixed map geometry (same map seed every combo).
# 3. For each combo, call roslaunch NUM_TRIALS times, merge the per-trial
#    JSON outputs into  eval/param_sweep/<label>.json
# 4. After the sweep, invoke score_param_sweep.py which ranks combos and
#    prints a leaderboard showing which params beat BRRT + BG-BRRT.
#
# Usage:
#   ./run_map_sweep.sh                  # full sweep with defaults
#   NUM_TRIALS=5 ./run_map_sweep.sh     # quick sanity check (5 trials)
#
# Override map geometry:
#   MAP_TYPE=random_buildings OBS_PCT=20 MAP_SIZE=300 ./run_map_sweep.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Map geometry (fixed for the whole sweep) ──────────────────────────────────
MAP_TYPE="${MAP_TYPE:-random_buildings}"
OBS_PCT="${OBS_PCT:-40}"
MAP_SIZE="${MAP_SIZE:-500}"
MAP_SIZE_Z="${MAP_SIZE_Z:-40}"

# Number of roslaunch invocations (= start/goal pairs) per parameter combo.
NUM_TRIALS="${NUM_TRIALS:-30}"

# ── Parameter grid ────────────────────────────────────────────────────────────
#
# eta_stagn (η):
#   Fraction of improvement in the stagnation window required to NOT trigger
#   a rho boost. Range: (0, 1). Smaller → more aggressive boosting.
ETA_STAGN_VALUES=(0.30 0.50 0.70 0.90)

# cylinder_radius_factor:
#   Radius of the heuristic-bias sampling cylinder = factor × steer_length.
#   Range: [1, ∞). Larger → wider cylinder, more uniform coverage.
CYLINDER_RADIUS_VALUES=(3.0 5.0 10.0 20.0)

# epsilon_h (ε_h):
#   Minimum fractional improvement threshold to reset stagnation counter.
#   Range: [0, 0.2]. Smaller → more sensitive to tiny improvements.
EPSILON_H_VALUES=(0.005 0.02 0.05 0.10)

# brrt_alpha (α):
#   Weight in the heuristic  h = α·|si−gi| + (1−α)·(|si−G|+|gi−S|)
#   Range: (0, 1). α=1 → pure bridge distance; α=0 → pure triangle term.
ALPHA_VALUES=(0.25 0.50 0.75 0.875)

# ── Directories ───────────────────────────────────────────────────────────────
OUTPUT_DIR="${SCRIPT_DIR}/eval/param_sweep"
LOG_DIR="${OUTPUT_DIR}/logs"
INPUT_PARAM="${SCRIPT_DIR}/brrt_input.json"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# ── Source ROS workspace ──────────────────────────────────────────────────────
cd "${SCRIPT_DIR}"
source devel/setup.bash

# ── Build once ────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  Building workspace..."
echo "================================================================"
bash ./build.sh

TOTAL_COMBOS=$(( ${#ETA_STAGN_VALUES[@]} * ${#CYLINDER_RADIUS_VALUES[@]} \
                 * ${#EPSILON_H_VALUES[@]} * ${#ALPHA_VALUES[@]} ))

echo ""
echo "================================================================"
echo "  PARAM SWEEP — Fixed-map BRRT_Optimize_Case3 tuning"
echo "  Map    : ${MAP_TYPE}  OBS=${OBS_PCT}%  ${MAP_SIZE}x${MAP_SIZE}x${MAP_SIZE_Z}"
echo "  Grid   : ${#ETA_STAGN_VALUES[@]} eta_stagn"
echo "         x ${#CYLINDER_RADIUS_VALUES[@]} cylinder_radius_factor"
echo "         x ${#EPSILON_H_VALUES[@]} epsilon_h"
echo "         x ${#ALPHA_VALUES[@]} brrt_alpha"
echo "  Total  : ${TOTAL_COMBOS} combos × ${NUM_TRIALS} trials each"
echo "  Output : ${OUTPUT_DIR}/"
echo "================================================================"
echo ""

SWEEP_START=$(date +%s)
SUCCESS_COUNT=0
FAIL_COUNT=0
COMBO_IDX=0

for ETA in "${ETA_STAGN_VALUES[@]}"; do
for CYL in "${CYLINDER_RADIUS_VALUES[@]}"; do
for EPS in "${EPSILON_H_VALUES[@]}"; do
for ALPHA in "${ALPHA_VALUES[@]}"; do

    (( COMBO_IDX++ )) || true
    LABEL="eta${ETA}_cyl${CYL}_eps${EPS}_a${ALPHA}"
    OUTPUT_FILE="${OUTPUT_DIR}/${LABEL}.json"
    TRIAL_PREFIX="${OUTPUT_DIR}/.trial_${LABEL}"

    echo "──────────────────────────────────────────────────────────────"
    printf "  [%3d / %d]  %s   @ %s\n" \
        "${COMBO_IDX}" "${TOTAL_COMBOS}" "${LABEL}" "$(date '+%H:%M:%S')"

    # Resume: skip already-completed combos
    if [[ -s "${OUTPUT_FILE}" ]]; then
        echo "  ↩  Already exists — skipping."
        (( SUCCESS_COUNT++ )) || true
        continue
    fi

    RUN_OK=true
    for TRIAL in $(seq 1 "${NUM_TRIALS}"); do
        TRIAL_OUT="${TRIAL_PREFIX}_${TRIAL}.json"
        TRIAL_LOG="${LOG_DIR}/${LABEL}_t${TRIAL}.log"

        if ! roslaunch path_finder test_planners.launch \
                map_type:="${MAP_TYPE}"               \
                obs_density_percentage:="${OBS_PCT}"  \
                map_size_x:="${MAP_SIZE}"             \
                map_size_y:="${MAP_SIZE}"             \
                map_size_z:="${MAP_SIZE_Z}"           \
                input_param:="${INPUT_PARAM}"         \
                output_result:="${TRIAL_OUT}"         \
                brrt_eta_stagn:="${ETA}"              \
                brrt_cylinder_radius:="${CYL}"        \
                brrt_epsilon_h:="${EPS}"              \
                brrt_alpha:="${ALPHA}"                \
                > "${TRIAL_LOG}" 2>&1; then
            echo "  ✗  trial ${TRIAL} exited non-zero. Log: ${TRIAL_LOG}"
            RUN_OK=false
            break
        fi

        if [[ ! -s "${TRIAL_OUT}" ]]; then
            echo "  ✗  trial ${TRIAL}: output JSON empty/missing."
            RUN_OK=false
            break
        fi
    done

    if $RUN_OK; then
        # Merge all trial JSONs into a single combo file
        python3 - "${OUTPUT_FILE}" "${LABEL}" \
                   "${ETA}" "${CYL}" "${EPS}" "${ALPHA}" \
                   "${TRIAL_PREFIX}"_*.json << 'PYEOF'
import sys, json, os

out_file    = sys.argv[1]
label       = sys.argv[2]
eta         = float(sys.argv[3])
cyl         = float(sys.argv[4])
eps         = float(sys.argv[5])
alpha       = float(sys.argv[6])
trial_files = sorted(sys.argv[7:])

all_results = []
for tf in trial_files:
    try:
        with open(tf) as f:
            data = json.load(f)
        # Support both {results:[...]} and flat list
        results = data.get("results", [data]) if isinstance(data, dict) else data
        all_results.extend(results)
        os.remove(tf)
    except Exception as e:
        print(f"  [warn] could not read {tf}: {e}", flush=True)

combo = {
    "label": label,
    "params": {
        "eta_stagn":              eta,
        "cylinder_radius_factor": cyl,
        "epsilon_h":              eps,
        "brrt_alpha":             alpha,
    },
    "num_trials": len(all_results),
    "results": all_results,
}
with open(out_file, "w") as f:
    json.dump(combo, f, indent=2)
print(f"  ✓  Merged {len(all_results)} trial results → {out_file}", flush=True)
PYEOF
        (( SUCCESS_COUNT++ )) || true
    else
        rm -f "${TRIAL_PREFIX}"_*.json
        (( FAIL_COUNT++ )) || true
    fi

done
done
done
done

SWEEP_END=$(date +%s)
ELAPSED=$(( SWEEP_END - SWEEP_START ))

echo ""
echo "================================================================"
echo "  SWEEP COMPLETE"
printf "  Combos done : %d / %d\n" "${SUCCESS_COUNT}" "${TOTAL_COMBOS}"
printf "  Failures    : %d\n" "${FAIL_COUNT}"
printf "  Elapsed     : %dm %ds\n" $(( ELAPSED/60 )) $(( ELAPSED%60 ))
echo "  Results dir : ${OUTPUT_DIR}/"
echo "================================================================"
echo ""

# ── Score and rank ────────────────────────────────────────────────────────────
if [[ -f "${SCRIPT_DIR}/score_param_sweep.py" ]]; then
    echo "  Scoring results..."
    python3 "${SCRIPT_DIR}/score_param_sweep.py" \
        --sweep-dir "${OUTPUT_DIR}" \
        --top 10
else
    echo "  [info] score_param_sweep.py not found — run it manually:"
    echo "    python3 score_param_sweep.py --sweep-dir ${OUTPUT_DIR}"
fi
