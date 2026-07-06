#!/bin/bash
# =============================================================================
# run_map_sweep.sh
#
# Runs 100 independent random maps under ONE fixed setting:
#   (MAP_TYPE, OBS_PCT, MAP_SIZE_X, MAP_SIZE_Y, MAP_SIZE_Z)
#
# For each map the ROS node runs NUMBER_TEST_TIMES=100 start/goal pairs
# and writes results to a per-map JSON file under eval/map_sweep/.
#
# Usage:
#   ./run_map_sweep.sh
#   MAP_TYPE=blob_map OBS_PCT=50 MAP_SIZE=500 ./run_map_sweep.sh
#   ./run_map_sweep.sh blob_map 50 500
#
# Override via positional args: $1=MAP_TYPE  $2=OBS_PCT  $3=MAP_SIZE (x=y)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Configurable defaults (override via env or positional args) ──────────────
MAP_TYPE="${1:-${MAP_TYPE:-blob_map}}"
OBS_PCT="${2:-${OBS_PCT:-40}}"
MAP_SIZE="${3:-${MAP_SIZE:-500}}"
MAP_SIZE_Z="${MAP_SIZE_Z:-40}"
NUM_MAPS="${NUM_MAPS:-50}"

OUTPUT_DIR="${SCRIPT_DIR}/eval/map_sweep"
INPUT_PARAM="${SCRIPT_DIR}/brrt_input.json"
LOG_DIR="${OUTPUT_DIR}/logs"

# ── Derived label used in filenames ─────────────────────────────────────────
SETTING_LABEL="${MAP_TYPE}_obs${OBS_PCT}_sz${MAP_SIZE}"

# ── Ensure directories exist ────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}/${SETTING_LABEL}"
mkdir -p "${LOG_DIR}"

# ── Source ROS workspace ─────────────────────────────────────────────────────
cd "${SCRIPT_DIR}"
source devel/setup.bash

# ── Build ────────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  Building workspace..."
echo "================================================================"
bash ./build.sh

echo ""
echo "================================================================"
echo "  MAP SWEEP STARTING"
echo "  Setting : MAP_TYPE=${MAP_TYPE}  OBS_PCT=${OBS_PCT}%"
echo "            MAP_SIZE=${MAP_SIZE}x${MAP_SIZE}x${MAP_SIZE_Z}"
echo "  Maps    : ${NUM_MAPS} random maps"
echo "  Pairs   : 100 start/goal pairs per map  (hardcoded in node)"
echo "  Output  : ${OUTPUT_DIR}/${SETTING_LABEL}/"
echo "================================================================"
echo ""

SWEEP_START=$(date +%s)
SUCCESS_COUNT=0
FAIL_COUNT=0

for i in $(seq 1 "${NUM_MAPS}"); do

    OUTPUT_FILE="${OUTPUT_DIR}/${SETTING_LABEL}/map_run${i}.json"
    ROS_LOG="${LOG_DIR}/${SETTING_LABEL}_run${i}.log"

    echo "──────────────────────────────────────────────────────────────"
    printf "  [%3d / %d]  %s\n" "${i}" "${NUM_MAPS}" "$(date '+%H:%M:%S')"
    echo "  Output : ${OUTPUT_FILE}"

    if roslaunch path_finder test_planners.launch \
            map_type:="${MAP_TYPE}" \
            obs_density_percentage:="${OBS_PCT}" \
            map_size_x:="${MAP_SIZE}" \
            map_size_y:="${MAP_SIZE}" \
            map_size_z:="${MAP_SIZE_Z}" \
            input_param:="${INPUT_PARAM}" \
            output_result:="${OUTPUT_FILE}" \
            > "${ROS_LOG}" 2>&1; then

        # Verify the output file was actually written and is non-empty
        if [[ -s "${OUTPUT_FILE}" ]]; then
            echo "  ✓  Run ${i} succeeded."
            (( SUCCESS_COUNT++ )) || true
        else
            echo "  ✗  Run ${i}: roslaunch exited cleanly but output JSON is empty/missing."
            (( FAIL_COUNT++ )) || true
        fi
    else
        echo "  ✗  Run ${i}: roslaunch returned non-zero exit code. See: ${ROS_LOG}"
        (( FAIL_COUNT++ )) || true
    fi

done

SWEEP_END=$(date +%s)
ELAPSED=$(( SWEEP_END - SWEEP_START ))
ELAPSED_MIN=$(( ELAPSED / 60 ))
ELAPSED_SEC=$(( ELAPSED % 60 ))

echo ""
echo "================================================================"
echo "  MAP SWEEP COMPLETE"
echo "  Setting  : ${SETTING_LABEL}"
echo "  Successes: ${SUCCESS_COUNT} / ${NUM_MAPS}"
echo "  Failures : ${FAIL_COUNT}"
echo "  Elapsed  : ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "  Results  : ${OUTPUT_DIR}/${SETTING_LABEL}/"
echo "================================================================"
echo ""
echo "  Next step — find the best map:"
echo "  python3 collect_best_map.py --sweep-dir \"${OUTPUT_DIR}/${SETTING_LABEL}\""
echo ""
