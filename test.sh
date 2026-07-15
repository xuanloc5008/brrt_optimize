#!/usr/bin/env bash
# =============================================================================
# test.sh — Build + chạy experiment so sánh BRRT
# =============================================================================
# Pipeline:
#   1) ./build.sh          → catkin_make toàn workspace
#   2) source devel/setup.bash
#   3) roslaunch path_finder test_planners.launch
#
# Môi trường:
#   - Map random_buildings (mặc định density 40%)
#   - Kích thước map: 300 x 300 x 50
#
# Thuật toán so sánh (flag mặc định = true):
#   - BRRT          : baseline RRT-Connect (bidirectional)
#   - BG_BRRT       : BRRT + goal bias
#   - BRRT_Case1    : ablation step 1
#   - BRRT_Case2    : ablation step 2
#   - BRRT_Case3    : main contribution (heuristic-cache + adaptive bias)
#
# Experiment:
#   - Đọc param heuristic từ brrt_input.json (p1, u_p, alpha, beta, gamma, epsilon)
#   - Mỗi lần chạy: 100 cặp start–goal ngẫu nhiên (dist >= 5)
#   - Mỗi cặp chạy lần lượt 5 thuật toán trên cùng start/goal
#   - Ghi kết quả: success, search_time, path_length, nodes, iterations
#   - Output mặc định: /tmp/brrt_result.json
#
# Cách chạy:
#   ./test.sh
#
# Tùy chọn (truyền thêm arg cho roslaunch):
#   ./test.sh run_brrt_case1:=false run_brrt_case2:=false
#   ./test.sh obs_density_percentage:=60.0 output_result:=/tmp/my_result.json
#   ./test.sh input_param:=/path/to/brrt_input.json
#
# RViz (terminal khác):
#   ./rviz.sh
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

echo "=============================================="
echo " BRRT Optimize — Comparison Experiment"
echo "=============================================="
echo " Build     : catkin_make (./build.sh)"
echo " Launch    : path_finder/test_planners.launch"
echo " Algorithms: BRRT | BG_BRRT | Case1 | Case2 | Case3*"
echo "             (* Case3 = main contribution)"
echo " Trials    : 100 start-goal pairs / run"
echo " Input     : brrt_input.json (hoặc arg input_param)"
echo " Output    : /tmp/brrt_result.json (hoặc arg output_result)"
echo "=============================================="
echo ""

bash ./build.sh
# shellcheck disable=SC1091
source devel/setup.bash
roslaunch path_finder test_planners.launch "$@"
