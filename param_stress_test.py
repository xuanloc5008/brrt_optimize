import os
import json
import itertools
import time
import pandas as pd
import subprocess  # Thêm thư viện này để chạy lệnh ROS
from itables import show

# --- 1. CẤU HÌNH THƯ MỤC ---
# Đường dẫn gốc của workspace
workspace_setup = "/home/phuong/DACN/ICIT/brrt_optimize/devel/setup.bash"

# Thư mục chứa file json
input_dir = "/home/phuong/DACN/ICIT/brrt_optimize/eval/input_case3/"
output_dir = "/home/phuong/DACN/ICIT/brrt_optimize/eval/output_case3/"

os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# --- 2. TẠO CÁC BỘ THAM SỐ CẦN TEST ---
weight_grades = [1.0, 3.0, 5.0]
lidar_radii = [10.0, 15.0, 20.0]
n_blocks_list = [8, 16, 32]
steer_lengths = [0.5, 1.0, 2.0]

# Giữ cố định heuristic
fixed_p1, fixed_u_p, fixed_alpha, fixed_beta, fixed_gamma, fixed_epsilon = 0.8, 2.0, 0.5, 0.3, 0.5, 1.0

combinations = list(itertools.product(weight_grades, lidar_radii, n_blocks_list, steer_lengths))
print(f"Total combinations to test: {len(combinations)}")

trial = 1
for wg, lr, nb, sl in combinations:
    environment = f"wg{wg}_lr{lr}_nb{nb}_sl{sl}"
    config = {
        "trial": trial, "environment": environment,
        "p1": fixed_p1, "u_p": fixed_u_p, "alpha": fixed_alpha, 
        "beta": fixed_beta, "gamma": fixed_gamma, "epsilon": fixed_epsilon,
        "weight_grade": wg, "lidar_radius": lr, "n_blocks": nb, "steer_length": sl
    }
    with open(os.path.join(input_dir, f"brrt_{environment}.json"), "w") as f:
        json.dump(config, f, indent=4)
    trial += 1

# --- 3. CHẠY TỰ ĐỘNG BẰNG ROS ---
files_input = set(os.listdir(input_dir))
files_output = set(os.listdir(output_dir))
only_in_input = files_input - files_output

print(f"Files left to run: {len(only_in_input)}")
for f in sorted(only_in_input):
    in_path = os.path.join(input_dir, f)
    out_path = os.path.join(output_dir, f)
    
    # Câu lệnh khởi chạy thuật toán (đã cập nhật đường dẫn workspace)
    cmd = f"source {workspace_setup} && roslaunch path_finder test_planners.launch input_param:={in_path} output_result:={out_path}"
    
    print(f"Running: {f}")
    # ĐÃ SỬA: Dùng subprocess thay cho os.system để gọi được lệnh source qua bash
    subprocess.run(cmd, shell=True, executable='/bin/bash')
    time.sleep(1) # Nghỉ một chút giữa các lần chạy để ROS dọn dẹp node

# --- 4. TỔNG HỢP VÀ TÌM BỘ SỐ TỐI ƯU ---
print("\n=== ĐANG PHÂN TÍCH KẾT QUẢ ===")
records = []
for filename in os.listdir(output_dir):
    if not filename.endswith(".json"): continue
        
    with open(os.path.join(output_dir, filename), 'r') as f:
        content = json.load(f)
        params = content.get("parameters", {})
        
        case3_runs = []
        for run_entry in content.get("results", []):
            if "BRRT_Case3" in run_entry["algorithms"]:
                res = run_entry["algorithms"]["BRRT_Case3"]
                case3_runs.append({
                    "success": res["success"],
                    "search_time": res["search_time"],
                    "path_length": res["path_length"],
                    "node_count": res.get("node_count", 0)
                })
        
        if case3_runs:
            df_temp = pd.DataFrame(case3_runs)
            df_success = df_temp[df_temp["success"] == True]
            
            records.append({
                "weight_grade": params.get("weight_grade"),
                "lidar_radius": params.get("lidar_radius"),
                "n_blocks": params.get("n_blocks"),
                "steer_length": params.get("steer_length"),
                "Success_Rate": df_temp["success"].mean(),
                "Avg_Time(s)": df_success["search_time"].mean() if not df_success.empty else None,
                "Avg_Length": df_success["path_length"].mean() if not df_success.empty else None,
                "Avg_Nodes": df_success["node_count"].mean() if not df_success.empty else None
            })


df_results = pd.DataFrame(records)

if df_results.empty:
    print("Chưa có dữ liệu hợp lệ trong output_dir. Hãy kiểm tra xem C++ planner đã ghi file thành công chưa.")
else:
    # Xếp hạng: Tỉ lệ thành công giảm dần -> Thời gian chạy tăng dần -> Chiều dài đường tăng dần
    leaderboard_df = df_results.sort_values(
        by=["Success_Rate", "Avg_Time(s)", "Avg_Length"], 
        ascending=[False, True, True]
    ).reset_index(drop=True)

    print("\n🏆 BẢNG XẾP HẠNG TOP 20 BỘ THAM SỐ TỐI ƯU NHẤT CHO CASE 3 🏆\n")
    
    # Dùng hàm của Pandas để in ra Terminal cho đẹp
    print(leaderboard_df.head(20).to_string())

    # LƯU RA FILE CSV ĐỂ DỄ ĐỌC BẰNG EXCEL/LIBREOFFICE
    csv_path = os.path.join(output_dir, "bang_xep_hang_case3.csv")
    leaderboard_df.to_csv(csv_path, index=False)
    print(f"\n✅ Đã lưu toàn bộ bảng xếp hạng ra file: {csv_path}")