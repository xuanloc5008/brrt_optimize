import os
import json
import time
import subprocess
import pandas as pd
import optuna

# --- 1. CẤU HÌNH THƯ MỤC & ROS ---
workspace_setup = "/home/phuong/DACN/ICIT/brrt_optimize/devel/setup.bash"
input_dir = "/home/phuong/DACN/ICIT/brrt_optimize/eval/input_optuna/"
output_dir = "/home/phuong/DACN/ICIT/brrt_optimize/eval/output_optuna/"

os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Các tham số cố định
fixed_p1, fixed_u_p, fixed_alpha, fixed_beta, fixed_gamma, fixed_epsilon = 0.8, 2.0, 0.5, 0.3, 0.5, 1.0

# --- 2. ĐỊNH NGHĨA HÀM MỤC TIÊU (OBJECTIVE) ---
def objective(trial):
    # a. Optuna gợi ý tham số cho vòng lặp này
    wg = trial.suggest_float("weight_grade", 1.0, 5.0, step=0.5)
    lr = trial.suggest_float("lidar_radius", 5.0, 25.0, step=2.5)
    nb = trial.suggest_categorical("n_blocks", [8, 16, 24, 32])
    sl = trial.suggest_float("steer_length", 0.5, 3.0, step=0.5)
    
    trial_name = f"trial_{trial.number}"
    in_path = os.path.join(input_dir, f"{trial_name}.json")
    out_path = os.path.join(output_dir, f"{trial_name}.json")
    
    # b. Ghi file cấu hình JSON
    config = {
        "trial": trial.number, "environment": trial_name,
        "p1": fixed_p1, "u_p": fixed_u_p, "alpha": fixed_alpha, 
        "beta": fixed_beta, "gamma": fixed_gamma, "epsilon": fixed_epsilon,
        "weight_grade": wg, "lidar_radius": lr, "n_blocks": nb, "steer_length": sl
    }
    with open(in_path, "w") as f:
        json.dump(config, f, indent=4)
        
    # c. Chạy ROS Node
    cmd = f"source {workspace_setup} && roslaunch path_finder test_planners.launch input_param:={in_path} output_result:={out_path}"
    print(f"\n[Trial {trial.number}] Running with wg={wg}, lr={lr}, nb={nb}, sl={sl}")
    subprocess.run(cmd, shell=True, executable='/bin/bash', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # Ẩn log ROS cho đỡ rối
    time.sleep(1)
    
    # d. Đọc kết quả và tính điểm
    if not os.path.exists(out_path):
        return 9999.0 # Lỗi crash node, phạt điểm tối đa
        
    with open(out_path, 'r') as f:
        content = json.load(f)
        
    case3_runs = []
    for run_entry in content.get("results", []):
        if "BRRT_Case3" in run_entry["algorithms"]:
            res = run_entry["algorithms"]["BRRT_Case3"]
            case3_runs.append({"success": res["success"], "search_time": res["search_time"], "path_length": res["path_length"]})
            
    if not case3_runs:
        return 9999.0
        
    df_temp = pd.DataFrame(case3_runs)
    success_rate = df_temp["success"].mean()
    
    df_success = df_temp[df_temp["success"] == True]
    avg_time = df_success["search_time"].mean() if not df_success.empty else 100.0
    avg_length = df_success["path_length"].mean() if not df_success.empty else 1000.0
    
    # e. Công thức tính điểm (Càng nhỏ càng tốt)
    # Phạt cực nặng nếu success_rate thấp (ví dụ: < 100% thì cộng thêm hàng chục giây)
    penalty = (1.0 - success_rate) * 200.0 
    
    # Mục tiêu chính: Giảm thời gian. Phụ: Giảm chiều dài (nhân hệ số nhỏ để không lấn át thời gian)
    score = avg_time + (avg_length * 0.01) + penalty
    
    print(f"   -> Result: Success={success_rate*100}%, Time={avg_time:.4f}s | Score={score:.4f}")
    return score

# --- 3. KHỞI ĐỘNG OPTUNA ---
print("Bắt đầu tối ưu hóa với Optuna...")
# Tạo study với mục tiêu là MINIMIZE (điểm càng thấp càng tốt)
study = optuna.create_study(direction="minimize", study_name="BRRT_Case3_Optimization")

# Chạy 50 vòng thử nghiệm (Bạn có thể tăng lên 100 nếu có thời gian)
study.optimize(objective, n_trials=100)

# --- 4. IN KẾT QUẢ TỐI ƯU NHẤT ---
print("\n" + "="*50)
print("🏆 ĐÃ TÌM RA BỘ THAM SỐ TỐI ƯU NHẤT 🏆")
print("="*50)
best_trial = study.best_trial
print(f"Điểm số (Score): {best_trial.value:.4f}")
print("Các tham số:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Lưu toàn bộ lịch sử thử nghiệm ra CSV
df_history = study.trials_dataframe()
csv_path = os.path.join(output_dir, "optuna_history.csv")
df_history.to_csv(csv_path, index=False)
print(f"\nĐã lưu lịch sử chạy ra file: {csv_path}")