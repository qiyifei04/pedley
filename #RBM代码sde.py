#RBM代码sde
import numpy as np
import matplotlib.pyplot as plt

# --- 物理核函数占位 (与之前相同) ---
def K1_short_range(dx):
    dist = np.linalg.norm(dx)
    if dist < 1e-5: return np.zeros_like(dx)
    return (dx / dist) * (1.0 / dist**2) 

def K2_long_range(dx):
    dist = np.linalg.norm(dx)
    if dist < 1e-5: return np.zeros_like(dx)
    return (dx / dist) * (1.0 / dist)

def calc_deterministic_torque(P_i):
    gravity_dir = np.array([0.0, 1.0]) 
    return gravity_dir - P_i 

# --- 修改后的主模拟函数：增加轨迹记录 ---
def simulate_active_cells_rbm_track(N=50, p=2, tau=0.01, T=2.0, dim=2):
    V_s = 1.0           
    D_r = 0.1           
    sigma_pos = 0.05    
    r_cutoff = 0.5      
    
    # 初始化
    X = np.random.rand(N, dim) * 10.0
    P = np.random.randn(N, dim)
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    
    num_steps = int(T / tau)
    
    # 创建一个数组来记录所有时间步的位置：形状为 (步数, 粒子数, 维度)
    X_history = np.zeros((num_steps, N, dim))
    
    for m in range(num_steps):
        X_new = np.copy(X)
        P_new = np.copy(P)
        
        indices = np.random.permutation(N)
        batches = [indices[k:k+p] for k in range(0, N, p)]
        
        for batch in batches:
            current_p = len(batch)
            
            for i in batch:
                # 短程力
                dx_all = X[i] - X
                distances = np.linalg.norm(dx_all, axis=1)
                mask_short = (distances < r_cutoff) & (np.arange(N) != i)
                neighbors_short = np.where(mask_short)[0]
                
                F_short = np.zeros(dim)
                for j in neighbors_short:
                    F_short += K1_short_range(X[i] - X[j])
                
                # 长程力 (RBM 核心)
                F_long_random = np.zeros(dim)
                if current_p >= 2:
                    for j in batch:
                        if j != i:
                            F_long_random += K2_long_range(X[i] - X[j])
                    F_long = F_long_random * (N - 1) / (current_p - 1)
                else:
                    F_long = np.zeros(dim)
                
                U_interact = F_short + F_long
                
                # SDE 更新
                dW_pos = np.random.randn(dim) * np.sqrt(tau)
                dW_rot = np.random.randn(dim) * np.sqrt(tau)
                
                X_new[i] = X[i] + (V_s * P[i] + U_interact) * tau + sigma_pos * dW_pos
                deterministic_torque = calc_deterministic_torque(P[i])
                P_new[i] = P[i] + deterministic_torque * tau + np.sqrt(2 * D_r) * dW_rot
                P_new[i] = P_new[i] / np.linalg.norm(P_new[i])
                
        X = X_new
        P = P_new
        
        # 记录当前步的位置
        X_history[m] = X

    return X_history, P

# --- 可视化函数 ---
def plot_trajectories(X_history, final_P):
    """
    绘制细胞的运动轨迹
    """
    plt.figure(figsize=(10, 8))
    
    N = X_history.shape[1]
    num_steps = X_history.shape[0]
    
    # 1. 绘制轨迹线
    for i in range(N):
        # 取出第 i 个细胞在所有时间步的 x 和 y 坐标
        x_traj = X_history[:, i, 0]
        y_traj = X_history[:, i, 1]
        
        # 绘制半透明的轨迹线
        plt.plot(x_traj, y_traj, color='gray', alpha=0.3, linewidth=1)
        
        # 2. 标记起点 (绿色圆点)
        plt.scatter(x_traj[0], y_traj[0], color='green', s=15, zorder=3, alpha=0.7)
        
        # 3. 标记终点 (红色圆点) 并在终点画出游动方向箭头
        plt.scatter(x_traj[-1], y_traj[-1], color='red', s=20, zorder=4)
        # 画方向箭头：dx, dy 取自 final_P，适当缩放长度方便观察
        arrow_length = 0.3
        plt.arrow(x_traj[-1], y_traj[-1], 
                  final_P[i, 0] * arrow_length, final_P[i, 1] * arrow_length, 
                  head_width=0.08, head_length=0.1, fc='blue', ec='blue', zorder=5)

    # 图例占位
    plt.scatter([], [], color='green', label='Start Position')
    plt.scatter([], [], color='red', label='End Position')
    plt.plot([], [], color='gray', alpha=0.5, label='Trajectory')
    plt.arrow(0, 0, 0, 0, head_width=0.1, fc='blue', ec='blue', label='Final Direction')

    plt.title("Active Cell Trajectories using Random Batch Method (RBM)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal') # 保持XY轴比例一致
    plt.show()

# --- 运行测试与绘图 ---
if __name__ == "__main__":
    print("开始带有轨迹记录的 RBM 模拟...")
    # 为了图像不至于太杂乱，这里将细胞数 N 减少到 50
    history, final_directions = simulate_active_cells_rbm_track(N=50, p=2, tau=0.01, T=3.0, dim=2)
    print("模拟结束，正在生成图像...")
    plot_trajectories(history, final_directions)