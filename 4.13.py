import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签（'SimHei' 是黑体）
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
def rbm_mkv_2d(N, p, tau, T, b, K, sigma, X0=None):
    """
    针对 2 维空间优化的随机批处理方法 (RBM) 计算 McKean-Vlasov 方程。
    """
    d = 2  # 强制锁定为 2 维
    assert N % p == 0, "粒子总数 N 必须能被批次大小 p 整除"
    
    n = N // p
    M = int(T / tau)
    
    if X0 is None:
        # 在二维平面内随机初始化粒子 (例如在 -5 到 5 之间)
        np.random.seed(42)  # 固定随机种子以便复现
    else:
        X = X0.copy()
        
    trajectories = np.zeros((M + 1, N, d))
    trajectories[0] = X
    
    for m in range(M):
        indices = np.random.permutation(N)
        batches = indices.reshape((n, p))
        X_next = np.zeros_like(X)
        
        for q in range(n):
            batch_idx = batches[q]
            X_batch = X[batch_idx] # 形状: (p, 2)
            
            # 计算批次内所有粒子对的向量差
            # diff 形状为 (p, p, 2)，其中 diff[i, j, :] = X_i - X_j (这是一个二维向量)
            diff = X_batch[:, np.newaxis, :] - X_batch[np.newaxis, :, :]
            
            # 计算二维核函数 K
            K_eval = K(diff) # 返回形状应为 (p, p, 2)
            
            # 将对角线（自己对自己的力）置为 [0, 0]
            for i in range(p):
                K_eval[i, i] = [0.0, 0.0]
                
            # 在 j 的维度上求和并取平均 (p > 1)
            F_batch = np.sum(K_eval, axis=1) / (p - 1)
            
            # 生成二维正态分布的布朗运动增量
            Z = np.random.randn(p, d)
            
            # 欧拉-丸山更新
            X_next[batch_idx] = (X_batch 
                                 + b(X_batch) * tau 
                                 + F_batch * tau 
                                 + sigma * np.sqrt(tau) * Z)
            
        X = X_next
        trajectories[m + 1] = X
        
    return trajectories

# ==========================================
# 2D 场景的具体函数定义与测试
# ==========================================
if __name__ == "__main__":
    # 1. 定义二维漂移项 (例如：一个较弱的中心拉力，防止粒子跑得太远)
    def b_2d(x):
        # x 形状: (p, 2)
        return -0.1 * x

    # 2. 定义二维相互作用核 K (例如：带距离衰减的吸引力)
    def K_2d(diff):
        # diff 形状: (p, p, 2)
        # 计算二维向量的模长 (距离矩阵)，形状: (p, p, 1)
        # 加上 1e-5 是为了防止重合时除以 0
        norm = np.linalg.norm(diff, axis=-1, keepdims=True) + 1e-5
        
        # 相互作用力：方向朝向对方 (-diff)，随距离衰减
        force = -diff / (1.0 + norm) 
        return force

    # 设置参数
    N_particles = 100
    p_batch = 4        # p=3 表示每三个粒子相互作用，这是 RBM 最极致的情况
    tau_step = 0.01
    T_total = 5.0      # 延长模拟时间以观察聚集效应
    sigma_diff = 0.5   # 噪声强度

    print("开始运行 2D RBM 模拟...")
    traj = rbm_mkv_2d(N=N_particles, p=p_batch, tau=tau_step, T=T_total, 
                      b=b_2d, K=K_2d, sigma=sigma_diff)
    
    # --- 简单的二维轨迹可视化 (仅绘制初始和最终状态) ---
    plt.figure(figsize=(8, 8))
    
    # 提取初始状态 (m=0) 和 最终状态 (m=M)
    X_start = traj[0]
    X_end = traj[-1]
    
    plt.scatter(X_start[:, 0], X_start[:, 1], c='blue', alpha=0.3, label='初始位置')
    plt.scatter(X_end[:, 0], X_end[:, 1], c='red', marker='x', label='最终位置 (T=5.0)')
    
    plt.title("2D MKV 粒子系统 (Random Batch Method)")
    plt.xlabel("X 坐标")
    plt.ylabel("Y 坐标")
    plt.legend()
    plt.grid(True)
    plt.show()