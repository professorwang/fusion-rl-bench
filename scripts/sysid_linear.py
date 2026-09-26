"""sysid_linear.py — B0：实际控制周期（0.5ms）下的离散模型辨识与开环验证

目标（阶段 B 首个验收点）：辨识离散模型 x_{k+1} = A x_k + B u_k（偏差坐标），
并验证其**一步与多步开环预测与原模拟器一致**，通过后才允许评估 LQR/LQG。

方法（按复核建议）：不直接把上游内部雅可比当控制矩阵用，而是用电压激励
做数值系统辨识——这本身就验证了"电压映射 + 积分步骤"链路。

- 状态 x = [12 个主动线圈电流, 4 个等离子体位置描述符(Rx,Zx,Zcur,Rin)]（16 维）
- 输入 u = 3 个控制线圈（P6,D5,P5）电压
- 偏差坐标：dx = x - x_baseline，拟合 dx_{k+1} = A dx_k + B du_k
- 激励：每个 rollout 从基线出发，40 步随机电压 U(±30V)
- 验证：①一步预测相对误差 ②50 步方波电压开环轨迹对比（模拟器 vs 模型）

输出：results/sysid_model.npz、results/sysid_validation.json、assets/sysid_openloop.png
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))
from freegsnke_linear_pos_env import FreeGSNKELinearPosEnv  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(OUT, exist_ok=True)
os.makedirs(ASSETS, exist_ok=True)

N_ACTIVE = 12
CTRL = ["P6", "D5", "P5"]
EXCITE_V = 30.0
ROLLOUTS, STEPS = 100, 40
SEED = 42


def get_state(env):
    """x = [主动线圈电流(12), 描述符(4)]"""
    st = env._stepping
    cur = np.array(st.currents_vec[:N_ACTIVE], dtype=float).real
    desc = np.array(st.plasma_descriptors_vec, dtype=float)
    return np.concatenate([cur, desc])


def apply_voltage(env, v_ctrl):
    st = env._stepping
    v = np.zeros(st.evol_metal_curr.n_active_coils)
    for a, i in zip(v_ctrl, env._ctrl_idx):
        v[i] = a
    st.nlstepper(active_voltage_vec=v, linear_only=True, no_GS=True, verbose=False)


def main():
    env = FreeGSNKELinearPosEnv(control_coils=CTRL, max_steps=10**9,
                                steps_per_action=1, max_voltage=500.0)
    env.reset(seed=0)
    x0 = get_state(env)
    rng = np.random.default_rng(SEED)

    # ---------- 1. 采集辨识数据 ----------
    DX, DU, DY = [], [], []
    for r in range(ROLLOUTS):
        env.reset(seed=SEED * 1000 + r)
        for _ in range(STEPS):
            x_k = get_state(env) - x0
            u_k = rng.uniform(-EXCITE_V, EXCITE_V, size=len(CTRL))
            apply_voltage(env, u_k)
            x_next = get_state(env) - x0
            DX.append(x_k)
            DU.append(u_k)
            DY.append(x_next)
    DX, DU, DY = np.array(DX), np.array(DU), np.array(DY)
    print(f"样本: {len(DX)}", flush=True)

    # ---------- 2. 最小二乘拟合 [A B] ----------
    Z = np.hstack([DX, DU])           # (N, 19)
    Theta, *_ = np.linalg.lstsq(Z, DY, rcond=None)  # (19, 16)
    A = Theta[:16].T                   # (16,16)
    B = Theta[16:].T                   # (16,3)
    np.savez(os.path.join(OUT, "sysid_model.npz"), A=A, B=B, x0=x0, ctrl=CTRL)

    # ---------- 3. 一步预测验证（留出 rollout） ----------
    env.reset(seed=999)
    rel_errs = {c: [] for c in ["coils", "Rx", "Zx", "Zcur", "Rin"]}
    for _ in range(50):
        x_k = get_state(env) - x0
        u_k = rng.uniform(-EXCITE_V, EXCITE_V, size=len(CTRL))
        apply_voltage(env, u_k)
        x_true = get_state(env) - x0
        x_pred = A @ x_k + B @ u_k
        scale = np.maximum(np.abs(x_true), 1e-6)
        e = np.abs(x_true - x_pred) / scale
        rel_errs["coils"].append(e[:12].mean())
        for j, c in enumerate(["Rx", "Zx", "Zcur", "Rin"]):
            rel_errs[c].append(e[12 + j])
    one_step = {c: float(np.mean(v)) for c, v in rel_errs.items()}
    print(f"一步预测平均相对误差: {one_step}", flush=True)

    # ---------- 4. 多步开环验证（方波电压程序） ----------
    env.reset(seed=777)
    desc_names = ["Rx", "Zx", "Zcur", "Rin"]
    sim_traj, mod_traj = [], []
    x_mod = get_state(env) - x0
    for k in range(50):
        u_k = np.array([20.0 if (k // 5) % 2 == 0 else -20.0, 0.0, 0.0])
        apply_voltage(env, u_k)
        sim_traj.append((get_state(env) - x0)[12:])
        x_mod = A @ x_mod + B @ u_k
        mod_traj.append(x_mod[12:])
    sim_traj, mod_traj = np.array(sim_traj), np.array(mod_traj)

    # 描述符尺度（用基线值的量级，防止除零）
    scale = np.maximum(np.abs(x0[12:]), np.array([0.1, 0.1, 0.1, 0.1]))
    multi_rmse = np.sqrt(np.mean(((sim_traj - mod_traj) / scale) ** 2, axis=0))
    multi_maxerr_cm = np.max(np.abs(sim_traj - mod_traj), axis=0) * 100  # m→cm
    print(f"50步开环 归一化RMSE: {multi_rmse.round(4)}", flush=True)
    print(f"50步开环 最大绝对误差(cm): {multi_maxerr_cm.round(3)}", flush=True)

    # ---------- 5. 图 ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), dpi=110)
    for j, ax in enumerate(axes):
        ax.plot(sim_traj[:, j], "k-", lw=2, label="simulator" if j == 0 else None)
        ax.plot(mod_traj[:, j], "r--", lw=1.5, label="identified model" if j == 0 else None)
        ax.set_title(desc_names[j])
        ax.set_xlabel("step (0.5ms)")
    axes[0].legend()
    fig.suptitle("B0: 50-step open-loop prediction vs simulator (square-wave voltage)")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "sysid_openloop.png"))

    # ---------- 6. 指标与验收 ----------
    metrics = {
        "state_dim": 16, "input_dim": 3, "control_period_s": 5e-4,
        "samples": len(DX), "rollouts": ROLLOUTS, "steps_per_rollout": STEPS,
        "one_step_mean_rel_err": one_step,
        "openloop_50step": {
            "normalized_rmse": multi_rmse.tolist(),
            "max_abs_err_cm": multi_maxerr_cm.tolist(),
        },
        "acceptance_note": "B0 验收点：一步/多步开环预测与原模拟器一致性。容差阈值由评审确认。",
    }
    with open(os.path.join(OUT, "sysid_validation.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"模型与验证数据已存 {OUT}", flush=True)


if __name__ == "__main__":
    main()
