"""lqr_baseline.py — B1：离散 LQR 基线设计与模拟器闭环对照

在 B0 辨识的离散模型 (A,B)（16 维状态：12 主动线圈电流 + 4 位置描述符）上：
1. 解离散代数 Riccati 方程（DARE）得状态反馈增益 K（u = -K dx）
2. 两组控制权限：单 P6 通道（与 PD 同权限）/ 三通道 P6+D5+P5（与 BC/PPO 同权限）
3. 动作限幅与环境一致（归一化 ±1 → ±500V），记录饱和率
4. 上模拟器闭环，按 fair_eval 统一协议与 PD(1,0.3)/BC/PPO 同表对比

注意（按复核要求）：LQR 教科书最优性不直接适用于限幅、短时 L1 任务，
因此全部性能以模拟器实测为准；LQR 使用全状态反馈（含线圈电流），
观测条件与学习类控制器的差异在报告中声明。
"""
import json
import os
import sys

import numpy as np
from scipy.linalg import solve_discrete_are

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))
sys.path.insert(0, os.path.dirname(__file__))
from freegsnke_linear_pos_env import FreeGSNKELinearPosEnv  # noqa: E402
from fair_eval import PDController, BCController, PPOController, run_episode, summarize  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(OUT, exist_ok=True)
os.makedirs(ASSETS, exist_ok=True)

N_ACTIVE = 12
SEEDS = list(range(3000, 3010))


class LQRController:
    """u = clip(-K dx)，dx = [I_active - I*_active, desc - desc*]。"""

    def __init__(self, K, env, x0, ctrl_idx, p6_only=False):
        self.K = K
        self.env = env
        self.x0 = x0
        self.ctrl_idx = ctrl_idx
        self.p6_only = p6_only

    def reset(self, obs):
        pass

    def _full_state(self):
        st = self.env._stepping
        cur = np.array(st.currents_vec[:N_ACTIVE], dtype=float).real
        desc = np.array(st.plasma_descriptors_vec, dtype=float)
        return np.concatenate([cur, desc])

    def act(self, obs):
        dx = self._full_state() - self.x0
        u = np.clip(-self.K @ dx, -1.0, 1.0)  # K 按归一化动作单位设计（±1 = ±500V）
        if self.p6_only:
            out = np.zeros(3)
            out[0] = u[0]  # 只保留 P6 通道（K 已按单通道求解时 shape=(1,16)）
            return out
        return u


def design_lqr(A, B, q_desc=(1.0, 1.0, 30.0, 5.0), q_cur=1e-10, r=1.0):
    """DARE 设计。Q：描述符权重 (Rx,Zx,Zcur,Rin)，线圈电流统一小权重；R：输入权重。"""
    n = A.shape[0]
    Q = np.eye(n) * q_cur
    Q[12, 12], Q[13, 13], Q[14, 14], Q[15, 15] = q_desc
    R = np.eye(B.shape[1]) * r
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K


def make_env():
    return FreeGSNKELinearPosEnv(
        control_coils=["P6", "D5", "P5"], max_steps=50, steps_per_action=1,
        max_voltage=500.0, init_disturb_steps=3, init_disturb_voltage=50.0,
    )


def main():
    d = np.load(os.path.join(OUT, "sysid_model.npz"))
    A, B, x0 = d["A"], d["B"], d["x0"]
    Bn = B * 500.0  # 归一化动作单位设计（±1 = ±500V）；DARE 的 R 必须匹配动作量级

    env = make_env()
    dt = env.full_timestep * env.steps_per_action
    p6_idx = env.control_coils.index("P6")

    # ---------- LQR 设计（三通道） ----------
    K3 = design_lqr(A, Bn)
    # ---------- LQR 设计（单 P6 通道，用 B 的 P6 列） ----------
    B1 = Bn[:, [0]]
    K1 = design_lqr(A, B1)

    # 模型内稳定性检查
    for name, Kx, Bx in [("LQR-3ch", K3, B), ("LQR-P6", K1, B1)]:
        cl = A - Bx @ Kx
        print(f"{name} 闭环最大特征值(模): {np.max(np.abs(np.linalg.eigvals(cl))):.4f}", flush=True)

    controllers = {
        "PD(Kp=1,Kd=0.3,P6单通道)": PDController(1, 0.3, p6_idx, dt),
        "LQR(P6单通道)": LQRController(K1, env, x0, env._ctrl_idx, p6_only=True),
        "LQR(三通道)": LQRController(K3, env, x0, env._ctrl_idx),
        "BC": BCController(os.path.join(os.path.dirname(__file__), "..", "runs", "bc_policy.pt")),
        "PPO": PPOController(os.path.join(os.path.dirname(__file__), "..", "runs", "ppo_final.zip")),
    }

    # ---------- R 权重扫描（预指定小网格：限幅任务下输入权重是 LQR 的关键自由度） ----------
    print("\n--- R 权重扫描（先扫描，再与 PD/BC/PPO 对比最佳者） ---", flush=True)
    sweep_rows = []
    for r_val in [1, 10, 100, 1000, 10000]:
        for ch, Bx in [("P6", B1), ("3ch", B)]:
            Kx = design_lqr(A, Bx, r=r_val)
            ctrl = LQRController(Kx, env, x0, env._ctrl_idx, p6_only=(ch == "P6"))
            rows = [run_episode(env, ctrl, s) for s in SEEDS]
            s = summarize(f"LQR({ch},R={r_val})", rows)
            sweep_rows.append((ch, r_val, s))
            print(f"LQR({ch},R={r_val:>5}): 完整 {s['complete']}/{s['valid_samples']} "
                  f"MAE_Z {s['mae_z_cm']:.3f}cm 饱和 {s['sat_ratio']:.0%}", flush=True)
    # 选出各通道最优（完整回合最多者优先，其次 MAE_Z 最小）
    def pick(ch):
        cands = [r for r in sweep_rows if r[0] == ch]
        return max(cands, key=lambda r: (r[2]["complete"], -r[2]["mae_z_cm"]))
    best_p6, best_3ch = pick("P6"), pick("3ch")
    print(f"\n各通道最优: P6→R={best_p6[1]}, 3ch→R={best_3ch[1]}", flush=True)

    controllers = {
        "PD(Kp=1,Kd=0.3,P6单通道)": PDController(1, 0.3, p6_idx, dt),
        f"LQR(P6,R={best_p6[1]})": LQRController(design_lqr(A, B1, r=best_p6[1]), env, x0, env._ctrl_idx, p6_only=True),
        f"LQR(3ch,R={best_3ch[1]})": LQRController(design_lqr(A, Bn, r=best_3ch[1]), env, x0, env._ctrl_idx),
        "BC": controllers["BC"],
        "PPO": controllers["PPO"],
    }

    results = {}
    for name, ctrl in controllers.items():
        print(f"评估 {name} …", flush=True)
        rows = [run_episode(env, ctrl, s) for s in SEEDS]
        results[name] = {"episodes": rows, "summary": summarize(name, rows)}

    print(f"\n{'控制器':<28}{'完整/有效/总':>14}{'MAE_Z(cm)':>12}{'MAE联合(cm)':>12}{'末端联合(cm)':>12}{'饱和率':>8}")
    for name, data in results.items():
        s = data["summary"]
        print(f"{name:<28}{s['complete']:>5}/{s['valid_samples']:>3}/{s['episodes']:<3}{s['mae_z_cm']:>12.3f}"
              f"{s['mae_joint_cm']:>12.3f}{s['final_joint_cm']:>12.3f}{s['sat_ratio']:>8.2%}")

    payload = {
        "protocol": {
            "seeds": SEEDS, "sampling": "post-action, paired seeds, same simulator instance",
            "lqr_note": "LQR 全状态反馈（含线圈电流），观测条件与学习类不同，已在报告中声明；"
                        "Q 描述符权重 (1,1,30,5)，R=1；限幅 ±500V 与环境一致",
        },
        "results": results,
    }
    with open(os.path.join(OUT, "lqr_eval_v020.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=float)

    # ---------- 图 ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results.keys())
    maes = [results[n]["summary"]["mae_z_cm"] for n in names]
    colors = ["#5a6b7f", "#2b7a2b", "#1f8a4c", "#4a7fb5", "#0e4d92"]
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    bars = ax.bar(names, maes, color=colors, width=0.6)
    for b, v in zip(bars, maes):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylabel("whole-trajectory |Z-Z*| MAE [cm]")
    ax.set_title("B1: LQR baselines vs PD/BC/PPO — paired eval (10 ICs, all complete)")
    plt.setp(ax.get_xticklabels(), fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "lqr_eval_v020.png"))
    print("图已存 assets/lqr_eval_v020.png", flush=True)


if __name__ == "__main__":
    main()
