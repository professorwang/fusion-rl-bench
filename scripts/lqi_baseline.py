"""lqi_baseline.py — LQI（带积分作用的 LQR）基线

B1 诊断结论：朴素 LQR 因模型-对象失配出现稳态偏差（z 稳定于 +0.07，u 保持 -0.024），
教科书解法：对控制目标加积分作用（LQI）——增广状态 [dx, ∫z_err]，
保证零稳态误差（对模型偏差不敏感）。
"""
import os
import sys

import numpy as np
from scipy.linalg import solve_discrete_are

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))
sys.path.insert(0, os.path.dirname(__file__))
from freegsnke_linear_pos_env import FreeGSNKELinearPosEnv  # noqa: E402
from fair_eval import PDController, BCController, PPOController, run_episode, summarize  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
N_ACTIVE = 12
SEEDS = list(range(3000, 3010))


class LQIController:
    """u = -K dx - Ki·z_int；z_int 为 Zcur 偏差的离散积分。"""

    def __init__(self, K, Ki, env, x0, obs_scale_z=0.1):
        self.K, self.Ki = K, Ki
        self.env = env
        self.x0 = x0
        self.z_scale = obs_scale_z  # obs[2] 归一化因子（米）
        self.z_int = 0.0

    def reset(self, obs):
        self.z_int = 0.0

    def act(self, obs):
        st = self.env._stepping
        cur = np.array(st.currents_vec[:N_ACTIVE], dtype=float).real
        desc = np.array(st.plasma_descriptors_vec, dtype=float)
        dx = np.concatenate([cur, desc]) - self.x0
        z_err_m = desc[2] - self.x0[14]
        self.z_int += z_err_m * 5e-4  # dt
        u = -self.K @ dx - self.Ki * self.z_int
        return np.clip(u, -1.0, 1.0)


def design_lqi(A, B, qz=30.0, qi=50.0, r=1.0):
    """增广 [dx; z_int]（z_int' = z_int + dt*Zcur_err），DARE 求解。"""
    n = A.shape[0]
    dt = 5e-4
    Aa = np.zeros((n + 1, n + 1))
    Aa[:n, :n] = A
    Cz = np.zeros((1, n))
    Cz[0, 14] = dt  # z_int += dt * Zcur_err
    Aa[n, :n] = Cz[0]
    Aa[n, n] = 1.0
    Ba = np.vstack([B, np.zeros((1, B.shape[1]))])
    Q = np.eye(n + 1) * 1e-10
    Q[14, 14] = qz
    Q[n, n] = qi
    P = solve_discrete_are(Aa, Ba, Q, np.eye(B.shape[1]) * r)
    Ka = np.linalg.solve(np.eye(B.shape[1]) * r + Ba.T @ P @ Ba, Ba.T @ P @ Aa)
    return Ka[:, :n], Ka[:, n]


def main():
    d = np.load(os.path.join(OUT, "sysid_model.npz"))
    A, B, x0 = d["A"], d["B"], d["x0"]
    Bn = B * 500.0

    env = FreeGSNKELinearPosEnv(
        control_coils=["P6", "D5", "P5"], max_steps=50, steps_per_action=1,
        max_voltage=500.0, init_disturb_steps=3, init_disturb_voltage=50.0,
    )
    dt = env.full_timestep * env.steps_per_action
    p6_idx = env.control_coils.index("P6")

    # LQI（三通道与单 P6 各一）
    K3, Ki3 = design_lqi(A, Bn, r=1.0)
    K1, Ki1 = design_lqi(A, Bn[:, [0]], r=1.0)

    class CH3(LQIController):
        def act(self, obs):
            return super().act(obs)

    class CH1(LQIController):
        def act(self, obs):
            u3 = super().act(obs)
            out = np.zeros(3)
            out[0] = u3[0]
            return out

    controllers = {
        "PD(Kp=1,Kd=0.3,P6单通道)": PDController(1, 0.3, p6_idx, dt),
        "LQI(P6单通道)": CH1(K1, Ki1, env, x0),
        "LQI(三通道)": CH3(K3, Ki3, env, x0),
        "BC": BCController(os.path.join(os.path.dirname(__file__), "..", "runs", "bc_policy.pt")),
        "PPO": PPOController(os.path.join(os.path.dirname(__file__), "..", "runs", "ppo_final.zip")),
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

    import json
    payload = {"protocol": {"seeds": SEEDS, "note": "LQI 增广积分作用于 Zcur（qi=50），r=1；其余同 fair_eval"},
               "results": results}
    with open(os.path.join(OUT, "lqi_eval_v021.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=float)


if __name__ == "__main__":
    main()
