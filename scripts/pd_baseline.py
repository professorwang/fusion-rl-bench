"""pd_baseline.py — 经典 PD 控制器 baseline：P6 电压 = -(Kp·z_err + Kd·ż_err)

每个 RL benchmark 都需要经典控制器对照（DeepMind TCV 论文亦如此）。
同时验证环境/奖励良定义：若 PD 能稳住而 PPO 学不会，说明是 RL 调参问题而非任务不可行。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))
from freegsnke_linear_pos_env import FreeGSNKELinearPosEnv  # noqa: E402


def run_episode(env, kp, kd, p6_idx, max_steps=50, seed=0):
    obs, _ = env.reset(seed=seed)
    z_prev = obs[2]
    errs, holds = [], 0
    done = False
    while not done:
        z_err = obs[2]  # 归一化（/10cm）
        z_dot = (obs[2] - z_prev) / (env.full_timestep * env.steps_per_action)
        z_prev = obs[2]
        a = np.zeros(env.action_space.shape)
        a[p6_idx] = np.clip(-(kp * z_err + kd * z_dot * 1e-3), -1, 1)  # z_dot 缩放到同量级
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        if not term:
            holds += 1
            errs.append(abs(z_err) * 0.1)  # 米
    return holds, (np.mean(errs) if errs else np.nan), info["fail_reason"]


def main():
    env = FreeGSNKELinearPosEnv(control_coils=["P6", "D5", "P5"],
                                max_steps=50, steps_per_action=1, max_voltage=500.0)
    p6_idx = env.control_coils.index("P6")
    print("Kp×Kd 网格扫描（P6 单通道 PD）：")
    best = None
    for kp in [10, 30, 100, 300]:
        for kd in [3, 10, 30, 100]:
            holds, err, fail = run_episode(env, kp, kd, p6_idx)
            tag = "✅" if holds >= 50 else "  "
            print(f"{tag} Kp={kp:4d} Kd={kd:4d}: 存活 {holds:2d}/50 步, 平均偏差 {err*100:.2f}cm ({fail})")
            if holds >= 50 and (best is None or err < best[1]):
                best = (kp, kd, err)
    if best:
        print(f"\n最佳: Kp={best[0]} Kd={best[1]}, 平均偏差 {best[2]*100:.2f}cm —— 经典控制可稳定 ✅")
    else:
        print("\n所有组合均失败 —— 需检查任务设置")


if __name__ == "__main__":
    main()
