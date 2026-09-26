"""pd_baseline.py — 经典 PD 控制器 baseline：P6 电压 = -(Kp·z_err + Kd·ż_err)

每个 RL benchmark 都需要经典控制器对照。
v0.1.1 修复（对应 2026-09-26 发表前审核 R2）：
- 挑参比较字段 bug：`err < best[1]` 误比 Kd，应为 `best[2]`（err）；
- 公平性：与 BC/PPO 相同的初始扰动（3 步 × 50 V 随机方向）与控制线圈集合；
- 多 episode 评估：每组增益跑 N_EPISODES 个 episode，报告完整回合比例与均值。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))
from freegsnke_linear_pos_env import FreeGSNKELinearPosEnv  # noqa: E402

N_EPISODES = 10


def run_episode(env, kp, kd, p6_idx, seed):
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
    env = FreeGSNKELinearPosEnv(
        control_coils=["P6", "D5", "P5"], max_steps=50, steps_per_action=1,
        max_voltage=500.0, init_disturb_steps=3, init_disturb_voltage=50.0,
    )
    p6_idx = env.control_coils.index("P6")
    print(f"Kp×Kd 网格扫描（P6 单通道 PD，{N_EPISODES} episode/组，带初始扰动）：")
    best = None
    n_success_total = 0
    for kp in [10, 30, 100, 300]:
        for kd in [3, 10, 30, 100]:
            holds_l, errs_l, fails = [], [], 0
            for seed in range(N_EPISODES):
                holds, err, fail = run_episode(env, kp, kd, p6_idx, seed)
                holds_l.append(holds)
                errs_l.append(err)
                if fail:
                    fails += 1
            full = sum(1 for h in holds_l if h >= 50)
            n_success_total += full
            mean_err = float(np.nanmean(errs_l))
            tag = "✅" if full == N_EPISODES else "  "
            print(f"{tag} Kp={kp:4d} Kd={kd:4d}: 完整回合 {full}/{N_EPISODES}, "
                  f"平均偏差 {mean_err*100:.2f}cm, 失败 {fails}")
            if full == N_EPISODES and (best is None or mean_err < best[2]):
                best = (kp, kd, mean_err)
    n_groups = 16
    print(f"\n全存活增益组: （见 ✅ 标记）")
    if best:
        print(f"最佳: Kp={best[0]} Kd={best[1]}, 平均偏差 {best[2]*100:.2f}cm —— 经典控制可稳定 ✅")
    else:
        print("所有组合均有失败回合 —— 需检查任务设置")


if __name__ == "__main__":
    main()
