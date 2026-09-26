"""bc_warmup.py — 行为克隆预热：PD 专家数据 → MLP 策略（模仿→超越范式的"模仿"段）

流程：
1. 用 PD 控制器（Kp=100, Kd=100，P6 通道）在带初始扰动的环境上采集 (obs, action) 专家数据
   （动作叠加小噪声扩大状态覆盖，DAgger 思想）
2. 训练与 SB3 MlpPolicy 同构的 MLP [64,64]，MSE 回归动作
3. 评估 BC 策略（存活步数/平均偏差），权重存 runs/bc_policy.pt
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))
from freegsnke_linear_pos_env import FreeGSNKELinearPosEnv  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "runs")
os.makedirs(OUT, exist_ok=True)  # v0.1.1：fresh clone 不含 runs/，保存前必须建目录
KP, KD = 100.0, 100.0
N_EPISODES = 150
EXPERT_NOISE = 0.05  # 专家动作噪声（扩大状态覆盖；过大会抬高 MSE 噪声地板，0.15 已实测欠拟合）
DEVICE = "cpu"


def pd_action(env, obs, z_prev, p6_idx, noise=0.0, rng=None):
    z_err = obs[2]
    z_dot = (obs[2] - z_prev) / (env.full_timestep * env.steps_per_action)
    a = np.zeros(env.action_space.shape)
    a[p6_idx] = np.clip(-(KP * z_err + KD * z_dot * 1e-3), -1, 1)
    if noise > 0 and rng is not None:
        a += rng.normal(0, noise, size=a.shape)
    return np.clip(a, -1, 1), z_err


def collect(env, n_episodes):
    p6_idx = env.control_coils.index("P6")
    rng = np.random.default_rng(0)
    X, Y = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        z_prev = obs[2]
        done = False
        while not done:
            a, z_prev = pd_action(env, obs, z_prev, p6_idx, EXPERT_NOISE, rng)
            X.append(obs.copy())
            Y.append(a.copy())
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


class BCPolicy(nn.Module):
    """与 SB3 MlpPolicy(net_arch=[64,64]) 同构，便于权重迁移。

    v0.1.1 复核（E项）：输出层改为**线性**（不再接 Tanh），与 PPO 的
    "线性均值→环境内裁剪"结构一致——迁移复制权重后动作映射等价；
    推理时的 [-1,1] 裁剪在控制器/环境侧统一处理。
    """

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, act_dim),  # 线性输出（迁移等价，见模块注释）
        )

    def forward(self, x):
        return self.net(x)


def evaluate(env, policy, n_episodes=10):
    holds_list, errs = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done, holds, ep_errs = False, 0, []
        while not done:
            with torch.no_grad():
                a = policy(torch.as_tensor(obs, dtype=torch.float32)).numpy()
            a = np.clip(a, -1, 1)  # 线性输出头：裁剪在控制器侧统一处理（与 PPO 迁移后行为一致）
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            if not term:
                holds += 1
                ep_errs.append(abs(obs[2]) * 0.1)
        holds_list.append(holds)
        errs.append(np.mean(ep_errs) if ep_errs else np.nan)
    return float(np.mean(holds_list)), float(np.nanmean(errs))


def main():
    torch.manual_seed(0)  # v0.1.1：固定初始化与抽样种子（审核 P2）
    np.random.seed(0)
    env = FreeGSNKELinearPosEnv(
        control_coils=["P6", "D5", "P5"], max_steps=50, steps_per_action=1,
        max_voltage=500.0, init_disturb_steps=3, init_disturb_voltage=50.0,
    )
    print("采集 PD 专家数据…", flush=True)
    X, Y = collect(env, N_EPISODES)
    print(f"样本数: {len(X)}", flush=True)

    policy = BCPolicy(env.observation_space.shape[0], env.action_space.shape[0]).to(DEVICE)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=800)
    loss_fn = nn.MSELoss()
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    for epoch in range(800):
        idx = torch.randperm(len(Xt))[:1024]
        loss = loss_fn(policy(Xt[idx]), Yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (epoch + 1) % 200 == 0:
            print(f"epoch {epoch+1}: loss={loss.item():.5f}", flush=True)

    holds, err = evaluate(env, policy)
    print(f"BC 策略评估: 存活 {holds:.0f}/50 步, 平均偏差 {err*100:.2f}cm（PD 专家参考: 50 步, ~1.3-2.1cm）")
    torch.save(policy.state_dict(), os.path.join(OUT, "bc_policy.pt"))
    print(f"权重已存 {OUT}/bc_policy.pt", flush=True)


if __name__ == "__main__":
    main()
