"""ablation_bc.py — B3 机制消融（第一批）：教师增益 × 专家噪声

研究问题：什么因素改变学习控制器与经典控制的相对排名？
本轮隔离两个因素（其余固定）：
- 教师增益：PD(100,100)（高增益振荡）/ PD(10,3)（中）/ PD(3,0.3)（低增益最优 0.014cm）
- 专家噪声：0.0 / 0.05（动作叠加高斯噪声——检验"噪声收缩→平滑化"假说 H1）

预测（预登记）：
- H1：若"噪声收缩"成立，BC(高增益教师, noise=0) ≈ 教师水平（2.42cm），
  而 BC(高增益教师, noise=0.05) 显著更低（≈0.2cm）；
- H2：BC(低增益教师) ≈ 教师水平（0.014–0.02cm），无论噪声。

每组：150 episode 采集 + BC 训练（与 v0.1.1 同配置）+ fair_eval 协议评估（种子 3000–3009）。
输出：results/ablation_bc.json + assets/ablation_bc.png
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))
from freegsnke_linear_pos_env import FreeGSNKELinearPosEnv  # noqa: E402
from bc_warmup import BCPolicy  # noqa: E402
from fair_eval import run_episode, summarize  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(OUT, exist_ok=True)
os.makedirs(ASSETS, exist_ok=True)

TEACHERS = [(100.0, 100.0), (10.0, 3.0), (3.0, 0.3)]
NOISES = [0.0, 0.05]
N_EPISODES_DATA = 150
EPOCHS = 800
EVAL_SEEDS = list(range(3000, 3010))
RUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "ablation")
os.makedirs(RUNS_DIR, exist_ok=True)


def pd_action(env, obs, z_prev, p6_idx, kp, kd, noise, rng):
    z_err = obs[2]
    z_dot = (obs[2] - z_prev) / (env.full_timestep * env.steps_per_action)
    a = np.zeros(env.action_space.shape)
    a[p6_idx] = np.clip(-(kp * z_err + kd * z_dot * 1e-3), -1, 1)
    if noise > 0:
        a = a + rng.normal(0, noise, size=a.shape)
    return np.clip(a, -1, 1), obs[2]


def collect(env, kp, kd, noise, n_episodes):
    p6_idx = env.control_coils.index("P6")
    rng = np.random.default_rng(0)
    X, Y = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        z_prev = obs[2]
        done = False
        while not done:
            a, z_prev = pd_action(env, obs, z_prev, p6_idx, kp, kd, noise, rng)
            X.append(obs.copy())
            Y.append(a.copy())
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def train_bc(X, Y, obs_dim, act_dim, save_path):
    torch.manual_seed(0)
    np.random.seed(0)
    policy = BCPolicy(obs_dim, act_dim)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = torch.nn.MSELoss()
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    for epoch in range(EPOCHS):
        idx = torch.randperm(len(Xt))[:1024]
        loss = loss_fn(policy(Xt[idx]), Yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    torch.save(policy.state_dict(), save_path)
    return policy


class PolicyCtrl:
    def __init__(self, policy):
        self.policy = policy

    def reset(self, obs):
        pass

    def act(self, obs):
        with torch.no_grad():
            return np.clip(self.policy(torch.as_tensor(obs, dtype=torch.float32)).numpy(), -1, 1)


def main():
    env = FreeGSNKELinearPosEnv(
        control_coils=["P6", "D5", "P5"], max_steps=50, steps_per_action=1,
        max_voltage=500.0, init_disturb_steps=3, init_disturb_voltage=50.0,
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    results = {}
    for kp, kd in TEACHERS:
        for noise in NOISES:
            name = f"BC(教师PD({kp},{kd}),noise={noise})"
            print(f"=== {name} ===", flush=True)
            X, Y = collect(env, kp, kd, noise, N_EPISODES_DATA)
            policy = train_bc(X, Y, obs_dim, act_dim,
                              os.path.join(RUNS_DIR, f"bc_t{kp}_{kd}_n{noise}.pt"))
            rows = [run_episode(env, PolicyCtrl(policy), s) for s in EVAL_SEEDS]
            summ = summarize(name, rows)
            results[name] = {"episodes": rows, "summary": summ}
            print(f"  → 完整 {summ['complete']}/{summ['valid_samples']}, "
                  f"MAE_Z {summ['mae_z_cm']:.3f}cm", flush=True)

    with open(os.path.join(OUT, "ablation_bc.json"), "w", encoding="utf-8") as f:
        json.dump({"protocol": {"teachers": TEACHERS, "noises": NOISES,
                                "eval_seeds": EVAL_SEEDS, "sampling": "post-action"},
                   "results": results}, f, ensure_ascii=False, indent=2, default=float)

    # 图：教师增益 × 噪声 → BC 误差（教师参考线标注）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    teacher_ref = {(100.0, 100.0): 2.42, (10.0, 3.0): 1.44, (3.0, 0.3): 0.014}
    x_labels = [f"PD({int(kp)},{kd:g})" for kp, kd in TEACHERS]
    xs = np.arange(len(TEACHERS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    for k, noise in enumerate(NOISES):
        vals = [results[f"BC(教师PD({kp},{kd}),noise={noise})"]["summary"]["mae_z_cm"] for kp, kd in TEACHERS]
        ax.bar(xs + (k - 0.5) * width, vals, width, label=f"BC from noisy={noise} data",
               color=["#4a7fb5", "#0e4d92"][k])
    ax.plot(xs - width / 2, [teacher_ref[t] for t in TEACHERS], "rD--", ms=6, lw=1,
            label="teacher PD itself")
    ax.set_xticks(xs, x_labels)
    ax.set_ylabel("whole-trajectory |Z-Z*| MAE [cm]")
    ax.set_title("B3 ablation: teacher gain × expert noise → BC performance\n(pre-registered hypotheses H1/H2; paired eval, 10 ICs)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "ablation_bc.png"))
    print("图已存 assets/ablation_bc.png", flush=True)


if __name__ == "__main__":
    main()
