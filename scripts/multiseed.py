"""multiseed.py — B4：多训练种子统计（投稿前硬门槛）

设计（按阶段 B 计划 B4）：
- 5 个独立训练种子（BC 与 BC+PPO 各 5 个）：专家数据采集 RNG、BC 初始化、PPO 均独立播种
- 教师固定 PD(100,100) + 专家噪声 0.05（主结果配置）；PPO 微调 8000 步（收敛证据见 W06）
- 测试集：30 个冻结初始条件（种子 4000–4029，与训练/调参所用 3000 系完全分离）
- 参照：PD(3,0.3)（确定性，同 30 IC）
- 统计：每控制器跨种子均值±标准差+95% CI；对测试 IC 做配对比较（符号检验+配对 t）

输出：results/multiseed_b4.json + assets/multiseed_b4.png
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
from fair_eval import PDController, run_episode, summarize  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs", "multiseed")
for d in (OUT, ASSETS, RUNS):
    os.makedirs(d, exist_ok=True)

TRAIN_SEEDS = [1, 2, 3, 4, 5]
TEST_SEEDS = list(range(4000, 4030))  # 30 个冻结测试 IC
PPO_STEPS = 8000
TEACHER_KP, TEACHER_KD, NOISE = 100.0, 100.0, 0.05


def pd_action(env, obs, z_prev, p6_idx, rng):
    z_err = obs[2]
    z_dot = (obs[2] - z_prev) / (env.full_timestep * env.steps_per_action)
    a = np.zeros(env.action_space.shape)
    a[p6_idx] = np.clip(-(TEACHER_KP * z_err + TEACHER_KD * z_dot * 1e-3), -1, 1)
    a = a + rng.normal(0, NOISE, size=a.shape)
    return np.clip(a, -1, 1), obs[2]


def collect(env, seed, n_episodes=150):
    p6_idx = env.control_coils.index("P6")
    rng = np.random.default_rng(1000 + seed)
    X, Y = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 100 + ep)
        z_prev = obs[2]
        done = False
        while not done:
            a, z_prev = pd_action(env, obs, z_prev, p6_idx, rng)
            X.append(obs.copy())
            Y.append(a.copy())
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def train_bc(X, Y, obs_dim, act_dim, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = BCPolicy(obs_dim, act_dim)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=800)
    loss_fn = torch.nn.MSELoss()
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    for epoch in range(800):
        idx = torch.randperm(len(Xt))[:1024]
        loss = loss_fn(policy(Xt[idx]), Yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return policy


class PolicyCtrl:
    def __init__(self, policy):
        self.policy = policy

    def reset(self, obs):
        pass

    def act(self, obs):
        with torch.no_grad():
            return np.clip(self.policy(torch.as_tensor(obs, dtype=torch.float32)).numpy(), -1, 1)


class PPOCtrl:
    def __init__(self, model):
        self.model = model

    def reset(self, obs):
        pass

    def act(self, obs):
        a, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(a, dtype=np.float64)


def eval_controller(env, ctrl, seeds):
    rows = [run_episode(env, ctrl, s) for s in seeds]
    return rows, summarize("x", rows)


def main():
    env = FreeGSNKELinearPosEnv(
        control_coils=["P6", "D5", "P5"], max_steps=50, steps_per_action=1,
        max_voltage=500.0, init_disturb_steps=3, init_disturb_voltage=50.0,
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # 参照：低增益最优 PD（确定性，无种子方差）
    dt = env.full_timestep * env.steps_per_action
    p6_idx = env.control_coils.index("P6")
    pd_rows, pd_sum = eval_controller(env, PDController(3.0, 0.3, p6_idx, dt), TEST_SEEDS)
    print(f"参照 PD(3,0.3): MAE_Z {pd_sum['mae_z_cm']:.3f}cm, 完整 {pd_sum['complete']}/{pd_sum['valid_samples']}", flush=True)

    bc_all, ppo_all = {}, {}
    for seed in TRAIN_SEEDS:
        print(f"=== 种子 {seed} ===", flush=True)
        X, Y = collect(env, seed)
        bc = train_bc(X, Y, obs_dim, act_dim, seed)
        torch.save(bc.state_dict(), os.path.join(RUNS, f"bc_seed{seed}.pt"))
        rows, summ = eval_controller(env, PolicyCtrl(bc), TEST_SEEDS)
        bc_all[f"seed{seed}"] = {"episodes": rows, "summary": summ}
        print(f"  BC: MAE_Z {summ['mae_z_cm']:.3f}cm", flush=True)

        # PPO：BC 权重迁移（线性头，迁移等价已验证）+ 独立种子
        ppo = PPO("MlpPolicy", env, verbose=0, seed=seed,
                  n_steps=128, batch_size=64, n_epochs=8,
                  learning_rate=3e-4, gamma=0.99, ent_coef=0.01,
                  policy_kwargs=dict(net_arch=[64, 64]))
        pnet, anet = ppo.policy.mlp_extractor.policy_net, ppo.policy.action_net
        sd = bc.state_dict()
        with torch.no_grad():
            pnet[0].weight.copy_(sd["net.0.weight"]); pnet[0].bias.copy_(sd["net.0.bias"])
            pnet[2].weight.copy_(sd["net.2.weight"]); pnet[2].bias.copy_(sd["net.2.bias"])
            anet.weight.copy_(sd["net.4.weight"]); anet.bias.copy_(sd["net.4.bias"])
            ppo.policy.log_std.fill_(-1.0)
        ppo.learn(total_timesteps=PPO_STEPS)
        ppo.save(os.path.join(RUNS, f"ppo_seed{seed}"))
        rows, summ = eval_controller(env, PPOCtrl(ppo), TEST_SEEDS)
        ppo_all[f"seed{seed}"] = {"episodes": rows, "summary": summ}
        print(f"  PPO: MAE_Z {summ['mae_z_cm']:.3f}cm", flush=True)

    # ---------- 统计 ----------
    def per_ic_mae(rows):
        return np.array([r["mae_z_cm"] if r["mae_z_cm"] is not None else np.nan for r in rows])

    stats = {"PD(3,0.3)": {"per_ic": per_ic_mae(pd_rows).tolist(), "note": "确定性控制器"}}
    for label, group in [("BC", bc_all), ("BC+PPO", ppo_all)]:
        per_seed_mean = [np.nanmean(per_ic_mae(group[s]["episodes"])) for s in group]
        per_ic_all = {s: per_ic_mae(group[s]["episodes"]) for s in group}
        stats[label] = {
            "per_seed_mean_cm": [float(v) for v in per_seed_mean],
            "mean_cm": float(np.mean(per_seed_mean)),
            "std_cm": float(np.std(per_seed_mean, ddof=1)),
            "ci95_cm": float(1.96 * np.std(per_seed_mean, ddof=1) / np.sqrt(len(per_seed_mean))),
        }
        # 配对比较：逐 IC 平均（跨种子）与 PD 的差
        diffs = []
        for i in range(len(TEST_SEEDS)):
            v = np.nanmean([per_ic_all[s][i] for s in per_ic_all])
            p = per_ic_mae(pd_rows)[i]
            if np.isfinite(v) and np.isfinite(p):
                diffs.append(v - p)
        diffs = np.array(diffs)
        stats[label]["paired_vs_PD"] = {
            "mean_diff_cm": float(np.mean(diffs)),
            "sign_test_wins": int(np.sum(diffs < 0)),
            "sign_test_total": int(len(diffs)),
            "ci95_diff": float(1.96 * np.std(diffs, ddof=1) / np.sqrt(len(diffs))),
        }

    payload = {"protocol": {"train_seeds": TRAIN_SEEDS, "test_seeds": TEST_SEEDS,
                            "teacher": f"PD({TEACHER_KP},{TEACHER_KD})+noise{NOISE}",
                            "ppo_steps": PPO_STEPS, "sampling": "post-action",
                            "note": "测试集与训练/调参（3000 系）完全分离"},
               "stats": stats,
               "bc": bc_all, "ppo": ppo_all,
               "pd": {"episodes": pd_rows, "summary": pd_sum}}
    with open(os.path.join(OUT, "multiseed_b4.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=float)

    # ---------- 图 ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    labels = ["PD(3,0.3)", "BC", "BC+PPO"]
    means = [stats["PD(3,0.3)"]["per_ic"] and np.mean(stats["PD(3,0.3)"]["per_ic"]),
             stats["BC"]["mean_cm"], stats["BC+PPO"]["mean_cm"]]
    ax.axhline(means[0], color="#5a6b7f", ls="--", label=f"PD(3,0.3) = {means[0]:.3f}cm")
    for i, label in enumerate(["BC", "BC+PPO"]):
        st = stats[label]
        xs = np.full(len(st["per_seed_mean_cm"]), i + 1)
        ax.scatter(xs, st["per_seed_mean_cm"], color="#0e4d92", zorder=3)
        ax.errorbar(i + 1, st["mean_cm"], yerr=st["ci95_cm"], fmt="s", color="#b3540a",
                    capsize=6, ms=8, label=f"{label} mean±95%CI")
    ax.set_xticks([0, 1, 2], labels)
    ax.set_ylabel("test MAE_Z [cm] (30 frozen ICs)")
    ax.set_title("B4: multi-seed statistics (5 independent training seeds)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "multiseed_b4.png"))

    print("\n=== 统计汇总 ===", flush=True)
    for label in ["BC", "BC+PPO"]:
        st = stats[label]
        pv = st["paired_vs_PD"]
        print(f"{label}: {st['mean_cm']:.3f}±{st['std_cm']:.3f}cm (95%CI ±{st['ci95_cm']:.3f})；"
              f"对PD配对：均值差 {pv['mean_diff_cm']:+.3f}cm，胜率 {pv['sign_test_wins']}/{pv['sign_test_total']}，"
              f"CI95 ±{pv['ci95_diff']:.3f}", flush=True)
    print(f"图已存 {ASSETS}/multiseed_b4.png", flush=True)


if __name__ == "__main__":
    main()
