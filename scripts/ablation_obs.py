"""ablation_obs.py — B3 机制消融（第二批 a）：观测条件（±导数项）

研究问题：观测中显式导数项（dZcur/dt, dRin/dt）是否改变 BC 与经典控制的相对排名？
本轮隔离观测因素（其余固定）：教师 ∈ {PD(3,0.3) 低增益最优, PD(100,100) 高增益振荡}，
专家噪声固定 0.05（v0.1.1 标准配置），评价协议与 ablation_bc.py 完全相同（种子 3000–3009）。

预测（预登记，写于跑实验之前）：
- H3a：无导数观测（obs[:4]）下，高增益教师数据的 BC 显著劣化（教师动作强依赖 ż，
  无记忆策略无法从瞬时观测恢复微分项）——完整回合数下降或 MAE 大幅上升；
- H3b：低增益教师（动作≈纯比例律）下，无导数 BC 劣化幅度更小（交互效应）——
  即观测消融的伤害随教师对导数项的依赖程度递增；
- H3c：6 维观测臂复用第一批已冻结权重（bc_t*_n0.05.pt），不重训，结果应复现第一批数值。

输出：results/ablation_obs.json + assets/ablation_obs.png + runs/ablation/bc4_*.pt
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))
from freegsnke_linear_pos_env import FreeGSNKELinearPosEnv  # noqa: E402
from ablation_bc import collect, train_bc, PolicyCtrl, RUNS_DIR, EVAL_SEEDS  # noqa: E402
from fair_eval import run_episode, summarize  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(OUT, exist_ok=True)
os.makedirs(ASSETS, exist_ok=True)

TEACHERS = [(3.0, 0.3), (100.0, 100.0)]
NOISE = 0.05
N_EPISODES_DATA = 150


class PolicyCtrlNoDot:
    """无导数臂：训练/推理均只用 obs[:4]（去掉 dZcur/dt, dRin/dt）。"""

    def __init__(self, policy):
        self.policy = policy

    def reset(self, obs):
        pass

    def act(self, obs):
        with torch.no_grad():
            a = self.policy(torch.as_tensor(obs[:4], dtype=torch.float32)).numpy()
        return np.clip(a, -1, 1)


def main():
    env = FreeGSNKELinearPosEnv(
        control_coils=["P6", "D5", "P5"], max_steps=50, steps_per_action=1,
        max_voltage=500.0, init_disturb_steps=3, init_disturb_voltage=50.0,
    )
    act_dim = env.action_space.shape[0]

    results = {}
    for kp, kd in TEACHERS:
        name6 = f"BC6d(教师PD({kp},{kd}),noise={NOISE})"
        print(f"=== {name6}（复用第一批冻结权重）===", flush=True)
        from bc_warmup import BCPolicy
        pol6 = BCPolicy(6, act_dim)
        sd = torch.load(os.path.join(RUNS_DIR, f"bc_t{kp}_{kd}_n{NOISE}.pt"), map_location="cpu")
        pol6.load_state_dict(sd)
        pol6.eval()
        rows = [run_episode(env, PolicyCtrl(pol6), s) for s in EVAL_SEEDS]
        summ = summarize(name6, rows)
        results[name6] = {"episodes": rows, "summary": summ}
        print(f"  → 完整 {summ['complete']}/{summ['valid_samples']}, MAE_Z {summ['mae_z_cm']:.3f}cm", flush=True)

        name4 = f"BC4d(教师PD({kp},{kd}),noise={NOISE},无导数)"
        print(f"=== {name4} ===", flush=True)
        X, Y = collect(env, kp, kd, NOISE, N_EPISODES_DATA)
        X4 = X[:, :4].copy()  # 去掉导数通道
        pol4 = BCPolicy(4, act_dim)
        save4 = os.path.join(RUNS_DIR, f"bc4_t{kp}_{kd}_n{NOISE}.pt")
        # 与 ablation_bc.train_bc 同配置，仅输入维度不同
        torch.manual_seed(0)
        np.random.seed(0)
        opt = torch.optim.Adam(pol4.parameters(), lr=3e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=800)
        loss_fn = torch.nn.MSELoss()
        Xt, Yt = torch.as_tensor(X4), torch.as_tensor(Y)
        for epoch in range(800):
            idx = torch.randperm(len(Xt))[:1024]
            loss = loss_fn(pol4(Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        torch.save(pol4.state_dict(), save4)
        rows = [run_episode(env, PolicyCtrlNoDot(pol4), s) for s in EVAL_SEEDS]
        summ = summarize(name4, rows)
        results[name4] = {"episodes": rows, "summary": summ}
        print(f"  → 完整 {summ['complete']}/{summ['valid_samples']}, MAE_Z {summ['mae_z_cm']:.3f}cm", flush=True)

    with open(os.path.join(OUT, "ablation_obs.json"), "w", encoding="utf-8") as f:
        json.dump({"protocol": {"teachers": TEACHERS, "noise": NOISE,
                                "obs_arms": {"6d": "d(4)+d_dot(2)", "4d": "d(4) only"},
                                "eval_seeds": EVAL_SEEDS, "sampling": "post-action",
                                "note": "6d 臂复用 ablation_bc 第一批冻结权重，未重训"},
                   "results": results}, f, ensure_ascii=False, indent=2, default=float)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x_labels = [f"PD({int(kp)},{kd:g})" for kp, kd in TEACHERS]
    xs = np.arange(len(TEACHERS))
    width = 0.35
    arms = [
        ("BC6d", "", "BC obs=d+d_dot (frozen batch-1)", "#4a7fb5"),
        ("BC4d", ",无导数", "BC obs=d only (no derivative)", "#c0392b"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    for k, (prefix, suffix, label, color) in enumerate(arms):
        vals = [results[f"{prefix}(教师PD({kp},{kd}),noise={NOISE}{suffix})"]["summary"]["mae_z_cm"]
                for kp, kd in TEACHERS]
        ax.bar(xs + (k - 0.5) * width, vals, width, label=label, color=color)
    ax.set_xticks(xs, x_labels)
    ax.set_ylabel("whole-trajectory |Z-Z*| MAE [cm]")
    ax.set_title("B3b ablation: observation (± derivative) × teacher gain → BC performance\n(pre-registered H3a/b/c; paired eval, 10 ICs, noise=0.05)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "ablation_obs.png"))
    print("图已存 assets/ablation_obs.png", flush=True)


if __name__ == "__main__":
    main()
