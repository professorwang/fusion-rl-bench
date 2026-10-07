"""ablation_horizon.py — B3 机制消融（第二批 b）：评价时域（25ms vs 100ms）

研究问题：评价时域拉长是否改变学习控制器与经典控制的相对排名？
本轮隔离评价时域：控制器全部冻结（不重训），仅改变回合长度
（max_steps 50 → 25ms / max_steps 200 → 100ms，控制周期 0.5ms 不变）。
配对协议不变：种子 3000–3009、同扰动、同模拟器实例、动作后采样。

预测（预登记，写于跑实验之前）：
- H4a：闭环稳定的控制器（PD(3,0.3)、低增益教师 BC）在 100ms 下 MAE 持平或下降
  （初始扰动瞬态被更长的稳态段摊薄），不出现新发失败回合；
- H4b：高增益教师 BC 与 PPO 在长尾暴露更多漂移/失败（PPO 种子间方差大，B4 已记录），
  排名随时域拉长向经典控制倾斜——即"学习方法优势"在 25ms 评价下被高估；
- H4c：PD(100,100) 极限环在 100ms 下 MAE 基本不变（周期稳态，非瞬态）。

控制器（全部冻结）：
- PD(3,0.3) / PD(100,100)（P6 单通道）
- BC 低增益教师臂 runs/ablation/bc_t3.0_0.3_n0.05.pt
- BC 高增益教师臂 runs/ablation/bc_t100.0_100.0_n0.05.pt
- PPO 冻结模型 runs/ppo_final.zip（初始化自旧版 Tanh 头 BC——归因限定沿用 W06 注记）

输出：results/ablation_horizon.json + assets/ablation_horizon.png
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))
from freegsnke_linear_pos_env import FreeGSNKELinearPosEnv  # noqa: E402
from ablation_bc import PolicyCtrl, RUNS_DIR, EVAL_SEEDS  # noqa: E402
from bc_warmup import BCPolicy  # noqa: E402
from fair_eval import PDController, run_episode, summarize  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
REPO = os.path.join(os.path.dirname(__file__), "..")
os.makedirs(OUT, exist_ok=True)
os.makedirs(ASSETS, exist_ok=True)

HORIZONS = {"25ms": 50, "100ms": 200}


def make_env(max_steps):
    return FreeGSNKELinearPosEnv(
        control_coils=["P6", "D5", "P5"], max_steps=max_steps, steps_per_action=1,
        max_voltage=500.0, init_disturb_steps=3, init_disturb_voltage=50.0,
    )


def make_controllers(env):
    dt = env.full_timestep * env.steps_per_action
    p6_idx = env.control_coils.index("P6")
    ctrls = {
        "PD(3,0.3)": PDController(3.0, 0.3, p6_idx, dt),
        "PD(100,100)": PDController(100.0, 100.0, p6_idx, dt),
    }
    for label, path in [
        ("BC(教师PD(3,0.3))", os.path.join(RUNS_DIR, "bc_t3.0_0.3_n0.05.pt")),
        ("BC(教师PD(100,100))", os.path.join(RUNS_DIR, "bc_t100.0_100.0_n0.05.pt")),
    ]:
        pol = BCPolicy(6, 3)
        pol.load_state_dict(torch.load(path, map_location="cpu"))
        pol.eval()
        ctrls[label] = PolicyCtrl(pol)
    ppo_path = os.path.join(REPO, "runs", "ppo_final.zip")
    if os.path.exists(ppo_path):
        try:
            from fair_eval import PPOController
            ctrls["PPO(冻结,Tanh头迁移)"] = PPOController(ppo_path)
        except ImportError as e:
            print(f"  [跳过 PPO] stable_baselines3 不可用：{e}", flush=True)
    return ctrls


def main():
    results = {}
    for hz, max_steps in HORIZONS.items():
        print(f"===== 评价时域 {hz}（max_steps={max_steps}）=====", flush=True)
        env = make_env(max_steps)
        for name, ctrl in make_controllers(env).items():
            rows = [run_episode(env, ctrl, s, max_steps=max_steps) for s in EVAL_SEEDS]
            summ = summarize(name, rows)
            results[f"{name} @ {hz}"] = {"episodes": rows, "summary": summ}
            print(f"  {name:<24} 完整 {summ['complete']}/{summ['valid_samples']}, "
                  f"MAE_Z {summ['mae_z_cm']:.3f}cm", flush=True)

    with open(os.path.join(OUT, "ablation_horizon.json"), "w", encoding="utf-8") as f:
        json.dump({"protocol": {"horizons": HORIZONS, "eval_seeds": EVAL_SEEDS,
                                "sampling": "post-action, paired seeds",
                                "note": "控制器全部冻结不重训；PPO 归因限定见模块注释"},
                   "results": results}, f, ensure_ascii=False, indent=2, default=float)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    names = [n.split(" @ ")[0] for n in results if n.endswith("@ 25ms")]
    xs = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    for k, (hz, color) in enumerate([("25ms", "#4a7fb5"), ("100ms", "#e67e22")]):
        vals = [results[f"{n} @ {hz}"]["summary"]["mae_z_cm"] for n in names]
        ax.bar(xs + (k - 0.5) * width, vals, width, label=hz, color=color)
        for x, v in zip(xs + (k - 0.5) * width, vals):
            ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs, names, rotation=15, ha="right")
    ax.set_ylabel("whole-trajectory |Z-Z*| MAE [cm]")
    ax.set_title("B3b ablation: evaluation horizon 25ms vs 100ms (frozen controllers, paired 10 ICs)\n(pre-registered H4a/b/c)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "ablation_horizon.png"))
    print("图已存 assets/ablation_horizon.png", flush=True)


if __name__ == "__main__":
    main()
