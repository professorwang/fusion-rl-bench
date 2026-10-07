"""ablation_transfer.py — B3 机制消融（第二批 c）：BC→PPO 迁移归因三版本对照

研究问题：BC→PPO 的"微调贡献"到底来自迁移本身还是微调训练？
三版本对照（同一 BC 基座：runs/ablation/bc_t100.0_100.0_n0.05.pt，线性输出头）：
- v1 BC 直接评估；
- v2 迁移未微调：BC 权重按 train_ppo.py 映射拷入 SB3 PPO MlpPolicy（log_std=-1），
  deterministic 预测、零训练步评估——检验线性头迁移的函数等价性（W06 实测差异 0.0 的复核）；
- v3 微调后：同一初始化起 PPO 训练 2×8000=16000 步（与 v0.1.1 冻结模型 16128 步同量级预算，
  差异 128 步=1 个 rollout，如实记录），评估 ppo_final。

预测（预登记，写于跑实验之前）：
- H5a：v2 ≈ v1（线性头迁移函数等价，差异应≈0；若 ≠0 则说明 SB3 内部还有未记录的处理）；
- H5b：v3 不会显著优于 v1（B4 已显示 PPO 配对胜率 0/30 vs PD、种子间方差大）——
  若 v3 显著更差，则"RL 微调贡献"在本任务上为负，需如实报告；
- H5c：v3 与 v0.1.1 冻结 PPO（旧 Tanh 头迁移，1.27cm）的差异用于量化"迁移映射"因素——
  两者训练预算相同，唯一差别是基座头型（线性 vs Tanh）。

用法：
  python scripts/ablation_transfer.py --stage eval_v1v2   # 采集 BC/迁移未微调评估
  python scripts/ablation_transfer.py --stage train       # PPO 微调（两段 8000 步，可后台）
  python scripts/ablation_transfer.py --stage eval_v3     # 微调后评估 + 出图
输出：results/ablation_transfer.json + assets/ablation_transfer.png + runs/transfer/
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))
from freegsnke_linear_pos_env import FreeGSNKELinearPosEnv  # noqa: E402
from ablation_bc import PolicyCtrl, RUNS_DIR, EVAL_SEEDS  # noqa: E402
from bc_warmup import BCPolicy  # noqa: E402
from fair_eval import run_episode, summarize  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(REPO, "results")
ASSETS = os.path.join(REPO, "assets")
TRANSFER_DIR = os.path.join(REPO, "runs", "transfer")
os.makedirs(OUT, exist_ok=True)
os.makedirs(ASSETS, exist_ok=True)
os.makedirs(TRANSFER_DIR, exist_ok=True)

BC_BASE = os.path.join(RUNS_DIR, "bc_t100.0_100.0_n0.05.pt")
JSON_PATH = os.path.join(OUT, "ablation_transfer.json")
# 与 v0.1.1 冻结 PPO 同量级预算：16128 步 = 126 rollout × 128；两段 8000 步 = 16000（差 1 rollout）
TRAIN_BUDGET_NOTE = "2x8000=16000 steps vs frozen 16128 (diff 128 = 1 rollout)"


def make_env():
    return FreeGSNKELinearPosEnv(
        control_coils=["P6", "D5", "P5"], max_steps=50, steps_per_action=1,
        max_voltage=500.0, init_disturb_steps=3, init_disturb_voltage=50.0,
    )


class PPOTransferredCtrl:
    """v2：BC 权重拷入 SB3 PPO 策略（不训练），deterministic 预测。"""

    def __init__(self, env, bc_path):
        from stable_baselines3 import PPO
        model = PPO(
            "MlpPolicy", env, verbose=0, seed=0,
            n_steps=128, batch_size=64, n_epochs=8,
            learning_rate=3e-4, gamma=0.99, ent_coef=0.01,
            policy_kwargs=dict(net_arch=[64, 64]),
        )
        sd = torch.load(bc_path, map_location="cpu")
        pnet = model.policy.mlp_extractor.policy_net
        anet = model.policy.action_net
        with torch.no_grad():
            pnet[0].weight.copy_(sd["net.0.weight"]); pnet[0].bias.copy_(sd["net.0.bias"])
            pnet[2].weight.copy_(sd["net.2.weight"]); pnet[2].bias.copy_(sd["net.2.bias"])
            anet.weight.copy_(sd["net.4.weight"]); anet.bias.copy_(sd["net.4.bias"])
            model.policy.log_std.fill_(-1.0)
        self.model = model

    def reset(self, obs):
        pass

    def act(self, obs):
        a, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(a, dtype=np.float64)


def load_json():
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"protocol": {"bc_base": "runs/ablation/bc_t100.0_100.0_n0.05.pt (线性头)",
                         "transfer_mapping": "net.0/2→policy_net, net.4→action_net, log_std=-1",
                         "train_budget": TRAIN_BUDGET_NOTE,
                         "eval_seeds": EVAL_SEEDS, "sampling": "post-action"},
            "results": {}}


def save_json(payload):
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=float)


def eval_arm(env, name, ctrl):
    rows = [run_episode(env, ctrl, s) for s in EVAL_SEEDS]
    summ = summarize(name, rows)
    print(f"  {name:<28} 完整 {summ['complete']}/{summ['valid_samples']}, "
          f"MAE_Z {summ['mae_z_cm']:.3f}cm", flush=True)
    return {"episodes": rows, "summary": summ}


def stage_eval_v1v2():
    env = make_env()
    payload = load_json()
    pol = BCPolicy(6, 3)
    pol.load_state_dict(torch.load(BC_BASE, map_location="cpu"))
    pol.eval()
    payload["results"]["v1-BC(线性头基座)"] = eval_arm(env, "v1-BC(线性头基座)", PolicyCtrl(pol))
    payload["results"]["v2-迁移未微调"] = eval_arm(env, "v2-迁移未微调", PPOTransferredCtrl(env, BC_BASE))
    save_json(payload)
    print("v1/v2 评估完成", flush=True)


def stage_train():
    leg1 = os.path.join(TRANSFER_DIR, "leg1")
    env_vars = dict(os.environ)
    env_vars["RUN_DIR"] = leg1
    env_vars["BC_INIT"] = BC_BASE
    print("=== 第 1 段：BC 初始化 PPO 8000 步 ===", flush=True)
    subprocess.run([sys.executable, os.path.join(REPO, "scripts", "train_ppo.py")],
                   check=True, env=env_vars)
    leg2 = os.path.join(TRANSFER_DIR, "leg2")
    env_vars["RUN_DIR"] = leg2
    env_vars["RESUME"] = os.path.join(leg1, "ppo_final")
    env_vars.pop("BC_INIT", None)
    print("=== 第 2 段：续训 8000 步（共 16000）===", flush=True)
    subprocess.run([sys.executable, os.path.join(REPO, "scripts", "train_ppo.py")],
                   check=True, env=env_vars)
    print("微调完成", flush=True)


def stage_eval_v3():
    env = make_env()
    payload = load_json()
    from fair_eval import PPOController
    final = os.path.join(TRANSFER_DIR, "leg2", "ppo_final.zip")
    payload["results"]["v3-微调后(16000步)"] = eval_arm(
        env, "v3-微调后(16000步)", PPOController(final))
    save_json(payload)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    names = list(payload["results"].keys())
    vals = [payload["results"][n]["summary"]["mae_z_cm"] for n in names]
    colors = ["#4a7fb5", "#7fb069", "#c0392b"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=120)
    bars = ax.bar(np.arange(len(names)), vals, 0.5, color=colors[:len(names)])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.014, color="gray", ls="--", lw=1, label="PD(3,0.3) 参考 0.014cm")
    ax.set_xticks(np.arange(len(names)), names, rotation=10, ha="right")
    ax.set_ylabel("whole-trajectory |Z-Z*| MAE [cm]")
    ax.set_title("B3b ablation: BC → PPO transfer attribution (3 versions, paired 10 ICs)\n(pre-registered H5a/b/c; linear-head base, 16000-step budget)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "ablation_transfer.png"))
    print("图已存 assets/ablation_transfer.png", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["eval_v1v2", "train", "eval_v3", "all"], default="all")
    args = ap.parse_args()
    if args.stage in ("eval_v1v2", "all"):
        stage_eval_v1v2()
    if args.stage in ("train", "all"):
        stage_train()
    if args.stage in ("eval_v3", "all"):
        stage_eval_v3()
