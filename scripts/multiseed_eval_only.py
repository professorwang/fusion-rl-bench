"""multiseed_eval_only.py — 加载已保存的多种子模型，仅重跑评估+统计（不重训）

用于恢复 multiseed.py 统计段 NameError 损失的逐回合数据（模型已在 runs/multiseed/）。
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
from multiseed import PolicyCtrl, PPOCtrl, TEST_SEEDS, TRAIN_SEEDS  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs", "multiseed")


def main():
    env = FreeGSNKELinearPosEnv(
        control_coils=["P6", "D5", "P5"], max_steps=50, steps_per_action=1,
        max_voltage=500.0, init_disturb_steps=3, init_disturb_voltage=50.0,
    )
    obs_dim = env.observation_space.shape[0]
    dt = env.full_timestep * env.steps_per_action
    p6_idx = env.control_coils.index("P6")

    pd_rows, pd_sum = None, None
    pd_rows = [run_episode(env, PDController(3.0, 0.3, p6_idx, dt), s) for s in TEST_SEEDS]
    pd_sum = summarize("PD(3,0.3)", pd_rows)
    print(f"参照 PD(3,0.3): {pd_sum['mae_z_cm']:.3f}cm, 完整 {pd_sum['complete']}/{pd_sum['valid_samples']}", flush=True)

    bc_all, ppo_all = {}, {}
    for seed in TRAIN_SEEDS:
        bc = BCPolicy(obs_dim, 3)
        bc.load_state_dict(torch.load(os.path.join(RUNS, f"bc_seed{seed}.pt"), map_location="cpu"))
        bc.eval()
        rows, summ = None, None
        rows = [run_episode(env, PolicyCtrl(bc), s) for s in TEST_SEEDS]
        summ = summarize(f"BC_seed{seed}", rows)
        bc_all[f"seed{seed}"] = {"episodes": rows, "summary": summ}

        ppo = PPO.load(os.path.join(RUNS, f"ppo_seed{seed}"))
        rows = [run_episode(env, PPOCtrl(ppo), s) for s in TEST_SEEDS]
        summ = summarize(f"PPO_seed{seed}", rows)
        ppo_all[f"seed{seed}"] = {"episodes": rows, "summary": summ}
        print(f"种子 {seed}: BC {bc_all[f'seed{seed}']['summary']['mae_z_cm']:.3f} | "
              f"PPO {ppo_all[f'seed{seed}']['summary']['mae_z_cm']:.3f}", flush=True)

    # ---------- 统计 ----------
    def per_ic_mae(rows):
        return np.array([r["mae_z_cm"] if r["mae_z_cm"] is not None else np.nan for r in rows])

    pd_ic = per_ic_mae(pd_rows)
    stats = {"PD(3,0.3)": {"per_ic": pd_ic.tolist(), "note": "确定性控制器"}}
    for label, group in [("BC", bc_all), ("BC+PPO", ppo_all)]:
        per_seed_mean = [np.nanmean(per_ic_mae(group[s]["episodes"])) for s in group]
        per_ic_all = {s: per_ic_mae(group[s]["episodes"]) for s in group}
        stats[label] = {
            "per_seed_mean_cm": [float(v) for v in per_seed_mean],
            "mean_cm": float(np.mean(per_seed_mean)),
            "std_cm": float(np.std(per_seed_mean, ddof=1)),
            "ci95_cm": float(1.96 * np.std(per_seed_mean, ddof=1) / np.sqrt(len(per_seed_mean))),
        }
        diffs = []
        for i in range(len(TEST_SEEDS)):
            v = np.nanmean([per_ic_all[s][i] for s in per_ic_all])
            p = pd_ic[i]
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
                            "teacher": "PD(100,100)+noise0.05", "ppo_steps": 8000,
                            "sampling": "post-action",
                            "note": "测试集与训练/调参（3000 系）完全分离；模型为已训练存档的再评估"},
               "stats": stats, "bc": bc_all, "ppo": ppo_all,
               "pd": {"episodes": pd_rows, "summary": pd_sum}}
    with open(os.path.join(OUT, "multiseed_b4.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=float)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    pd_mean = float(np.nanmean(pd_ic))
    ax.axhline(pd_mean, color="#5a6b7f", ls="--", label=f"PD(3,0.3) = {pd_mean:.3f}cm")
    for i, label in enumerate(["BC", "BC+PPO"]):
        st = stats[label]
        xs = np.full(len(st["per_seed_mean_cm"]), i + 1)
        ax.scatter(xs, st["per_seed_mean_cm"], color="#0e4d92", zorder=3, s=40)
        ax.errorbar(i + 1, st["mean_cm"], yerr=st["ci95_cm"], fmt="s", color="#b3540a",
                    capsize=6, ms=8, label=f"{label} mean±95%CI")
    ax.set_xticks([0, 1, 2], ["PD(3,0.3)", "BC", "BC+PPO"])
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
