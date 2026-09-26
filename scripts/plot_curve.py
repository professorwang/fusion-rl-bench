"""plot_curve.py — 画 PPO 训练曲线（train_log.csv → training_curve.png）"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "runs")
log = os.path.join(OUT, "train_log.csv")

rows = list(csv.DictReader(open(log, encoding="utf-8")))
steps = [int(r["steps"]) for r in rows]
pos_err = [float(r["final_pos_err_mean"]) * 100 for r in rows]
fail = [float(r["fail_rate"]) * 100 for r in rows]
reward = [float(r["eval_reward_mean"]) for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120)

ax1.plot(steps, pos_err, "o-", color="#0e4d92", label="final position error (PPO)")
ax1.set_xlabel("training steps")
ax1.set_ylabel("position error [cm]")
ax1.set_title("Position-hold task: Zcur+Rin error vs training")
ax1.legend()
ax1.grid(alpha=0.3)

ax2r = ax2.twinx()
ax2.plot(steps, fail, "s-", color="#b3540a", label="failure rate [%]")
ax2r.plot(steps, reward, "^-", color="#2b7a2b", label="eval reward", alpha=0.7)
ax2.set_xlabel("training steps")
ax2.set_ylabel("failure rate [%]", color="#b3540a")
ax2r.set_ylabel("reward", color="#2b7a2b")
ax2.set_ylim(-5, 105)
ax2.set_title("Failure rate & eval reward")
ax2.grid(alpha=0.3)

fig.suptitle("fusion-rl-bench v0.4 — PPO on FreeGSNKE linear-evolution position control (MAST-U)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "training_curve.png"))
print("saved:", os.path.join(OUT, "training_curve.png"))
