"""replot_b3b.py — 一次性字体修复：从已有结果 JSON 重绘 B3b 图（macOS 中文缺字形）

不重跑任何实验，只读 results/ablation_horizon.json / ablation_transfer.json
重画 assets/ablation_horizon.png / ablation_transfer.png（Windows 上中文字体正常，
macOS 需显式指定 PingFang SC；对应脚本内的绘图段已同步加 rcParams，本脚本仅为
不重跑实验的补救）。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPO = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(REPO, "results")
ASSETS = os.path.join(REPO, "assets")

# --- horizon ---
with open(os.path.join(OUT, "ablation_horizon.json"), encoding="utf-8") as f:
    results = json.load(f)["results"]
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
plt.close(fig)

# --- transfer ---
with open(os.path.join(OUT, "ablation_transfer.json"), encoding="utf-8") as f:
    results = json.load(f)["results"]
names = list(results.keys())
vals = [results[n]["summary"]["mae_z_cm"] for n in names]
colors = ["#4a7fb5", "#7fb069", "#c0392b"]
fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=120)
bars = ax.bar(np.arange(len(names)), vals, 0.5, color=colors[:len(names)])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
ax.axhline(0.014, color="gray", ls="--", lw=1, label="PD(3,0.3) 参考 0.014cm")
ax.set_xticks(np.arange(len(names)), names, rotation=10, ha="right")
ax.set_ylabel("whole-trajectory |Z-Z*| MAE [cm]")
ax.set_title("B3b ablation: BC → PPO transfer attribution (3 versions, paired 10 ICs)\n(pre-registered H5a/b/c; linear-head base, 16000-step budget)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "ablation_transfer.png"))
plt.close(fig)
print("两张图已重绘（CJK 字体修复）")
