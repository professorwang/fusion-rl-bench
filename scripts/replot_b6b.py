"""replot_b6b.py — 重画 B6b 图（assets/lqg_b6b.png）

原因（诚实记录）：首版图 (b) 面板中 LQG-LQI 存活 MAE 的数值标注（~3.9cm 处）
与面板标题重叠。本脚本不重跑模拟器，直接从已冻结的 results/lqg_b6b.json
重画（仓库惯例参照 scripts/replot_b3b.py / replot_b6.py）：
标注改放到数据点下方，(b) 面板 y 轴留出顶部余量。数据本身未受影响。
"""
import json
import os

import numpy as np

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "results")
ASSETS = os.path.join(HERE, "..", "assets")

SIGMAS_MM = [0.0, 0.1, 1.0, 10.0]


def main():
    results = json.load(open(os.path.join(OUT, "lqg_b6b.json"), encoding="utf-8"))["results"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=120)
    xs = np.arange(len(SIGMAS_MM))
    xticklabels = [f"{s:g}" for s in SIGMAS_MM]

    # (a) 完整率 vs σ
    for label, prefix, color in [("LQG-LQI（KF+部分观测）", "LQG-LQI", "#1f8a4c"),
                                 ("PD(3,0.3)", "PD(3,0.3)", "#5a6b7f")]:
        comps = [results[f"{prefix},σ={s:g}mm"]["summary"]["complete"] for s in SIGMAS_MM]
        ax1.plot(xs, comps, "o-", color=color, lw=2, label=label)
    ax1.set_xticks(xs, xticklabels)
    ax1.set_xlabel("测量噪声 σ [mm]（等距网格，含 0）")
    ax1.set_ylabel("完整回合数 / 10")
    ax1.set_ylim(-0.5, 10.9)
    ax1.set_title("B6b-1: 观测完备性悬崖——完整率 vs 噪声幅值")
    ax1.legend(fontsize=9, loc="center left")
    ax1.grid(alpha=0.3)

    # (b) MAE vs σ（存活回合均值，标注 n；标注放数据点下方防标题碰撞）
    for label, prefix, color in [("LQG-LQI（存活回合）", "LQG-LQI", "#1f8a4c"),
                                 ("PD(3,0.3)", "PD(3,0.3)", "#5a6b7f")]:
        maes, ns = [], []
        for s in SIGMAS_MM:
            summ = results[f"{prefix},σ={s:g}mm"]["summary"]
            maes.append(summ["mae_z_cm"])
            ns.append(summ["complete"])
        ax2.plot(xs, maes, "o-", color=color, lw=2, label=label)
        for x, v, n in zip(xs, maes, ns):
            if v == v:  # 非 NaN
                ax2.annotate(f"{v:.2f} (n={n})", (x, v), textcoords="offset points",
                             xytext=(0, -16), ha="center", fontsize=8)
    ref_mae = results["LQI(三通道,全状态)"]["summary"]["mae_z_cm"]
    ax2.axhline(ref_mae, color="#2b7a2b", ls="--", lw=1,
                label=f"全状态 LQI 参照 ({ref_mae:.2f}cm)")
    ax2.set_ylim(5e-3, 30)
    ax2.set_xticks(xs, xticklabels)
    ax2.set_xlabel("测量噪声 σ [mm]")
    ax2.set_ylabel("全过程垂直 MAE [cm]（对数轴）")
    ax2.set_yscale("log")
    ax2.set_title("B6b-1: 存活误差 vs 噪声幅值（失败回合不入均值）")
    ax2.legend(fontsize=9, loc="center left")
    ax2.grid(alpha=0.3)

    # (c) 拉起策略：逐种子 holds 对照（σ=1mm）
    cfgs = [("LQG-LQI（无拉起）", "LQG-LQI,σ=1mm", "#c08030", "o"),
            ("+PD交接 N=5", "LQG-LQI+PD交接(N=5),σ=1mm", "#4a7fb5", "s"),
            ("+PD交接 N=10", "LQG-LQI+PD交接(N=10),σ=1mm", "#1f8a4c", "^")]
    for i, (label, name, color, marker) in enumerate(cfgs):
        holds = [r["holds"] for r in results[name]["episodes"]]
        comp = results[name]["summary"]["complete"]
        ax3.scatter(np.arange(10) + (i - 1) * 0.18, holds, marker=marker, color=color,
                    s=42, label=f"{label}（完整 {comp}/10）", zorder=3)
    ax3.axhline(50, color="k", lw=0.8, ls=":", alpha=0.6)
    ax3.set_xticks(np.arange(10), [f"s{3000 + i}" for i in range(10)], fontsize=8)
    ax3.set_xlabel("配对种子")
    ax3.set_ylabel("存活控制步数 holds（50=完整）")
    ax3.set_title("B6b-2: PD 交接拉起 vs 无拉起（σ=1mm，逐种子）")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "lqg_b6b.png"))
    print("图已重画 assets/lqg_b6b.png", flush=True)


if __name__ == "__main__":
    main()
