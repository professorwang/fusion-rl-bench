"""replot_b6.py — 重画 B6 Part 1 图（assets/lqg_eval.png）

原因（诚实记录）：首轮运行中轨迹记录器未按回合隔离，assets/lqg_eval.png 右面板
把 10 个回合的估计轨迹拼接成一条（bug 已修：lqg_baseline.py 的 record_first 逻辑）。
本脚本不重跑模拟器，直接从已冻结的 JSON 重画：
- 左面板：results/lqg_eval.json 的逐回合汇总（失败回合画斜纹占位柱）；
- 右面板：results/model_platform_check.json 的诊断轨迹——同一 LQI 增益在
  本机模拟器上 3 步打飞等离子体 vs 开环自由发散 vs 模型内闭环稳定，
  这比估计轨迹更直接地回答"Part 1 为什么全军覆没"。
（仓库惯例参照 scripts/replot_b3b.py：图可从冻结 JSON 确定性重生成。）
"""
import json
import os

import numpy as np

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "results")
ASSETS = os.path.join(HERE, "..", "assets")


def main():
    ev = json.load(open(os.path.join(OUT, "lqg_eval.json")))
    diag = json.load(open(os.path.join(OUT, "model_platform_check.json")))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=120)

    names = list(ev["results"].keys())
    maes = [ev["results"][n]["summary"]["mae_z_cm"] for n in names]
    completes = [ev["results"][n]["summary"]["complete"] for n in names]
    colors = ["#5a6b7f", "#8a97a8", "#2b7a2b", "#1f8a4c", "#c08030", "#b04030"]
    ok_maes = [v for v, c in zip(maes, completes) if c > 0]
    fail_h = max(ok_maes) * 3
    for i, (n, v, c) in enumerate(zip(names, maes, completes)):
        if c > 0:
            ax1.bar(i, v, color=colors[i % len(colors)], width=0.65)
            ax1.text(i, v * 1.12, f"{v:.2f}\n({c}/10)", ha="center", fontsize=9)
        else:
            ax1.bar(i, fail_h, color="#d08080", hatch="//", width=0.65, alpha=0.6)
            ax1.text(i, fail_h * 1.12, f"失败\n({c}/10)", ha="center", fontsize=9, color="#a03030")
    ax1.set_ylim(min(ok_maes) / 5, fail_h * 6)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, fontsize=8.5, rotation=18, ha="right")
    ax1.set_ylabel("全过程垂直 MAE [cm]（对数轴）")
    ax1.set_title("B6 Part 1（预登记原配置，Windows 冻结模型）：模型驱动控制器全部失败")
    ax1.grid(axis="y", alpha=0.3)

    dt_ms = 0.5
    ol = np.array(diag["openloop_z_m"]) * 100
    lqi = np.array([t["z_err_m"] for t in diag["lqi_trace"]]) * 100
    mod = np.array(diag["model_closedloop_z_m"]) * 100
    ax2.plot(np.arange(len(ol)) * dt_ms, ol, "k-o", ms=4, lw=1.5,
             label="开环零作用（自然发散）")
    ax2.plot(np.arange(len(lqi)) * dt_ms, lqi, "r-s", ms=5, lw=2,
             label="LQI 全状态 × 本机模拟器（3 步打飞）")
    ax2.plot(np.arange(len(mod)) * dt_ms, mod, "g--^", ms=4, lw=1.5,
             label="同一增益 × 辨识模型内闭环（稳定）")
    ax2.axhline(-10, color="gray", ls=":", lw=1)
    ax2.axhline(10, color="gray", ls=":", lw=1)
    ax2.text(0.1, -11.6, "±10cm 丢失线", fontsize=8, color="gray")
    ax2.set_xlabel("时间 [ms]（seed 3000）")
    ax2.set_ylabel("Zcur 偏差 [cm]")
    ax2.set_title("失败根因：模型-对象失配（跨平台漂移）使反馈增益主动失稳")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "lqg_eval.png"))
    print("图已重画 assets/lqg_eval.png", flush=True)


if __name__ == "__main__":
    main()
