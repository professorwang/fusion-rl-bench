"""pd_gain_surface.py — B2：PD 增益族性能面（预先指定网格）

问题：教师增益如何改变经典控制的性能？（"什么因素改变学习与经典排名"研究的第一块拼图）
设计：Kp × Kd = {0.3,1,3,10,30,100}² 共 36 组，每组 10 个配对 episode（种子 3000–3009），
统一协议（动作后采样、失败分组）。输出性能面 JSON + 热力图。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fair_eval import PDController, make_env, run_episode, summarize  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
KPS = [0.3, 1, 3, 10, 30, 100]
KDS = [0.1, 0.3, 1, 3, 10, 100]


def main():
    env = make_env()
    dt = env.full_timestep * env.steps_per_action
    p6_idx = env.control_coils.index("P6")

    surface = np.full((len(KPS), len(KDS)), np.nan)
    completes = np.zeros((len(KPS), len(KDS)))
    for i, kp in enumerate(KPS):
        for j, kd in enumerate(KDS):
            ctrl = PDController(kp, kd, p6_idx, dt)
            rows = [run_episode(env, ctrl, s) for s in range(3000, 3010)]
            s = summarize(f"PD({kp},{kd})", rows)
            surface[i, j] = s["mae_z_cm"]
            completes[i, j] = s["complete"]
        print(f"Kp={kp} 完成", flush=True)

    payload = {"protocol": {"kps": KPS, "kds": KDS, "seeds": "3000-3009",
                            "sampling": "post-action", "note": "nan=无完整回合"},
               "mae_z_cm": surface.tolist(), "complete": completes.tolist()}
    with open(os.path.join(OUT, "pd_gain_surface.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=120)
    masked = np.ma.masked_invalid(surface)
    im = ax.pcolormesh(range(len(KDS)), range(len(KPS)), masked, cmap="viridis_r", shading="auto")
    for i in range(len(KPS)):
        for j in range(len(KDS)):
            v = surface[i, j]
            ax.text(j, i, "FAIL" if np.isnan(v) else f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if (not np.isnan(v) and v > 1.0) else "black")
    ax.set_xticks(range(len(KDS)), [str(k) for k in KDS])
    ax.set_yticks(range(len(KPS)), [str(k) for k in KPS])
    ax.set_xlabel("Kd"); ax.set_ylabel("Kp")
    fig.colorbar(im, label="whole-trajectory |Z-Z*| MAE [cm]")
    ax.set_title("B2: PD gain surface (10 paired ICs each; FAIL = no complete episode)")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "pd_gain_surface.png"))
    print("图已存 assets/pd_gain_surface.png", flush=True)


if __name__ == "__main__":
    main()
