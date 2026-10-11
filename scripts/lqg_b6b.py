"""lqg_b6b.py — B6b：观测噪声幅值扫描 × Kalman 初值拉起策略（2026-10-10）

承接 B6（notebooks/B6-LQG部分观测基线.md）第五节的两个明确后续方向：
1. 测量噪声幅值扫描，定量刻画"观测完备性悬崖"（存活率与 MAE 随 σ 的曲线）；
2. Kalman 初值拉起策略——B6 Part 2 失败呈早簇（6–13 步，估计器拉起瞬态）
   与晚簇（31–40 步）两簇，本实验测试预指定的 PD 交接拉起策略。

模型固定为 Part 2 的本机重辨识模型 results/sysid_model_macos.npz
（Windows 冻结模型在本平台已诊断失效，见 B6 笔记第二节）；
反馈增益、KF 结构、观测矩阵 C=[Zcur;Rin]、配对协议（种子 3000–3009、同一
模拟器实例、动作后采样）全部与 B6 Part 2 相同。唯一自变量是 σ 与拉起策略。

=====================================================================
预登记（2026-10-10，写于任何 B6b 实验运行之前；σ 网格与拉起策略
参数冻结，不因结果回头修改）
=====================================================================
实验 1（噪声扫描）：LQG-LQI 与 PD(3,0.3) 各跑 σ ∈ {0, 0.1, 1, 10} mm。
- σ=0 与 σ=1mm 两格在 B6 Part 2 已有数据（PD: 0.0139/0.1246cm 均 10/10；
  LQG-LQI σ=1mm: 2/10、存活 3.784cm）——本脚本重跑这两格作为**逐位复现
  校验**（同机同代码同噪声种子 777+回合序号，预期逐位一致；不一致则停下
  查因），新增 0.1mm 与 10mm 两格。
- 全状态 LQI（无噪声）重跑作锚点（B6 Part 2：10/10，3.418cm）。

假说：
- H5（LQG-LQI 扫描，兼作早簇机理判别）：σ=0 时完整率落入 3–6/10——
  预测无噪声能修复部分但不是全部失败（拉起瞬态在无噪声下依然存在）。
  判别规则：σ=0 完整率 ≤2/10 ⇒ 早簇与测量噪声无关（纯瞬态/模型残余）；
  ≥7/10 ⇒ 1mm 噪声本身就是早簇主因。σ=0.1mm 与 σ=0 完整率差 ≤1 回合
  （噪声低于估计误差地板）。σ=10mm 完整率 ≤1/10（悬崖）。
  完整率随 σ 单调不增。
- H6（PD 扫描）：四档全部 10/10 完整（PD 无模型可失配）；MAE 随 σ 单调
  上升：MAE(0.1mm) ≤ 0.05cm；MAE(10mm) ∈ [0.4, 1.5]cm。

实验 2（拉起策略，σ=1mm 固定）：PD 交接拉起——回合前 N 步由 PD(3,0.3)
在带噪 obs[2] 上闭环，同时 KF 照常做测量更新并以**实际施加的（限幅后）**
PD 电压做时间更新（只观测不闭环）；第 N 步起硬切换到 u = −K·x̂ − Ki·z_int，
积分项 z_int 在交接期间冻结为 0、切换后才开始累积。N ∈ {5, 10} 两档。
- H7：N=10 完整率 ≥5/10（早簇 6–13 步失败被消灭；晚簇 31–40 步预计
  残留——拉起策略不治晚簇）；N=5 完整率 ≤ N=10（拉起不充分）。
  切换冲击判别：若无失败落在 [N, N+3] 窗口则硬切换本身无害；若落入则
  说明切换瞬态成为新失稳源，如实记录。存活回合 MAE ≤ 4.5cm。
- H8（排名不变性）：每一档 σ 下，PD 完整率 ≥ LQG-LQI 完整率，
  且 PD MAE ≤ LQG-LQI 存活回合 MAE——即 B6 的"PD 居首"结论
  在全噪声谱上成立。

证伪/意外结果如实记录于 notebooks/B6-LQG部分观测基线.md 的 B6b 一节。
=====================================================================

诊断（不进控制律）：LQG 类控制器逐步记录 |x̂_z − z_true|（真状态只读）。
输出：results/lqg_b6b.json（逐回合原始数据）、assets/lqg_b6b.png
（完整率-σ 曲线、MAE-σ 曲线、拉起策略逐种子 holds 对照）。
"""
import json
import os
import sys

import numpy as np
from scipy.linalg import solve_discrete_are

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))
sys.path.insert(0, os.path.dirname(__file__))
from fair_eval import run_episode, summarize  # noqa: E402
from lqi_baseline import design_lqi, LQIController  # noqa: E402
from lqg_baseline import (LQGController, NoisyPDController, make_env,  # noqa: E402
                          SEEDS, SIGMA_U, DT, OUT, ASSETS)

MODEL = os.path.join(OUT, "sysid_model_macos.npz")
SIGMAS_MM = [0.0, 0.1, 1.0, 10.0]      # 预登记噪声网格
N_HANDOFFS = [5, 10]                    # 预登记拉起长度网格
SIGMA_HANDOFF = 1e-3                    # 拉起实验固定 σ=1mm


class LQGSigma(LQGController):
    """与 B6 的 LQGController 逐位同律，仅 σ_meas 改为实例参数（扫描需要）。"""

    def __init__(self, *a, sigma_meas=1e-3, **kw):
        super().__init__(*a, **kw)
        self.sigma_meas = sigma_meas

    def act(self, obs):
        y = np.array([obs[2] * 0.1, obs[3] * 0.1]) + self.rng.normal(0.0, self.sigma_meas, 2)
        self.x_hat = self.x_hat + self.Lc @ (y - self.C @ self.x_hat)
        z_true = float(np.asarray(self.env._stepping.plasma_descriptors_vec, dtype=float)[2] - self.x0[14])
        self._est_errs.append(abs(self.x_hat[14] - z_true))
        if self.Ki is not None:
            self.z_int += DT * self.x_hat[14]
        u = -self.K @ self.x_hat
        if self.Ki is not None:
            u = u - self.Ki * self.z_int
        u = np.clip(u, -1.0, 1.0)
        self.x_hat = self.A @ self.x_hat + self.Bn @ u
        return u


class NoisyPDSigma(NoisyPDController):
    """与 B6 的 NoisyPDController 逐位同律，仅 σ_meas 改为实例参数。"""

    def __init__(self, kp, kd, p6_idx, dt, sigma_meas):
        super().__init__(kp, kd, p6_idx, dt)
        self.sigma_meas = sigma_meas

    def reset(self, obs):
        self.rng = np.random.default_rng(777 + self._ep)
        self._ep += 1
        self.z_prev = obs[2] + self.rng.normal(0.0, self.sigma_meas / 0.1)

    def act(self, obs):
        z = obs[2] + self.rng.normal(0.0, self.sigma_meas / 0.1)  # obs 单位（10cm=1）
        z_dot = (z - self.z_prev) / self.dt
        self.z_prev = z
        a = np.zeros(3)
        a[self.p6_idx] = np.clip(-(self.kp * z + self.kd * z_dot * 1e-3), -1, 1)
        return a


class PDHandoffLQG(LQGController):
    """PD 交接拉起：前 n_handoff 步 PD(3,0.3) 闭环 + KF 只观测；随后硬切换 LQG-LQI。

    交接期间 KF 的测量更新照常，时间更新用实际施加的（限幅后）PD 电压；
    积分项 z_int 交接期间冻结为 0，切换后才开始累积（防瞬态偏差注入积分）。
    噪声序列与 LQGController 对齐（只在 act 抽一次 2 维），保证与 σ=1mm
    无拉起基线逐种子配对。
    """

    def __init__(self, K, Ki, A, Bn, C, Lc, env, x0, kp, kd, p6_idx, dt,
                 n_handoff, sigma_meas):
        super().__init__(K, Ki, A, Bn, C, Lc, env, x0)
        self.kp, self.kd, self.p6_idx, self.dt = kp, kd, p6_idx, dt
        self.n_handoff = n_handoff
        self.sigma_meas = sigma_meas

    def reset(self, obs):
        super().reset(obs)
        self._k = 0
        self._z_prev = None

    def act(self, obs):
        y = np.array([obs[2] * 0.1, obs[3] * 0.1]) + self.rng.normal(0.0, self.sigma_meas, 2)
        self.x_hat = self.x_hat + self.Lc @ (y - self.C @ self.x_hat)
        z_true = float(np.asarray(self.env._stepping.plasma_descriptors_vec, dtype=float)[2] - self.x0[14])
        self._est_errs.append(abs(self.x_hat[14] - z_true))
        if self._k < self.n_handoff:
            z = y[0] / 0.1  # 归一化 obs 单位（10cm=1），与 NoisyPD 同律
            z_dot = 0.0 if self._z_prev is None else (z - self._z_prev) / self.dt
            self._z_prev = z
            u = np.zeros(3)
            u[self.p6_idx] = np.clip(-(self.kp * z + self.kd * z_dot * 1e-3), -1, 1)
        else:
            self.z_int += DT * self.x_hat[14]
            u = np.clip(-self.K @ self.x_hat - self.Ki * self.z_int, -1.0, 1.0)
        self._k += 1
        self.x_hat = self.A @ self.x_hat + self.Bn @ u
        return u


def eval_controller(name, env, ctrl, results):
    print(f"评估 {name} …", flush=True)
    rows = []
    for s in SEEDS:
        row = run_episode(env, ctrl, s)
        est = getattr(ctrl, "last_ep_est_mae_z_mm", None)
        if est is not None:
            row["est_mae_z_mm"] = est
        rows.append(row)
    summ = summarize(name, rows)
    ests = [r["est_mae_z_mm"] for r in rows if r.get("est_mae_z_mm") is not None and r["sample_valid"]]
    if ests:
        summ["est_mae_z_mm"] = float(np.mean(ests))
    results[name] = {"episodes": rows, "summary": summ}
    holds = [r["holds"] for r in rows]
    print(f"  → 完整 {summ['complete']}/{summ['valid_samples']}, "
          f"MAE_Z {summ['mae_z_cm'] if summ['mae_z_cm']==summ['mae_z_cm'] else float('nan'):.3f}cm"
          f" holds={holds}"
          + (f", 估计误差 {summ.get('est_mae_z_mm'):.2f}mm" if ests else ""), flush=True)


def main():
    d = np.load(MODEL)
    A, B, x0 = d["A"], d["B"], d["x0"]
    Bn = B * 500.0

    # KF 与反馈增益：与 B6 Part 2 完全同配置（V 按基准 σ=1mm 冻结——扫描只动
    # 注入的测量噪声，不动滤波器设计，模拟"滤波器按标称 1mm 设计、实际噪声未知"）
    C = np.zeros((2, A.shape[0]))
    C[0, 14], C[1, 15] = 1.0, 1.0
    V = 1e-3**2 * np.eye(2)
    W = SIGMA_U**2 * (B @ B.T)
    P = solve_discrete_are(A.T, C.T, W, V)
    Lc = P @ C.T @ np.linalg.inv(C @ P @ C.T + V)
    K_lqi, Ki_lqi = design_lqi(A, Bn, r=1.0)
    cl = A - Bn @ K_lqi
    print(f"LQI-3ch 模型内闭环最大特征值(模): {np.max(np.abs(np.linalg.eigvals(cl))):.4f}",
          flush=True)

    env = make_env()
    dt = env.full_timestep * env.steps_per_action
    p6_idx = env.control_coils.index("P6")

    results = {}

    # ---------- 实验 1：噪声扫描（σ=0/1mm 格为逐位复现校验） ----------
    for sigma_mm in SIGMAS_MM:
        s = sigma_mm * 1e-3
        eval_controller(f"PD(3,0.3),σ={sigma_mm:g}mm", env,
                        NoisyPDSigma(3.0, 0.3, p6_idx, dt, s), results)
        eval_controller(f"LQG-LQI,σ={sigma_mm:g}mm", env,
                        LQGSigma(K_lqi, Ki_lqi, A, Bn, C, Lc, env, x0, sigma_meas=s),
                        results)

    # ---------- 锚点：全状态 LQI（无噪声） ----------
    eval_controller("LQI(三通道,全状态)", env, LQIController(K_lqi, Ki_lqi, env, x0), results)

    # ---------- 实验 2：PD 交接拉起（σ=1mm 固定） ----------
    for n in N_HANDOFFS:
        eval_controller(f"LQG-LQI+PD交接(N={n}),σ=1mm", env,
                        PDHandoffLQG(K_lqi, Ki_lqi, A, Bn, C, Lc, env, x0,
                                     3.0, 0.3, p6_idx, dt, n, SIGMA_HANDOFF),
                        results)

    # ---------- 逐位复现校验（对照 B6 Part 2 冻结 JSON） ----------
    ref = json.load(open(os.path.join(OUT, "lqg_eval_macmodel.json"), encoding="utf-8"))
    repro = {}
    checks = [
        ("PD(3,0.3),σ=0mm", "PD(3,0.3)干净观测"),
        ("PD(3,0.3),σ=1mm", "PD(3,0.3)带噪观测"),
        ("LQG-LQI,σ=1mm", "LQG-LQI(KF,部分观测)"),
        ("LQI(三通道,全状态)", "LQI(三通道,全状态)"),
    ]
    for new_name, ref_name in checks:
        h_new = [r["holds"] for r in results[new_name]["episodes"]]
        h_ref = [r["holds"] for r in ref["results"][ref_name]["episodes"]]
        m_new = results[new_name]["summary"]["mae_z_cm"]
        m_ref = ref["results"][ref_name]["summary"]["mae_z_cm"]
        same = (h_new == h_ref) and (m_new == m_ref or (m_new != m_new and m_ref != m_ref))
        repro[new_name] = {"holds_match": h_new == h_ref, "mae_match": m_new == m_ref,
                           "ref_complete": ref["results"][ref_name]["summary"]["complete"],
                           "new_complete": results[new_name]["summary"]["complete"]}
        print(f"复现校验 {new_name}: holds {'一致' if h_new == h_ref else '不一致!'} "
              f"({h_new} vs {h_ref}), MAE {'一致' if m_new == m_ref else '不一致!'}", flush=True)

    payload = {
        "protocol": {
            "seeds": SEEDS, "sampling": "post-action, paired seeds, same simulator instance",
            "model_file": "sysid_model_macos.npz（B6 Part 2 本机重辨识模型）",
            "noise_sweep": "σ ∈ {0, 0.1, 1, 10} mm（预登记网格）；KF 的 V 按标称 1mm 冻结设计，"
                           "只动注入噪声（滤波器按标称设计、实际噪声未知的情形）",
            "handoff": "前 N 步 PD(3,0.3) 带噪闭环 + KF 只观测（时间更新用实际限幅动作），"
                       "第 N 步硬切换 LQG-LQI；交接期间 z_int 冻结为 0；N ∈ {5, 10}，σ=1mm",
            "noise_rng": "每回合 default_rng(777+回合序号)，各控制器独立；"
                         "交接控制器与无拉起基线逐种子噪声对齐",
            "preregistration": "见 scripts/lqg_b6b.py 文档字符串（2026-10-10，先于任何运行冻结）",
            "repro_check": "σ=0/1mm 与全状态锚点四格应与 B6 Part 2 冻结值逐位一致",
        },
        "repro_check": repro,
        "results": results,
    }
    out_json = os.path.join(OUT, "lqg_b6b.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n原始逐回合数据已存 {out_json}", flush=True)

    # ---------- 图 ----------
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
    ax1.set_xlabel("测量噪声 σ [mm]（对数等距网格，含 0）")
    ax1.set_ylabel("完整回合数 / 10")
    ax1.set_ylim(-0.5, 10.5)
    ax1.set_title("B6b-1: 观测完备性悬崖——完整率 vs 噪声幅值")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # (b) MAE vs σ（存活回合均值，标注 n）
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
                ax2.annotate(f"{v:.2f}\n(n={n})", (x, v), textcoords="offset points",
                             xytext=(0, 8), ha="center", fontsize=8)
    ref_mae = results["LQI(三通道,全状态)"]["summary"]["mae_z_cm"]
    ax2.axhline(ref_mae, color="#2b7a2b", ls="--", lw=1,
                label=f"全状态 LQI 参照 ({ref_mae:.2f}cm)")
    ax2.set_xticks(xs, xticklabels)
    ax2.set_xlabel("测量噪声 σ [mm]")
    ax2.set_ylabel("全过程垂直 MAE [cm]（对数轴）")
    ax2.set_yscale("log")
    ax2.set_title("B6b-1: 存活误差 vs 噪声幅值（失败回合不入均值）")
    ax2.legend(fontsize=9)
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
    print("图已存 assets/lqg_b6b.png", flush=True)


if __name__ == "__main__":
    main()
