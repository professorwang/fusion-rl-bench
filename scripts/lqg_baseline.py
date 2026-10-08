"""lqg_baseline.py — B6：LQG 部分观测基线（Kalman 滤波 + LQR/LQI 状态反馈）

研究问题：当控制器只能拿到部分带噪观测（而非全状态）时，基于模型的最优控制
（LQR/LQI）排名如何变化？这是与学习类控制器"同观测条件"的公平模型驱动对手
（B1 笔记指出的 B-next 方向）。

=====================================================================
预登记（2026-10-08，写于任何闭环实验运行之前；此后观测配置冻结，
不因结果回头修改）
=====================================================================
观测模型（冻结）：
- y_k = C·dx_k + v_k，C 只取 Zcur（状态 idx14）与 Rin（idx15）——对应真实
  装置磁诊断可得量；Rx/Zx 与 12 个主动线圈电流全部不可测。
- 测量噪声 v_k ~ N(0, σ²·I₂)，σ = 1mm（磁诊断位置重建的典型量级）。
- 噪声序列按回合固定种子（777+回合序号），可复现；不同控制器各自独立抽取。
  噪声只进控制器，不进模拟器（配对协议的其他要素与 fair_eval 完全一致）。

Kalman 滤波器（冻结）：
- 稳态滤波器，当前估计器形式：测量更新 x̂ += Lc·(y − C·x̂)，随后时间更新
  x̂ ← A·x̂ + B·u_applied（u_applied 用限幅后的实际动作，防估计器饱和失配）。
- 过程噪声 W = σ_u²·B·Bᵀ，σ_u = 1.0V（0.2% 满量程的电压参考过程噪声，
  涵盖模型失配）；测量噪声协方差 V = σ²·I₂。
- Lc 由 solve_discrete_are(Aᵀ, Cᵀ, W, V) 稳态解给出：Lc = P·Cᵀ·(C·P·Cᵀ+V)⁻¹。
- reset 时 x̂ = 0（基线偏差坐标原点），由首次测量更新拉起。

反馈律（与 B1 完全相同的权重矩阵，保证与 B1 可比）：
- LQG-LQI：K, Ki = design_lqi(A, B·500, qz=30, qi=50, r=1)（B1 最优模型驱动
  配置），u = −K·x̂ − Ki·∫ẑ_err dt，积分项用**估计的** Zcur 偏差。
- LQG-LQR：K = design_lqr(A, B·500, R=100)（B1 三通道最优 R），u = −K·x̂。
对照组：
- LQI（三通道，全状态）：B1 复跑（同权重同协议），检验跨平台数值漂移；
- PD(3,0.3) 干净观测（B3/B4 最优经典基线）与带噪观测版（obs[2] 加同样
  1mm 噪声，导数由带噪差分计算）；
- LQR 直馈（无滤波）：伪状态 dx=[0…0, y_z, y_r]（不可测状态硬置零），
  u = −K·dx——"若适用"的朴素部分观测基线。

设计可行性检查（先于实验的纯设计计算，非实验结果）：
(A,C) 可观测秩 16/16；全部不稳定模（1.1254、1.038、1.027）PBH 可检测；
稳态 Lc 的 Zcur 通道增益 0.32，估计误差极点最大模 0.995（稳定），
稳态 Zcur 估计标准差 0.69mm。

假说：
- H1（主）：LQG-LQI 接近全状态 LQI——退化 <50%（MAE_Z ≤ 1.4cm，对照 B1
  全状态 LQI 的 0.91cm），且 10/10 完整回合。理由：不稳定垂直模可由 Zcur
  直接观测，1mm 噪声相对 cm 级初始扰动很小。
- H2：LQG-LQR（无积分）仍有与全状态 LQR 同量级的稳态偏差（MAE_Z 落在
  2.2±1cm 区间）——即"积分作用有无"而非"观测完备性"是 B1 稳态偏差主因。
- H3：LQR 直馈（无滤波）显著劣于 LQG-LQR（MAE_Z ≥ 2 倍）或出现失败回合
  ——硬置零的 12 个线圈电流状态使反馈施加错误的电流补偿。
- H4（排名）：部分观测下总排名方向不变：PD(3,0.3) 仍居首位；带噪 PD 相对
  干净 PD 退化 <3 倍（1mm 噪声经有限差分放大后等效电压抖动仅 ~4V）。

证伪/意外结果将如实记录于 notebooks/B6-LQG部分观测基线.md。
=====================================================================

事后补记（2026-10-08，Part 1 跑完之后追加，以上预登记原文未动）：
Part 1（默认参数，Windows 冻结模型）结果：全部模型驱动控制器 10/10 回合失败
——诊断为跨平台模型失效（见 scripts/model_platform_check.py 与
results/model_platform_check.json：基线逐位一致但一步预测误差放大 ~30 倍）。
Part 2（--model results/sysid_model_macos.npz --tag _macmodel）：用本机重新辨识
的模型（scripts/sysid_macos.py，同一权重矩阵）重跑同一控制器套件，
回答原研究问题。两部分在笔记中分开陈述。

诊断（不进控制律）：LQG 控制器逐步记录 |x̂_z − z_true|（真状态只读，仅用于
事后评估估计质量；控制律只接触 x̂ 与 y）。
输出：results/lqg_eval{tag}.json（逐回合原始数据 + 估计误差诊断）、
assets/lqg_eval{tag}.png（误差柱状图 + seed 3000 估计轨迹）。
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.linalg import solve_discrete_are

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))
sys.path.insert(0, os.path.dirname(__file__))
from freegsnke_linear_pos_env import FreeGSNKELinearPosEnv  # noqa: E402
from fair_eval import PDController, run_episode, summarize  # noqa: E402
from lqr_baseline import design_lqr  # noqa: E402
from lqi_baseline import design_lqi, LQIController  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(OUT, exist_ok=True)
os.makedirs(ASSETS, exist_ok=True)

N_ACTIVE = 12
SEEDS = list(range(3000, 3010))
SIGMA_MEAS = 1e-3   # 测量噪声标准差 [m]（预登记冻结）
SIGMA_U = 1.0       # 电压参考过程噪声 [V]（预登记冻结）
DT = 5e-4           # 控制周期 [s]


class LQGController:
    """稳态 Kalman（当前估计器）+ LQR/LQI 增益对估计状态反馈。

    控制律只接触 x̂ 与 y；真状态只读用于估计误差诊断（记录用，不进反馈）。
    """

    def __init__(self, K, Ki, A, Bn, C, Lc, env, x0, record_first=False):
        self.K, self.Ki = K, Ki
        self.A, self.Bn, self.C, self.Lc = A, Bn, C, Lc
        self.env, self.x0 = env, x0
        self.record_first = record_first
        self._ep = 0
        self.traj = None

    def reset(self, obs):
        self.x_hat = np.zeros(self.A.shape[0])
        self.z_int = 0.0
        self.rng = np.random.default_rng(777 + self._ep)
        self._recording = self.record_first and self._ep == 0  # 只记录首个回合（seed 3000）
        self._ep += 1
        self._est_errs = []
        if self._recording:
            self.traj = {"y_z": [], "z_hat": [], "z_true": [], "u_p6": []}

    def act(self, obs):
        y = np.array([obs[2] * 0.1, obs[3] * 0.1]) + self.rng.normal(0.0, SIGMA_MEAS, 2)
        # 测量更新（当前估计器）
        self.x_hat = self.x_hat + self.Lc @ (y - self.C @ self.x_hat)
        # 诊断：估计误差（只读真状态，不进控制律）
        z_true = float(np.asarray(self.env._stepping.plasma_descriptors_vec, dtype=float)[2] - self.x0[14])
        self._est_errs.append(abs(self.x_hat[14] - z_true))
        if self._recording:
            self.traj["y_z"].append(float(y[0]))
            self.traj["z_hat"].append(float(self.x_hat[14]))
            self.traj["z_true"].append(z_true)
        # 反馈（对估计状态）
        if self.Ki is not None:
            self.z_int += DT * self.x_hat[14]
        u = -self.K @ self.x_hat
        if self.Ki is not None:
            u = u - self.Ki * self.z_int
        u = np.clip(u, -1.0, 1.0)
        if self._recording:
            self.traj["u_p6"].append(float(u[0]))
        # 时间更新（用限幅后的实际动作，防估计器饱和失配）
        self.x_hat = self.A @ self.x_hat + self.Bn @ u
        return u

    @property
    def last_ep_est_mae_z_mm(self):
        return float(np.mean(self._est_errs) * 1e3) if self._est_errs else None


class DirectLQRController:
    """无滤波直馈：伪状态 dx = [0…0, y_z, y_r]（不可测状态硬置零），u = −K·dx。"""

    def __init__(self, K):
        self.K = K
        self._ep = 0

    def reset(self, obs):
        self.rng = np.random.default_rng(777 + self._ep)
        self._ep += 1

    def act(self, obs):
        y = np.array([obs[2] * 0.1, obs[3] * 0.1]) + self.rng.normal(0.0, SIGMA_MEAS, 2)
        dx = np.zeros(self.K.shape[1])
        dx[14], dx[15] = y
        return np.clip(-self.K @ dx, -1.0, 1.0)


class NoisyPDController:
    """PD(3,0.3) 带噪观测版：obs[2] 加 1mm 高斯噪声，导数由带噪差分计算。"""

    def __init__(self, kp, kd, p6_idx, dt):
        self.kp, self.kd, self.p6_idx, self.dt = kp, kd, p6_idx, dt
        self._ep = 0

    def reset(self, obs):
        self.rng = np.random.default_rng(777 + self._ep)
        self._ep += 1
        self.z_prev = obs[2] + self.rng.normal(0.0, SIGMA_MEAS / 0.1)

    def act(self, obs):
        z = obs[2] + self.rng.normal(0.0, SIGMA_MEAS / 0.1)  # obs 单位（10cm=1）
        z_dot = (z - self.z_prev) / self.dt
        self.z_prev = z
        a = np.zeros(3)
        a[self.p6_idx] = np.clip(-(self.kp * z + self.kd * z_dot * 1e-3), -1, 1)
        return a


def make_env():
    return FreeGSNKELinearPosEnv(
        control_coils=["P6", "D5", "P5"], max_steps=50, steps_per_action=1,
        max_voltage=500.0, init_disturb_steps=3, init_disturb_voltage=50.0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(OUT, "sysid_model.npz"),
                    help="默认 Windows 冻结模型（预登记 Part 1）；"
                         "results/sysid_model_macos.npz 为事后 Part 2（本机重辨识）")
    ap.add_argument("--tag", default="", help="输出文件后缀（Part 2 用 _macmodel）")
    args = ap.parse_args()

    d = np.load(args.model)
    A, B, x0 = d["A"], d["B"], d["x0"]
    Bn = B * 500.0  # 归一化动作单位（±1 = ±500V），与 B1 一致

    # ---------- 观测模型与稳态 Kalman 增益（预登记配置） ----------
    C = np.zeros((2, A.shape[0]))
    C[0, 14], C[1, 15] = 1.0, 1.0  # 只测 Zcur、Rin
    V = SIGMA_MEAS**2 * np.eye(2)
    W = SIGMA_U**2 * (B @ B.T)
    P = solve_discrete_are(A.T, C.T, W, V)
    Lc = P @ C.T @ np.linalg.inv(C @ P @ C.T + V)
    est_poles = np.linalg.eigvals((np.eye(A.shape[0]) - Lc @ C) @ A)
    print(f"KF: Lc[14,:]={Lc[14,:].round(4)}, Lc[15,:]={Lc[15,:].round(4)}, "
          f"估计误差极点最大模={np.max(np.abs(est_poles)):.4f}, "
          f"稳态估计std Zcur={np.sqrt(P[14,14])*1e3:.3f}mm Rin={np.sqrt(P[15,15])*1e3:.3f}mm",
          flush=True)

    # ---------- 反馈增益（与 B1 同权重） ----------
    K_lqi, Ki_lqi = design_lqi(A, Bn, r=1.0)          # qz=30, qi=50（默认）
    K_lqr = design_lqr(A, Bn, r=100.0)                # B1 三通道最优 R
    for name, Kx, Bx in [("LQI-3ch", K_lqi, Bn), ("LQR-3ch-R100", K_lqr, Bn)]:
        cl = A - Bx @ Kx
        print(f"{name} 模型内闭环最大特征值(模): {np.max(np.abs(np.linalg.eigvals(cl))):.4f}",
              flush=True)

    env = make_env()
    dt = env.full_timestep * env.steps_per_action
    p6_idx = env.control_coils.index("P6")

    controllers = {
        "PD(3,0.3)干净观测": PDController(3.0, 0.3, p6_idx, dt),
        "PD(3,0.3)带噪观测": NoisyPDController(3.0, 0.3, p6_idx, dt),
        "LQI(三通道,全状态)": LQIController(K_lqi, Ki_lqi, env, x0),
        "LQG-LQI(KF,部分观测)": LQGController(K_lqi, Ki_lqi, A, Bn, C, Lc, env, x0,
                                             record_first=True),
        "LQG-LQR(R=100,KF,部分观测)": LQGController(K_lqr, None, A, Bn, C, Lc, env, x0),
        "LQR直馈(部分观测,无KF)": DirectLQRController(K_lqr),
    }

    results = {}
    for name, ctrl in controllers.items():
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
        print(f"  → 完整 {summ['complete']}/{summ['valid_samples']}, "
              f"MAE_Z {summ['mae_z_cm']:.3f}cm"
              + (f", 估计误差 {summ.get('est_mae_z_mm', float('nan')):.2f}mm" if ests else ""),
              flush=True)

    print(f"\n{'控制器':<30}{'完整/有效/总':>14}{'MAE_Z(cm)':>12}{'MAE联合(cm)':>12}{'饱和率':>8}{'估计误差(mm)':>14}")
    for name, data in results.items():
        s = data["summary"]
        print(f"{name:<30}{s['complete']:>5}/{s['valid_samples']:>3}/{s['episodes']:<3}"
              f"{s['mae_z_cm']:>12.3f}{s['mae_joint_cm']:>12.3f}{s['sat_ratio']:>8.2%}"
              f"{s.get('est_mae_z_mm', float('nan')):>14.2f}")

    payload = {
        "protocol": {
            "seeds": SEEDS, "sampling": "post-action, paired seeds, same simulator instance",
            "model_file": os.path.basename(args.model),
            "phase": ("Part 1（预登记）：Windows 冻结模型" if not args.tag else
                      "Part 2（事后补救）：macOS 本机重辨识模型，同一权重矩阵"),
            "observation_model": "y = C dx + v; C = [Zcur(idx14); Rin(idx15)]; "
                                 "sigma_meas = 1e-3 m（预登记冻结）",
            "kalman_filter": "稳态当前估计器；W = (1.0V)^2 · B·Bᵀ；时间更新用限幅后实际动作；"
                             "reset 时 x̂=0",
            "feedback_gains": "LQI qz=30 qi=50 r=1（B1 同权重）；LQR R=100（B1 三通道最优）",
            "noise_rng": "每回合 default_rng(777+回合序号)，各控制器独立",
            "preregistration": "见 scripts/lqg_baseline.py 文档字符串（2026-10-08，先于任何运行冻结）",
            "diagnostic_note": "真状态只读用于估计误差诊断，不进控制律",
        },
        "results": results,
        "lqg_lqi_traj_seed3000": controllers["LQG-LQI(KF,部分观测)"].traj,
    }
    out_json = os.path.join(OUT, f"lqg_eval{args.tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n原始逐回合数据已存 {out_json}", flush=True)

    # ---------- 图 ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=120)

    names = list(results.keys())
    maes = [results[n]["summary"]["mae_z_cm"] for n in names]
    completes = [results[n]["summary"]["complete"] for n in names]
    colors = ["#5a6b7f", "#8a97a8", "#2b7a2b", "#1f8a4c", "#c08030", "#b04030"]
    # 失败控制器（0 完整回合）画红色斜纹占位柱，完整控制器画实际误差柱
    ok_maes = [v for v, c in zip(maes, completes) if c > 0]
    fail_h = (max(ok_maes) * 3) if ok_maes else 1.0
    for i, (n, v, c) in enumerate(zip(names, maes, completes)):
        if c > 0:
            ax1.bar(i, v, color=colors[i % len(colors)], width=0.65)
            ax1.text(i, v * 1.12, f"{v:.2f}\n({c}/10)", ha="center", fontsize=9)
        else:
            ax1.bar(i, fail_h, color="#d08080", hatch="//", width=0.65, alpha=0.6)
            ax1.text(i, fail_h * 1.12, f"失败\n({c}/10)", ha="center", fontsize=9, color="#a03030")
    lo = (min(ok_maes) / 5) if ok_maes else 1e-2
    hi = fail_h * 6 if any(c == 0 for c in completes) else max(ok_maes) * 6
    ax1.set_ylim(lo, hi)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, fontsize=8.5, rotation=18, ha="right")
    ax1.set_ylabel("全过程垂直 MAE [cm]（对数轴）")
    ax1.set_yscale("log")
    ax1.set_title(f"B6: LQG 部分观测 vs 全状态/PD/直馈 — 配对评估（模型：{os.path.basename(args.model)}）")
    ax1.grid(axis="y", alpha=0.3)

    traj = controllers["LQG-LQI(KF,部分观测)"].traj
    if traj:
        t_ms = np.arange(len(traj["z_true"])) * DT * 1e3
        ax2.plot(t_ms, np.array(traj["y_z"]) * 100, ".", color="#999999", ms=4,
                 alpha=0.7, label="带噪测量 y (Zcur)")
        ax2.plot(t_ms, np.array(traj["z_true"]) * 100, "k-", lw=2, label="真实 Zcur 偏差")
        ax2.plot(t_ms, np.array(traj["z_hat"]) * 100, "g--", lw=1.5, label="KF 估计 x̂ (Zcur)")
        ax2.set_xlabel("时间 [ms]")
        ax2.set_ylabel("Zcur 偏差 [cm]")
        ax2.set_title("seed 3000：LQG-LQI 的 Kalman 估计质量（σ_meas=1mm）")
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, f"lqg_eval{args.tag}.png"))
    print(f"图已存 assets/lqg_eval{args.tag}.png", flush=True)


if __name__ == "__main__":
    main()
