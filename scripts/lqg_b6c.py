"""lqg_b6c.py — B6c：鲁棒性修复与失谐分解（2026-10-31）

承接 B6b 笔记（notebooks/B6-LQG部分观测基线.md）8.6 节的三个后续方向。
B6b 已确立：LQG-LQI 完整率在 σ≤1mm 平直 2/10（结构悬崖，与噪声幅值无关）、
10mm 崩塌 0/10；PD 交接硬切换证伪（失败整齐落在切换点后 2–4 步）。
本批把"为什么 LQG 在估计状态上失稳"分解为三个可分离的候选因素，逐一隔离：
① 切换瞬态（bumpless 软切换能否救回交接方案）；
② 反馈增益对估计状态的裕度不足（面向鲁棒性而非最优性重设计增益）；
③ 10mm 档的 KF 失谐（V 冻结在标称 1mm，按实际噪声重设计可部分挽回？）。

模型固定 results/sysid_model_macos.npz；观测矩阵 C=[Zcur;Rin]、过程噪声
W=(1V)²·BBᵀ、配对协议（种子 3000–3009、同一模拟器实例、动作后采样、
噪声种子 777+回合序号）与 B6/B6b 完全一致。LQI 基准权重 qz=30、qi=50（B1 冻结值）。

=====================================================================
预登记（2026-10-31，写于任何 B6c 实验运行之前；网格与切换规则冻结，
不因结果回头修改）
=====================================================================
实验 1（bumpless 软切换，σ=1mm、N=10，与 B6b 硬切换逐种子配对）：
交接框架与 B6b 相同（前 N 步 PD(3,0.3) 带噪闭环 + KF 只观测，KF 时间更新
用实际限幅动作）。切换步 k=N 改为按**控制量连续**初始化积分项：
  u_fb = −K·x̂_N；z_int := (u_fb[0] − u_prev[0]) / Ki[0]
  （u_prev = k=N−1 步实际限幅施加的 PD 动作；P6 通道严格连续——
  设计检查：Ki[0]≈−2.0，z_int 量级 O(0.1)，无病态）
  u_N = clip(u_fb − Ki·z_int)；D5/P5 通道允许跳变（PD 期恒为 0），
  跳变量作诊断记录；k≥N+1 起 z_int += DT·x̂_z 正常累积。
- H9：若切换瞬态是交接失败主因，软切换完整率 ≥5/10（硬切换为 1/10）；
  点预测 3–5/10（部分救回，晚簇 31–40 步残留）。若 ≤2/10 ⇒ 失败主因是
  估计滞后/相位裕度，切换冲击只是次级因素。判别窗口：失败不再集中于
  [N, N+4]。切换冲击对照诊断：硬切换复跑格的 |Δu| 跳变量 vs 软切换 ≈0（P6）。

实验 2（鲁棒性导向增益重设计，LQG-LQI，σ=1mm）：
预指定网格 r ∈ {0.1, 1, 10}（输入权重缩放，qz=30、qi=50 冻结）——
r 大 ⇒ 增益软 ⇒ 对估计滞后/残差的响应弱 ⇒ 相位裕度大但精度差。
全状态 LQI 在 r=0.1、r=10 各跑一组作锚（r=1 锚点引用 B6b 冻结值）。
设计检查（纯计算，先于实验）：r={0.1,1,10} 的模型内闭环极点 {0.9970,
0.9974,0.9974}，|K|max {9390, 4635, 3078}。
- H10：LQG-LQI 完整率随 r 单调（软增益救稳定）：r=0.1 ≤ 2/10（更硬更差）；
  r=10 ≥ 5/10；存活 MAE 随 r 上升（r=10 ≥ 4.5cm，软增益精度代价）。
  全状态锚：r=10 时 MAE > 3.42cm 且 10/10（验证"软=精度差但稳"在真状态上成立）。

实验 3（10mm 档 KF 失谐分解，LQG-LQI，r=1）：
同一批 10mm 噪声（同种子同抽样序列），KF 的 V 按实际 σ=10mm 重设计
（V=(10mm)²·I₂，W 不变，Lc 重新解 DARE）vs B6b 的冻结 V=1mm 设计
（本批复跑作锚）。判别：重设计格完整率 0/10 ⇒ 崩塌纯结构性（10mm 噪声
地板超出估计能力）；1–2/10 ⇒ 失谐有贡献但结构地板仍在（点预测：此档）；
≥3/10 ⇒ 失谐为主导因素。辅证：重设计格的估计误差应显著低于失谐格。
- H11：点预测 1–2/10；est_mae_z(重设计) < est_mae_z(失谐冻结)。

综合判别：
- H12：若实验 1 与实验 2 均不能把 LQG-LQI 完整率提到 ≥5/10 ⇒ 2/10 平台
  的瓶颈是估计滞后本身（环路有效延迟），需补偿器级重设计（超前校正/延迟
  补偿/鲁棒合成），超出增益与滤波器调参范畴，结论留待 B6d。

复现锚（同机同代码同种子，预期逐位一致；不一致先停查因）：
LQG-LQI σ=1mm r=1（B6b: 2/10, 3.784cm）；LQG-LQI σ=10mm V=1mm（B6b: 0/10）；
PD 交接 N=10 硬切换（B6b: 1/10, holds=[14,34,11,12,36,12,11,50,36,33]）。

证伪/意外结果如实记录于 notebooks/B6c-鲁棒性修复与失谐分解.md。
=====================================================================

诊断（不进控制律）：真状态只读记录估计误差；切换步记录跳变向量与 z_int 初值。
输出：results/lqg_b6c.json（逐回合原始数据 + 复现校验 + 切换诊断）、
assets/lqg_b6c.png（软切换逐种子对照、R 网格完整率/MAE、10mm 失谐分解）。
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
from lqg_baseline import make_env, SEEDS, SIGMA_U, DT, OUT, ASSETS  # noqa: E402
from lqg_b6b import LQGSigma, PDHandoffLQG  # noqa: E402

MODEL = os.path.join(OUT, "sysid_model_macos.npz")
R_GRID = [0.1, 1.0, 10.0]      # 预登记输入权重缩放网格
N_HANDOFF = 10                 # 预登记软切换交接长度


class BumplessHandoffLQG(PDHandoffLQG):
    """PD 交接 + 切换步积分项初始化：bumpless=True 时按 P6 通道 u 连续设 z_int。

    bumpless=False 时与 B6b 的 PDHandoffLQG 逐位同律（硬切换复跑锚）。
    切换步诊断：记录 z_int 初值、上一步施加动作 u_prev、跳变向量 bump。
    """

    def __init__(self, *a, bumpless=False, **kw):
        super().__init__(*a, **kw)
        self.bumpless = bumpless

    def reset(self, obs):
        super().reset(obs)
        self._u_prev = None
        self.switch_diag = None

    def act(self, obs):
        y = np.array([obs[2] * 0.1, obs[3] * 0.1]) + self.rng.normal(0.0, self.sigma_meas, 2)
        self.x_hat = self.x_hat + self.Lc @ (y - self.C @ self.x_hat)
        z_true = float(np.asarray(self.env._stepping.plasma_descriptors_vec, dtype=float)[2] - self.x0[14])
        self._est_errs.append(abs(self.x_hat[14] - z_true))
        if self._k < self.n_handoff:
            z = y[0] / 0.1  # 归一化 obs 单位（10cm=1）
            z_dot = 0.0 if self._z_prev is None else (z - self._z_prev) / self.dt
            self._z_prev = z
            u = np.zeros(3)
            u[self.p6_idx] = np.clip(-(self.kp * z + self.kd * z_dot * 1e-3), -1, 1)
        else:
            if self.bumpless and self._k == self.n_handoff:
                # 无扰切换：z_int 使 P6 通道 u_LQG = u_prev（控制量连续）
                u_fb = -self.K @ self.x_hat
                self.z_int = float((u_fb[self.p6_idx] - self._u_prev[self.p6_idx])
                                   / self.Ki[self.p6_idx])
                u = np.clip(u_fb - self.Ki * self.z_int, -1.0, 1.0)
                self.switch_diag = {"z_int_init": self.z_int,
                                    "u_prev": [float(v) for v in self._u_prev],
                                    "bump": [float(v) for v in (u - self._u_prev)]}
            else:
                self.z_int += DT * self.x_hat[14]
                u = np.clip(-self.K @ self.x_hat - self.Ki * self.z_int, -1.0, 1.0)
                if self._k == self.n_handoff:
                    self.switch_diag = {"z_int_init": float(self.z_int),
                                        "u_prev": [float(v) for v in self._u_prev],
                                        "bump": [float(v) for v in (u - self._u_prev)]}
        self._k += 1
        self._u_prev = u.copy()
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
        diag = getattr(ctrl, "switch_diag", None)
        if diag is not None:
            row["switch_diag"] = diag
        rows.append(row)
    summ = summarize(name, rows)
    ests = [r["est_mae_z_mm"] for r in rows if r.get("est_mae_z_mm") is not None and r["sample_valid"]]
    if ests:
        summ["est_mae_z_mm"] = float(np.mean(ests))
    bumps = [np.linalg.norm(r["switch_diag"]["bump"]) for r in rows
             if r.get("switch_diag") is not None and r["sample_valid"]]
    if bumps:
        summ["switch_bump_norm_mean"] = float(np.mean(bumps))
    results[name] = {"episodes": rows, "summary": summ}
    holds = [r["holds"] for r in rows]
    mae = summ["mae_z_cm"]
    print(f"  → 完整 {summ['complete']}/{summ['valid_samples']}, "
          f"MAE_Z {mae if mae == mae else float('nan'):.3f}cm holds={holds}"
          + (f", 估计误差 {summ.get('est_mae_z_mm'):.2f}mm" if ests else "")
          + (f", 切换跳变 {summ.get('switch_bump_norm_mean'):.3f}" if bumps else ""), flush=True)


def main():
    d = np.load(MODEL)
    A, B, x0 = d["A"], d["B"], d["x0"]
    Bn = B * 500.0

    # ---------- 观测模型 / 两套 KF（标称 V=1mm 与失谐分解用 V=10mm） ----------
    C = np.zeros((2, A.shape[0]))
    C[0, 14], C[1, 15] = 1.0, 1.0
    W = SIGMA_U**2 * (B @ B.T)

    def kf_gain(sigma_meas):
        V = sigma_meas**2 * np.eye(2)
        P = solve_discrete_are(A.T, C.T, W, V)
        return P @ C.T @ np.linalg.inv(C @ P @ C.T + V)

    Lc1 = kf_gain(1e-3)    # 标称设计（B6/B6b 冻结）
    Lc10 = kf_gain(10e-3)  # 实验 3：按实际 10mm 重设计
    print(f"KF(1mm) Lc[14,:]={Lc1[14,:].round(4)}；KF(10mm) Lc[14,:]={Lc10[14,:].round(4)}",
          flush=True)

    # ---------- 增益网格（qz=30、qi=50 冻结，r 缩放） ----------
    gains = {}
    for r in R_GRID:
        K, Ki = design_lqi(A, Bn, r=r)
        gains[r] = (K, Ki)
        cl = A - Bn @ K
        print(f"r={r:>4}: Ki[0]={Ki[0]:+.4f}, |K|max={np.abs(K).max():.1f}, "
              f"模型内闭环极点={np.max(np.abs(np.linalg.eigvals(cl))):.4f}", flush=True)

    env = make_env()
    dt = env.full_timestep * env.steps_per_action
    p6_idx = env.control_coils.index("P6")
    K1, Ki1 = gains[1.0]

    results = {}

    # ---------- 实验 1：bumpless 软切换（含硬切换复跑锚） ----------
    eval_controller("交接N=10,硬切换(复跑),σ=1mm", env,
                    BumplessHandoffLQG(K1, Ki1, A, Bn, C, Lc1, env, x0,
                                       3.0, 0.3, p6_idx, dt, N_HANDOFF, 1e-3,
                                       bumpless=False), results)
    eval_controller("交接N=10,bumpless,σ=1mm", env,
                    BumplessHandoffLQG(K1, Ki1, A, Bn, C, Lc1, env, x0,
                                       3.0, 0.3, p6_idx, dt, N_HANDOFF, 1e-3,
                                       bumpless=True), results)

    # ---------- 实验 2：R 网格（LQG-LQI σ=1mm + 全状态锚） ----------
    for r in R_GRID:
        K, Ki = gains[r]
        tag = "(复跑锚)" if r == 1.0 else ""
        eval_controller(f"LQG-LQI,σ=1mm,R×{r:g}{tag}", env,
                        LQGSigma(K, Ki, A, Bn, C, Lc1, env, x0, sigma_meas=1e-3),
                        results)
    for r in [0.1, 10.0]:
        K, Ki = gains[r]
        eval_controller(f"LQI(全状态),R×{r:g}", env, LQIController(K, Ki, env, x0),
                        results)

    # ---------- 实验 3：10mm 档 KF 失谐分解 ----------
    eval_controller("LQG-LQI,σ=10mm,V=1mm(复跑锚)", env,
                    LQGSigma(K1, Ki1, A, Bn, C, Lc1, env, x0, sigma_meas=10e-3),
                    results)
    eval_controller("LQG-LQI,σ=10mm,V=10mm重设计", env,
                    LQGSigma(K1, Ki1, A, Bn, C, Lc10, env, x0, sigma_meas=10e-3),
                    results)

    # ---------- 复现校验（对照 B6b 冻结 JSON，预期逐位一致） ----------
    ref = json.load(open(os.path.join(OUT, "lqg_b6b.json"), encoding="utf-8"))
    repro = {}
    checks = [
        ("LQG-LQI,σ=1mm,R×1(复跑锚)", "LQG-LQI,σ=1mm"),
        ("LQG-LQI,σ=10mm,V=1mm(复跑锚)", "LQG-LQI,σ=10mm"),
        ("交接N=10,硬切换(复跑),σ=1mm", "LQG-LQI+PD交接(N=10),σ=1mm"),
    ]
    for new_name, ref_name in checks:
        h_new = [r["holds"] for r in results[new_name]["episodes"]]
        h_ref = [r["holds"] for r in ref["results"][ref_name]["episodes"]]
        m_new = results[new_name]["summary"]["mae_z_cm"]
        m_ref = ref["results"][ref_name]["summary"]["mae_z_cm"]
        same = (h_new == h_ref) and (m_new == m_ref or (m_new != m_new and m_ref != m_ref))
        repro[new_name] = {"holds_match": h_new == h_ref, "mae_match": m_new == m_ref,
                           "ref_holds": h_ref, "new_holds": h_new, "bitwise": bool(same)}
        print(f"复现校验 {new_name}: holds {'一致' if h_new == h_ref else '不一致!'}, "
              f"MAE {'一致' if m_new == m_ref else '不一致!'}", flush=True)

    payload = {
        "protocol": {
            "seeds": SEEDS, "sampling": "post-action, paired seeds, same simulator instance",
            "model_file": "sysid_model_macos.npz（B6 Part 2 本机重辨识模型）",
            "exp1_bumpless": "交接 N=10、σ=1mm；切换步 z_int := (u_fb[0]−u_prev[0])/Ki[0]，"
                             "P6 通道 u 严格连续；D5/P5 允许跳变（诊断记录）；k≥N+1 起积分正常累积",
            "exp2_r_grid": "r ∈ {0.1, 1, 10}（qz=30、qi=50 冻结）；KF 固定标称 V=1mm",
            "exp3_detune": "σ=10mm 固定；V=(1mm)²I₂ 冻结设计 vs V=(10mm)²I₂ 按实际重设计；"
                           "两格同种子同噪声序列",
            "noise_rng": "每回合 default_rng(777+回合序号)，各控制器独立；交接格与 B6b 逐种子对齐",
            "preregistration": "见 scripts/lqg_b6c.py 文档字符串（2026-10-31，先于任何运行冻结）",
        },
        "repro_check": repro,
        "results": results,
    }
    out_json = os.path.join(OUT, "lqg_b6c.json")
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

    # (a) 软切换 vs 硬切换 vs 无拉起（逐种子 holds）
    cfgs = [("无拉起（R×1 复跑）", "LQG-LQI,σ=1mm,R×1(复跑锚)", "#c08030", "o"),
            ("交接 N=10 硬切换", "交接N=10,硬切换(复跑),σ=1mm", "#4a7fb5", "s"),
            ("交接 N=10 bumpless", "交接N=10,bumpless,σ=1mm", "#1f8a4c", "^")]
    for i, (label, name, color, marker) in enumerate(cfgs):
        holds = [r["holds"] for r in results[name]["episodes"]]
        comp = results[name]["summary"]["complete"]
        ax1.scatter(np.arange(10) + (i - 1) * 0.18, holds, marker=marker, color=color,
                    s=42, label=f"{label}（完整 {comp}/10）", zorder=3)
    ax1.axhline(50, color="k", lw=0.8, ls=":", alpha=0.6)
    ax1.set_xticks(np.arange(10), [f"s{3000 + i}" for i in range(10)], fontsize=8)
    ax1.set_xlabel("配对种子")
    ax1.set_ylabel("存活控制步数 holds（50=完整）")
    ax1.set_title("B6c-1: bumpless 软切换 vs 硬切换 vs 无拉起（σ=1mm）")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # (b) R 网格：完整率（柱）+ 存活 MAE（标注）
    xs = np.arange(len(R_GRID))
    lqg_comp = [results[f"LQG-LQI,σ=1mm,R×{r:g}{'(复跑锚)' if r == 1.0 else ''}"]["summary"]["complete"]
                for r in R_GRID]
    lqg_mae = [results[f"LQG-LQI,σ=1mm,R×{r:g}{'(复跑锚)' if r == 1.0 else ''}"]["summary"]["mae_z_cm"]
               for r in R_GRID]
    fs_comp = [results[f"LQI(全状态),R×{r:g}"]["summary"]["complete"] if r != 1.0 else 10
               for r in R_GRID]
    bars1 = ax2.bar(xs - 0.18, lqg_comp, 0.36, color="#1f8a4c", label="LQG-LQI（估计状态）")
    bars2 = ax2.bar(xs + 0.18, fs_comp, 0.36, color="#8a97a8", label="LQI（全状态）")
    for x, v, m in zip(xs, lqg_comp, lqg_mae):
        mtxt = f"{m:.2f}cm" if m == m else "—"
        ax2.annotate(f"{v}/10\n{mtxt}", (x - 0.18, v), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=8.5)
    for x, v in zip(xs, fs_comp):
        ax2.annotate(f"{v}/10", (x + 0.18, v), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=8.5)
    ax2.set_xticks(xs, [f"R×{r:g}" for r in R_GRID])
    ax2.set_xlabel("LQI 输入权重缩放（qz=30、qi=50 冻结）")
    ax2.set_ylabel("完整回合数 / 10")
    ax2.set_ylim(0, 13.5)
    ax2.set_title("B6c-2: 鲁棒性导向增益网格（σ=1mm）")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    # (c) 10mm 失谐分解（逐种子 holds）
    cfgs3 = [("σ=10mm，V=1mm 失谐（复跑）", "LQG-LQI,σ=10mm,V=1mm(复跑锚)", "#b04030", "o"),
             ("σ=10mm，V=10mm 重设计", "LQG-LQI,σ=10mm,V=10mm重设计", "#1f8a4c", "^")]
    for i, (label, name, color, marker) in enumerate(cfgs3):
        holds = [r["holds"] for r in results[name]["episodes"]]
        comp = results[name]["summary"]["complete"]
        est = results[name]["summary"].get("est_mae_z_mm", float("nan"))
        ax3.scatter(np.arange(10) + (i - 0.5) * 0.22, holds, marker=marker, color=color,
                    s=46, label=f"{label}（完整 {comp}/10，估计误差 {est:.1f}mm）", zorder=3)
    ax3.axhline(50, color="k", lw=0.8, ls=":", alpha=0.6)
    ax3.set_xticks(np.arange(10), [f"s{3000 + i}" for i in range(10)], fontsize=8)
    ax3.set_xlabel("配对种子")
    ax3.set_ylabel("存活控制步数 holds（50=完整）")
    ax3.set_title("B6c-3: 10mm 档 KF 失谐分解（同噪声序列）")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "lqg_b6c.png"))
    print("图已存 assets/lqg_b6c.png", flush=True)


if __name__ == "__main__":
    main()
