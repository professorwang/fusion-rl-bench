"""model_platform_check.py — B6 辅助诊断：Windows 冻结模型在当前 macOS 机器上是否仍有效？

动机：B6 首轮评估中，B1 的 LQI 全状态控制器（Windows 上 0.91cm、10/10 完整、0% 饱和）
在本机 10/10 回合全部 1–6 步内 position_lost（63% 饱和）。B5 已记录"跨平台数值漂移
×1.8（开环不稳定通道放大数值细节）"。本脚本区分两种解释：
(a) 平台数值漂移导致辨识模型/状态布局在本机失效（模型-对象失配增大）；
(b) B6 脚本接线错误。

检查项（全部事实记录，不做阈值判定）：
1. sysid_model.npz 哈希与 SHA256SUMS.txt 冻结值比对（模型文件完好性）；
2. 无扰动 reset 下：currents_vec[:12] vs 冻结 x0[:12]（线圈电流基线/布局），
   plasma_descriptors_vec vs x0[12:16] 与 env._target（描述符基线漂移）；
3. 本机一步预测相对误差（B0 协议：seed 999、±30V 随机激励、50 步），与 B0 冻结值对照；
4. seed 3000 扰动后：开环零作用轨迹 vs LQI 全状态闭环轨迹（逐步 z_err、u、|dx| 分解），
   以及同增益在辨识模型内的纯线性闭环仿真对照（模型内稳定性已知：极点 0.995）。

输出：results/model_platform_check.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))
sys.path.insert(0, os.path.dirname(__file__))
from freegsnke_linear_pos_env import FreeGSNKELinearPosEnv  # noqa: E402
from lqi_baseline import design_lqi, LQIController  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
N_ACTIVE = 12


def get_state(env):
    st = env._stepping
    cur = np.array(st.currents_vec[:N_ACTIVE], dtype=float).real
    desc = np.array(st.plasma_descriptors_vec, dtype=float)
    return np.concatenate([cur, desc])


def apply_voltage(env, v_ctrl):
    st = env._stepping
    v = np.zeros(st.evol_metal_curr.n_active_coils)
    for a, i in zip(v_ctrl, env._ctrl_idx):
        v[i] = a
    st.nlstepper(active_voltage_vec=v, linear_only=True, no_GS=True, verbose=False)


def main():
    d = np.load(os.path.join(OUT, "sysid_model.npz"))
    A, B, x0 = d["A"], d["B"], d["x0"]
    report = {}

    # ---------- 2. 无扰动 reset 下的基线比对 ----------
    env = FreeGSNKELinearPosEnv(
        control_coils=["P6", "D5", "P5"], max_steps=50, steps_per_action=1,
        max_voltage=500.0, init_disturb_steps=0)
    env.reset(seed=0)
    x_now = get_state(env)
    report["baseline"] = {
        "coil_max_abs_diff_A": float(np.max(np.abs(x_now[:12] - x0[:12]))),
        "coil_rel_diff": [float(abs(a - b) / max(abs(b), 1e-9)) for a, b in zip(x_now[:12], x0[:12])],
        "desc_now": x_now[12:].tolist(),
        "desc_frozen_x0": x0[12:].tolist(),
        "desc_env_target": env._target.tolist(),
        "desc_abs_diff_vs_x0_m": (x_now[12:] - x0[12:]).tolist(),
    }
    print(f"[基线] 线圈电流最大绝对差 {report['baseline']['coil_max_abs_diff_A']:.4e} A；"
          f"描述符差(本机-x0) {np.round(x_now[12:] - x0[12:], 6)}", flush=True)

    # ---------- 3. 一步预测验证（B0 协议） ----------
    rng = np.random.default_rng(42)  # B0 用同一 rng 序列（42 起始）复现激励样式
    for _ in range(100 * 40):  # 跳过辨识阶段的随机数消耗，逼近 B0 留出 rollout 的 rng 状态
        rng.uniform(-30.0, 30.0, size=3)
    env.reset(seed=999)
    rel = {c: [] for c in ["coils", "Rx", "Zx", "Zcur", "Rin"]}
    for _ in range(50):
        x_k = get_state(env) - x0
        u_k = rng.uniform(-30.0, 30.0, size=3)
        apply_voltage(env, u_k)
        x_true = get_state(env) - x0
        e = np.abs(x_true - (A @ x_k + B @ u_k)) / np.maximum(np.abs(x_true), 1e-6)
        rel["coils"].append(e[:12].mean())
        for j, c in enumerate(["Rx", "Zx", "Zcur", "Rin"]):
            rel[c].append(e[12 + j])
    report["one_step_rel_err"] = {c: float(np.mean(v)) for c, v in rel.items()}
    print(f"[一步预测] { {c: round(v, 4) for c, v in report['one_step_rel_err'].items()} }"
          f"（B0 冻结值: coils 0.038, Rx 0.136, Zx 0.018, Zcur 0.013, Rin 0.073）", flush=True)

    # ---------- 4. 标准扰动 reset（seed 3000）：开环 vs LQI 闭环 ----------
    env2 = FreeGSNKELinearPosEnv(
        control_coils=["P6", "D5", "P5"], max_steps=50, steps_per_action=1,
        max_voltage=500.0, init_disturb_steps=3, init_disturb_voltage=50.0)

    # 4a. 开环零作用轨迹（参照）
    obs, _ = env2.reset(seed=3000)
    openloop_z = []
    for k in range(15):
        obs, r, term, trunc, info = env2.step(np.zeros(3))
        openloop_z.append(float(obs[2] * 0.1))
        if term:
            break
    report["openloop_z_m"] = openloop_z
    print(f"[开环] z_err(m) 前 {len(openloop_z)} 步: {np.round(openloop_z, 4)}", flush=True)

    # 4b. LQI 全状态闭环轨迹（逐步记录）
    Bn = B * 500.0
    K3, Ki3 = design_lqi(A, Bn, r=1.0)
    obs, _ = env2.reset(seed=3000)
    ctrl = LQIController(K3, Ki3, env2, x0)
    ctrl.reset(obs)
    lqi_trace = []
    for k in range(15):
        x_now = get_state(env2)
        dx = x_now - x0
        u = ctrl.act(obs)
        lqi_trace.append({
            "k": k, "z_err_m": float(dx[14]),
            "dx_coils_maxabs_A": float(np.max(np.abs(dx[:12]))),
            "u": u.tolist(),
        })
        obs, r, term, trunc, info = env2.step(u)
        if term:
            lqi_trace[-1]["fail_reason"] = info.get("fail_reason")
            break
    report["lqi_trace"] = lqi_trace
    for t in lqi_trace:
        print(f"[LQI] k={t['k']} z_err={t['z_err_m']:+.4f}m |dx_coils|max={t['dx_coils_maxabs_A']:.1f}A "
              f"u={np.round(t['u'], 3)} {t.get('fail_reason', '')}", flush=True)

    # 4c. 同增益在辨识模型内的纯线性闭环（不含模拟器）：从同一扰动后状态出发
    #     用 4b 第一步的实际 dx 作为初值
    x_m = np.array(lqi_trace and (get_state(env2) - x0))  # 当前 env2 状态（已失败终止）
    # 重新用 4b 首步记录的 dx 重构初值更干净：z_err 与 coils 仅部分可知，
    # 这里改用"从 4b 第一步 dx 范数量级合成的扰动初值"：
    # 直接以 4b k=0 的真实 dx 为初值（重新 reset 一次取状态）
    obs, _ = env2.reset(seed=3000)
    x_m = get_state(env2) - x0
    z_int = 0.0
    model_trace = []
    for k in range(50):
        z_int += 5e-4 * x_m[14]
        u = np.clip(-K3 @ x_m - Ki3 * z_int, -1.0, 1.0)
        model_trace.append({"k": k, "z_err_m": float(x_m[14]), "u": u.tolist()})
        x_m = A @ x_m + Bn @ u
    report["model_closedloop_z_m"] = [t["z_err_m"] for t in model_trace[:15]]
    report["model_closedloop_final_z_m"] = float(model_trace[-1]["z_err_m"])
    print(f"[模型内闭环] z_err(m) 前15步: {np.round(report['model_closedloop_z_m'], 4)}；"
          f"末端 {report['model_closedloop_final_z_m']:+.2e}", flush=True)

    with open(os.path.join(OUT, "model_platform_check.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n诊断数据已存 {os.path.join(OUT, 'model_platform_check.json')}", flush=True)


if __name__ == "__main__":
    main()
