"""sysid_macos.py — B6 第二部分（事后补救，明确标注）：在当前 macOS 机器上重新辨识离散模型

背景（诚实记录）：B6 首轮按预登记使用 Windows 冻结模型 results/sysid_model.npz，
全部模型驱动控制器（含 B1 全状态 LQI 复跑）在本机 10/10 回合 1–6 步内 position_lost。
scripts/model_platform_check.py 诊断：基线平衡逐位一致，但一步预测误差在本机放大
~30 倍（Zcur 40.2% vs B0 冻结值 1.3%）——冻结模型在本平台失效（跨平台数值漂移
对模型驱动控制是致命的，而不仅仅是 B5 记录的 MAE ×1.8 漂移）。

本脚本用与 B0 完全相同的协议在本机重新辨识（激励 ±30V、100 rollout × 40 步、
最小二乘），输出到**新文件** results/sysid_model_macos.npz 与
results/sysid_validation_macos.json——冻结资产 sysid_model.npz 保持原样不动。

注意：辨识得到的增益与 B1 仅在"同一权重矩阵"意义下可比（Q/R 权重相同），
(A,B) 本身不同平台不同——所有结论以同机配对评估为准。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))
sys.path.insert(0, os.path.dirname(__file__))
from freegsnke_linear_pos_env import FreeGSNKELinearPosEnv  # noqa: E402
from sysid_linear import get_state, apply_voltage  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

CTRL = ["P6", "D5", "P5"]
EXCITE_V = 30.0
ROLLOUTS, STEPS = 100, 40
SEED = 42


def main():
    env = FreeGSNKELinearPosEnv(control_coils=CTRL, max_steps=10**9,
                                steps_per_action=1, max_voltage=500.0)
    env.reset(seed=0)
    x0 = get_state(env)
    rng = np.random.default_rng(SEED)

    DX, DU, DY = [], [], []
    for r in range(ROLLOUTS):
        env.reset(seed=SEED * 1000 + r)
        for _ in range(STEPS):
            x_k = get_state(env) - x0
            u_k = rng.uniform(-EXCITE_V, EXCITE_V, size=len(CTRL))
            apply_voltage(env, u_k)
            DX.append(x_k)
            DU.append(u_k)
            DY.append(get_state(env) - x0)
        if (r + 1) % 20 == 0:
            print(f"辨识采样 {r + 1}/{ROLLOUTS} rollout", flush=True)
    DX, DU, DY = np.array(DX), np.array(DU), np.array(DY)

    Z = np.hstack([DX, DU])
    Theta, *_ = np.linalg.lstsq(Z, DY, rcond=None)
    A = Theta[:16].T
    B = Theta[16:].T
    np.savez(os.path.join(OUT, "sysid_model_macos.npz"), A=A, B=B, x0=x0, ctrl=CTRL)
    print(f"macOS 模型已存 {os.path.join(OUT, 'sysid_model_macos.npz')}", flush=True)

    # 一步预测验证（留出 rollout，与 B0 同协议）
    env.reset(seed=999)
    rel = {c: [] for c in ["coils", "Rx", "Zx", "Zcur", "Rin"]}
    for _ in range(50):
        x_k = get_state(env) - x0
        u_k = rng.uniform(-EXCITE_V, EXCITE_V, size=len(CTRL))
        apply_voltage(env, u_k)
        x_true = get_state(env) - x0
        e = np.abs(x_true - (A @ x_k + B @ u_k)) / np.maximum(np.abs(x_true), 1e-6)
        rel["coils"].append(e[:12].mean())
        for j, c in enumerate(["Rx", "Zx", "Zcur", "Rin"]):
            rel[c].append(e[12 + j])
    one_step = {c: float(np.mean(v)) for c, v in rel.items()}
    eig_max = float(np.max(np.abs(np.linalg.eigvals(A))))
    print(f"[macOS 模型] 一步预测相对误差: { {c: round(v, 4) for c, v in one_step.items()} }", flush=True)
    print(f"[macOS 模型] A 最大特征值(模): {eig_max:.4f}（Windows 冻结模型 1.1254）", flush=True)

    metrics = {
        "note": "macOS (x86_64) 本机重新辨识；协议与 B0 相同；冻结 Windows 模型未改动",
        "state_dim": 16, "input_dim": 3, "control_period_s": 5e-4,
        "samples": len(DX), "rollouts": ROLLOUTS, "steps_per_rollout": STEPS,
        "one_step_mean_rel_err": one_step,
        "A_max_eigval_abs": eig_max,
    }
    with open(os.path.join(OUT, "sysid_validation_macos.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"验证数据已存 {os.path.join(OUT, 'sysid_validation_macos.json')}", flush=True)


if __name__ == "__main__":
    main()
