# fusion-rl-bench

**Reinforcement learning benchmarks for tokamak plasma control — with an honest evaluation protocol.**

一个面向托卡马克等离子体控制的强化学习实验库：真实物理模拟环境、经典控制器/模仿学习/RL 基线，以及**统一口径的配对评估流程**。（中文说明见下文）

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## Status (v0.1.1, 2026-09-26)

**What this is**: a working, reproducible experiment chain — Gymnasium environments on real physics backends, PD/BC/PPO baselines, and a unified paired evaluation harness (`scripts/fair_eval.py`).

**What we measured** (paired evaluation: 10 initial-condition seeds, same disturbance, same simulator instance, post-action sampling; errors per-episode then averaged; v0.1.3):

| Controller | Complete episodes | Whole-trajectory \|Z–Z*\| MAE |
|---|---|---|
| **Classical PD (Kp=3, Kd=0.3, P6 only — best of 36-pair gain surface)** | 10/10 | **0.014 cm** |
| Classical PD (Kp=1, Kd=0.3, P6 only) | 10/10 | 0.09 cm (0.08 cm on a second seed batch) |
| BC (imitating high-gain PD) | 10/10 | 0.19 cm |
| LQI (qi=50, r=1, integral action) | 10/10 | 0.91 cm |
| BC + PPO fine-tune (16,128 steps) | 10/10 | 1.27 cm |
| naive LQR (R=100) | 10/10 | 2.19 cm |
| LQI (r=1000, best single-channel) | 10/10 | 1.92 cm |

![controller family portrait](assets/lqr_lqi_eval.png) ![PD gain surface](assets/pd_gain_surface.png)

**The honest headline**: learned controllers beat their high-gain teacher — *but classical gain tuning is remarkably strong*. The PD gain surface (B2, 36 pre-specified gain pairs) shows a low-gain basin at **0.01–0.11 cm** (best: PD(3,0.3) = **0.014 cm**, near the model's resolution limit on this 25 ms linear task) and a failure region at high derivative gains. Trajectory comparison shows gain-saturated PD chatters (rail-to-rail actions injecting a limit cycle), while low-gain or smoothed control avoids it. The "BC learns smoothing from noisy expert data" reading is currently a **mechanism hypothesis**, not an established finding — it is a target of our stage-B ablation.

**What this does NOT establish**: single training seed for the learned controllers, a **linearized MAST-U-like model** (not an exact MAST-U replica), 25 ms simulated episodes, **asymmetric actuator authority** (PD/LQI-P6 single-channel vs learned 3-channel), and model-based baselines limited by the identified model's validity region. On this task, a well-tuned classical controller is the ceiling — learned controllers must prove their value where classical control cannot go (multiple operating points, nonlinearity, noise, real-device constraints): that is our stage-B/C program. The frozen PPO was initialized from an earlier BC version, so BC→PPO attribution is reported separately.

> v0.1.3→v0.2.0 changelog: B0 (discrete model validated: dominant eigenvalue within 0.14% of simulator) + B1 (naive LQR shows steady offset; LQI fixes it to 0.91cm) + B2 (PD gain surface: low-gain basin 0.014cm) — see `notebooks/B0`、`notebooks/B1`。

> v0.1.2→v0.1.3 changelog: added pre-specified low-gain PD control (PD(1,0.3)); fixed invalid-reset samples leaking into summary metrics (evaluator now filters `sample_valid` in all summaries; training eval skips anomalous resets); mechanism framing downgraded to hypothesis.

## Environments

| Env | Physics | Task | Speed |
|---|---|---|---|
| `envs/freegsnke_linear_pos_env.py` (recommended) | FreeGSNKE linear evolution (`nl_solver`, `linear_only=True`) | vertical/radial position hold under random initial disturbances | ~50 ms/step |
| `envs/freegsnke_shape_env.py` | FreeGSNKE static Newton–Krylov solves | LCFS shape control (advanced) | slower; see notes |

Physical facts measured on this model (baseline equilibrium, 65×129 grid — model-specific, not device measurements):

- Vertical instability growth rate ≈ **233/s** (e-folding ≈ 4.3 ms) from the linearization diagnostics.
- Actuator authority scan (100 V, 5 ms pulse): coil **P6 ≈ 3.4 mm/kV** on vertical position vs 0.01–0.02 mm/kV for the other 11 active coils.
- Observation must include the **time derivative** of position error for a memoryless policy to imitate a PD law (position alone is not a learnable mapping).

## Repository layout

```
envs/           Gymnasium environments (FreeGSNKE-based)
scripts/        train_ppo.py / pd_baseline.py / bc_warmup.py / fair_eval.py / plot_curve.py
notebooks/      Engineering reports (Chinese): reproduction, profiling, post-mortems, evaluation fixes
assets/         Figures
results/        Paired evaluation raw per-episode data (JSON)
docs/           Setup guide and troubleshooting log (12+ real pitfalls)
```

## Quickstart

### 1. Install (tested on Python 3.13.5 / Windows; other versions untested)

```bash
python -m venv .venv && source .venv/Scripts/activate  # Windows Git Bash; Linux: source .venv/bin/activate
pip install gymnasium stable-baselines3 torch matplotlib shapely

# FreeGSNKE upstream (contains machine configs + example data)
git clone https://github.com/FusionComputingLab/freegsnke
pip install -e ./freegsnke
export FREEGSNKE_REPO=$PWD/freegsnke   # envs read this variable
```

> Historical note: FreeGSNKE previously pinned `numpy~=1.26` while TORAX needs `numpy>2` — use **separate virtual environments** for TORAX-based and FreeGSNKE-based work (see `docs/SETUP.md`). Current upstream constraint is `numpy>=1.26.4,<2.5`.

### 2. Smoke test

```bash
python envs/smoke_test_linear.py   # builds MAST-U-like machine, solves baseline equilibrium, steps actions
```

### 3. Reproduce the baselines

```bash
python scripts/pd_baseline.py                          # PD grid search (10 episodes per gain)
python scripts/bc_warmup.py                            # PD expert data + behavior cloning
BC_INIT=runs/bc_policy.pt python scripts/train_ppo.py  # PPO fine-tune (own run dir per launch)
python scripts/fair_eval.py --episodes 10 --seed0 3000 \
    --bc runs/bc_policy.pt --ppo <run_dir>/ppo_final.zip \
    --out results/fair_eval.json                       # unified paired evaluation
```

Frozen models and raw evaluation data are attached to [GitHub Releases](https://github.com/professorwang/fusion-rl-bench/releases) (bc_policy.pt, ppo_final.zip, fair_eval JSON) with SHA-256 hashes.

## Roadmap (verification before claims)

- [x] **B0 — discrete model validation**: sysid at actual 0.5 ms control period; one-step Zcur err 1.3%; dominant eigenvalue within 0.14% of simulator (1.1254 vs 1.1238) (`notebooks/B0`)
- [x] **B1 — LQR/LQI baselines**: naive DARE-LQR shows steady-state offset (model mismatch + no integral action); LQI fixes it (0.91 cm, 3ch) but still trails PD(1,0.3)=0.09cm and BC=0.19cm; unit-confusion pitfalls documented (`notebooks/B1`)
- [x] B2 — PD gain-family performance surface (low-gain basin 0.014cm, best PD(3,0.3); failure region at high Kd) (`assets/pd_gain_surface.png`)
- [x] B3 — mechanism ablation batch 1 (teacher gain × expert noise): **pre-registered noise-shrinkage hypothesis falsified** — BC beats chattering teacher 10× even with zero expert noise; mechanism revised to "function-approximation smoothing of the saturated law" (`notebooks/B3`)
- [ ] ≥5 independent training seeds; frozen 100–200 test ICs; confidence intervals
- [ ] Robustness: observation noise, actuation latency, parameter error, sustained disturbances
- [ ] Re-check in a convergent nonlinear evolution model
- [ ] Real-device path (with university partners): model validation → sim closed-loop → shadow mode → approved experiment

## Citation

```bibtex
@software{fusion_rl_bench_2026,
  title  = {fusion-rl-bench: Reinforcement learning benchmarks for tokamak plasma control},
  author = {fusion-rl-bench contributors},
  year   = {2026},
  url    = {https://github.com/professorwang/fusion-rl-bench}
}
```

Also cite FreeGSNKE (Amorisco et al., *Physics of Plasmas* 31, 042517, 2024) and TORAX (Google DeepMind) when applicable.

## License

[Apache-2.0](LICENSE). © 2026 fusion-rl-bench contributors. FreeGSNKE/TORAX and other upstream dependencies retain their own licenses.

---

## 中文简介

本仓库提供基于真实物理后端（UKAEA FreeGSNKE 平衡求解器、DeepMind TORAX 输运模拟器）的托卡马克等离子体控制强化学习环境，以及经典 PD / 行为克隆 / PPO 微调的完整基线链与**统一口径的配对评估流程**。

**当前状态（v0.2.0）**：工程链路完整可复现，且已完成 B0（离散模型验证）与 B1（LQR/LQI 基线）。统一口径配对评估全家福：低增益盆地 PD(3,0.3)=**0.014cm**（36 组增益性能面的最优）→ PD(1,0.3) 0.09cm → BC 0.19cm → LQI(三通道） 0.91cm → PPO 1.27cm → 朴素 LQR 2.19cm（稳态偏差，已诊断）。诚实的结论是：**在这个任务上，会调参的经典控制就是天花板**；朴素 DARE-LQR 并不自动赢（稳态偏差+无积分作用），模型驱动设计的每一步都必须数值验证。AI 控制器要证明价值，必须去经典控制力所不能及的地方（多工况/非线性/噪声/真机约束）——这是阶段 B/C 计划。限制：学习类为单训练种子、线性化 MAST-U-like 模型、25 毫秒回合、执行器权限不对等。此前基于口径不一致数据的"超越"表述已撤回（见 `notebooks/W06`）。

工程实录（复现报告、性能分析、三次翻车根因、评估方法返工）见 `notebooks/`；环境搭建与排坑记录见 `docs/SETUP.md`。

交流与合作：等离子体控制、聚变数字化方向，欢迎 issue 联系。
