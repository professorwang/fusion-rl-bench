"""fair_eval.py — 统一口径配对评估（对应 2026-09-26 审核阶段A第3项）

所有控制器在**相同初始条件种子、相同扰动、同一模拟器实例、动作后采样**下比较。
指标（逐 episode 计算后再跨 episode 取均值）：
- mae_z_cm      全过程垂直绝对误差均值（存活步，动作后采样）
- mae_joint_cm  全过程 |ΔZ|+|ΔRin| L1 误差均值（存活步）
- final_joint_cm 末端联合 L1 误差
- holds         存活控制步数（每步 0.5ms）
- complete      是否完整跑满 max_steps（未失败）
- fail_reason   失败原因（如有）
- act_mean / sat_ratio  控制量均值与饱和比例（|a|>0.99）

失败 episode 不删除：单独计数，不混入误差均值（避免"失败但误差零"）。

用法：
  python scripts/fair_eval.py --episodes 10 --seed0 3000 \
      --bc runs/bc_policy.pt --ppo runs/ppo_final.zip --pd-kp 100 --pd-kd 100 \
      --out results/fair_eval.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))
from freegsnke_linear_pos_env import FreeGSNKELinearPosEnv  # noqa: E402


def make_env():
    return FreeGSNKELinearPosEnv(
        control_coils=["P6", "D5", "P5"], max_steps=50, steps_per_action=1,
        max_voltage=500.0, init_disturb_steps=3, init_disturb_voltage=50.0,
    )


class PDController:
    def __init__(self, kp, kd, p6_idx, dt):
        self.kp, self.kd, self.p6_idx, self.dt = kp, kd, p6_idx, dt

    def reset(self, obs):
        self.z_prev = obs[2]

    def act(self, obs):
        z_err = obs[2]
        z_dot = (obs[2] - self.z_prev) / self.dt
        self.z_prev = obs[2]
        a = np.zeros(3)
        a[self.p6_idx] = np.clip(-(self.kp * z_err + self.kd * z_dot * 1e-3), -1, 1)
        return a


class BCController:
    def __init__(self, path):
        import torch
        sys.path.insert(0, os.path.dirname(__file__))
        from bc_warmup import BCPolicy
        sd = torch.load(path, map_location="cpu")
        obs_dim = 6  # v0.1.1 观测：d(4)+d_dot(2)
        self.policy = BCPolicy(obs_dim, 3)
        self.policy.load_state_dict(sd)
        self.policy.eval()
        self.torch = torch

    def reset(self, obs):
        pass

    def act(self, obs):
        with self.torch.no_grad():
            return self.policy(self.torch.as_tensor(obs, dtype=self.torch.float32)).numpy()


class PPOController:
    def __init__(self, path):
        from stable_baselines3 import PPO
        self.model = PPO.load(path)

    def reset(self, obs):
        pass

    def act(self, obs):
        a, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(a, dtype=np.float64)


def run_episode(env, controller, seed, max_steps=50):
    obs, _ = env.reset(seed=seed)
    controller.reset(obs)
    z_errs, joint_errs, acts = [], [], []
    holds, done = 0, False
    final_joint = np.nan
    while not done:
        a = np.clip(controller.act(obs), -1, 1)
        acts.append(a)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        if not term:
            holds += 1
            z_errs.append(abs(obs[2]) * 0.1)
            joint_errs.append((abs(obs[2]) + abs(obs[3])) * 0.1)
            final_joint = joint_errs[-1]
    acts = np.abs(np.array(acts)) if acts else np.zeros((1, 3))
    return {
        "seed": seed,
        "complete": bool(holds >= max_steps),
        "holds": holds,
        "mae_z_cm": float(np.mean(z_errs) * 100) if z_errs else None,
        "mae_joint_cm": float(np.mean(joint_errs) * 100) if joint_errs else None,
        "final_joint_cm": float(final_joint * 100) if np.isfinite(final_joint) else None,
        "fail_reason": info["fail_reason"],
        "act_mean": float(acts.mean()),
        "sat_ratio": float((acts > 0.99).mean()),
    }


def summarize(name, rows):
    def m(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else float("nan")
    n_complete = sum(1 for r in rows if r["complete"])
    return {
        "controller": name,
        "episodes": len(rows),
        "complete": n_complete,
        "mae_z_cm": m("mae_z_cm"),
        "mae_joint_cm": m("mae_joint_cm"),
        "final_joint_cm": m("final_joint_cm"),
        "act_mean": m("act_mean"),
        "sat_ratio": m("sat_ratio"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=3000)
    ap.add_argument("--pd-kp", type=float, default=100.0)
    ap.add_argument("--pd-kd", type=float, default=100.0)
    ap.add_argument("--bc", type=str, default="")
    ap.add_argument("--ppo", type=str, default="")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    env = make_env()
    dt = env.full_timestep * env.steps_per_action
    p6_idx = env.control_coils.index("P6")

    controllers = {"PD(Kp=%g,Kd=%g,P6单通道)" % (args.pd_kp, args.pd_kd):
                   PDController(args.pd_kp, args.pd_kd, p6_idx, dt)}
    if args.bc and os.path.exists(args.bc):
        controllers["BC"] = BCController(args.bc)
    if args.ppo and os.path.exists(args.ppo):
        controllers["PPO"] = PPOController(args.ppo)

    results = {}
    for name, ctrl in controllers.items():
        print(f"评估 {name} …", flush=True)
        rows = [run_episode(env, ctrl, args.seed0 + i) for i in range(args.episodes)]
        results[name] = {"episodes": rows, "summary": summarize(name, rows)}

    print(f"\n{'控制器':<32}{'完整回合':>8}{'MAE_Z(cm)':>12}{'MAE联合(cm)':>12}{'末端联合(cm)':>12}{'饱和率':>8}")
    for name, data in results.items():
        s = data["summary"]
        print(f"{name:<32}{s['complete']:>5}/{s['episodes']:<3}{s['mae_z_cm']:>12.3f}"
              f"{s['mae_joint_cm']:>12.3f}{s['final_joint_cm']:>12.3f}{s['sat_ratio']:>8.2%}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        payload = {
            "protocol": {
                "episodes": args.episodes, "seed0": args.seed0,
                "sampling": "post-action, paired seeds, same simulator instance",
                "note": "失败episode不计入误差均值；误差先按回合计算再跨回合取均值",
            },
            "results": results,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n原始逐回合数据已存 {args.out}")


if __name__ == "__main__":
    main()
