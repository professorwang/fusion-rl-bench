"""train_ppo.py — PPO baseline：FreeGSNKEShapeEnv 形状保持任务（W05–W06）

训练：PPO (MlpPolicy)，粗网格快档；定期评估（细网格评估暂同环境）。
产出：checkpoints/ppo_*.zip、train_log.csv、训练曲线 training_curve.png。

运行（后台）：
  .venv-gs\\Scripts\\python code\\fusion-rl-bench\\scripts\\train_ppo.py
"""
import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "envs"))
from freegsnke_linear_pos_env import FreeGSNKELinearPosEnv  # noqa: E402

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

TOTAL_STEPS = 8000
EVAL_EVERY = 1000
EVAL_EPISODES = 5
OUT = os.path.join(_HERE, "..", "runs")
os.makedirs(OUT, exist_ok=True)


def make_env():
    # 控制权限扫描实测：P6 对垂直位置的灵敏度是其他线圈 ~200 倍（3.4mm/kV vs 0.01–0.02）。
    # 聚焦 Top-3 可控线圈 + ±500V 权限 + 稠密有界奖励（见 env 注释）+ 初始扰动起点。
    return FreeGSNKELinearPosEnv(
        control_coils=["P6", "D5", "P5"],
        max_steps=50, steps_per_action=1, max_voltage=500.0,
        init_disturb_steps=3, init_disturb_voltage=50.0,
    )


class EvalCallback(BaseCallback):
    """周期性评估：独立环境跑 EVAL_EPISODES 个 episode。

    v0.1.1（审核 R3/R6）：
    - 失败与有效轨迹误差**分开报告**（失败回合不计入误差均值，避免"失败但误差零"）；
    - 评估 episode 使用固定种子序列（2000+），保证各检查点可比；
    - 每次运行写入独立 run 目录，CSV 采用追加模式，不再覆盖历史（审核 P1）。
    """

    def __init__(self, run_dir):
        super().__init__()
        self.rows = []
        self.run_dir = run_dir
        self.eval_env = make_env()

    def _on_step(self) -> bool:
        if self.num_timesteps % EVAL_EVERY == 0 and self.num_timesteps > 0:
            rewards, valid_errs, fails, ran = [], [], 0, 0
            for ep in range(EVAL_EPISODES):
                obs, info = self.eval_env.reset(seed=2000 + ep)  # 固定评估种子
                if info.get("reset_anomaly") or info.get("obs_error"):
                    # v0.1.2 复核：异常 reset 样本显式剔除，不得混入评估
                    print(f"  [eval] 跳过异常 reset 样本 (seed={2000+ep})", flush=True)
                    continue
                ran += 1
                done = False
                cum = 0.0
                ep_errs = []
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, r, term, trunc, i2 = self.eval_env.step(action)
                    cum += r
                    if not term:
                        ep_errs.append((abs(obs[2]) + abs(obs[3])) * 0.1)  # 归一化→米
                    done = term or trunc
                    if term:
                        fails += 1
                rewards.append(cum)
                # v0.1.1 复核（C项）：失败回合的存活段误差不计入误差均值（与 fair_eval 一致）
                if ep_errs and not term:
                    valid_errs.append(float(np.mean(ep_errs)))
            row = {
                "steps": self.num_timesteps,
                "eval_reward_mean": float(np.mean(rewards)),
                "valid_pos_err_mean": float(np.mean(valid_errs)) if valid_errs else float("nan"),
                "n_valid": len(valid_errs),
                "fail_rate": fails / max(ran, 1),
            }
            self.rows.append(row)
            print(f"[eval@{row['steps']}] reward={row['eval_reward_mean']:.3f} "
                  f"valid_pos_err={row['valid_pos_err_mean']*100:.2f}cm (n={row['n_valid']}) "
                  f"fail={row['fail_rate']:.0%}", flush=True)
            csv_path = os.path.join(self.run_dir, "train_log.csv")
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
                if write_header:
                    w.writeheader()
                w.writerow(row)
            self.model.save(os.path.join(self.run_dir, f"ppo_{self.num_timesteps}"))
        return True


def main():
    # v0.1.1：每次运行独立目录（追加而非覆盖；审核 P1）。可用 RUN_DIR 覆盖。
    run_dir = os.environ.get("RUN_DIR") or os.path.join(
        OUT, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    env = make_env()
    resume = os.environ.get("RESUME", "")
    bc_init = os.environ.get("BC_INIT", "")
    if resume and os.path.exists(resume + ".zip"):
        model = PPO.load(resume, env=env)
        print(f"从 {resume}.zip 恢复训练", flush=True)
    else:
        model = PPO(
            "MlpPolicy", env, verbose=0, seed=0,
            n_steps=128, batch_size=64, n_epochs=8,
            learning_rate=3e-4, gamma=0.99, ent_coef=0.01,
            policy_kwargs=dict(net_arch=[64, 64]),
        )
        if bc_init and os.path.exists(bc_init):
            # BC 权重预热：bc net[0,2]→policy_net，net[4]→action_net。
            # 注意（审核 R4）：BC 输出层 Tanh 与 PPO 线性均值+裁剪的映射并不等价，
            # 迁移后策略函数会发生变化；log_std=-1.0 对应归一化标准差≈0.368，
            # 在 500V 尺度下约 184V，不能视为"接近确定性"。
            import torch
            sd = torch.load(bc_init, map_location="cpu")
            pnet = model.policy.mlp_extractor.policy_net
            anet = model.policy.action_net
            with torch.no_grad():
                pnet[0].weight.copy_(sd["net.0.weight"]); pnet[0].bias.copy_(sd["net.0.bias"])
                pnet[2].weight.copy_(sd["net.2.weight"]); pnet[2].bias.copy_(sd["net.2.bias"])
                anet.weight.copy_(sd["net.4.weight"]); anet.bias.copy_(sd["net.4.bias"])
                model.policy.log_std.fill_(-1.0)
            print(f"BC 预热权重已加载（{bc_init}），log_std=-1.0（映射差异见代码注释 R4）", flush=True)
    t0 = time.time()
    model.learn(total_timesteps=TOTAL_STEPS, callback=EvalCallback(run_dir), reset_num_timesteps=not resume)
    model.save(os.path.join(run_dir, "ppo_final"))
    print(f"训练完成，用时 {(time.time()-t0)/60:.1f} 分钟，输出目录 {run_dir}", flush=True)


if __name__ == "__main__":
    main()
