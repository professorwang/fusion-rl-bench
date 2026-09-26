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
    """周期性评估：独立环境跑 EVAL_EPISODES 个 episode，记录位置误差（Zcur+Rin 偏差）。"""

    def __init__(self):
        super().__init__()
        self.rows = []
        self.eval_env = make_env()

    def _on_step(self) -> bool:
        if self.num_timesteps % EVAL_EVERY == 0 and self.num_timesteps > 0:
            rewards, final_errs, fails = [], [], 0
            for _ in range(EVAL_EPISODES):
                obs, info = self.eval_env.reset()
                done = False
                cum = 0.0
                last_d = np.zeros(4)
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, r, term, trunc, i2 = self.eval_env.step(action)
                    cum += r
                    if not term:
                        last_d = obs[:4]
                    done = term or trunc
                    if term:
                        fails += 1
                rewards.append(cum)
                final_errs.append((abs(last_d[2]) + abs(last_d[3])) * 0.1)  # 归一化→米
            row = {
                "steps": self.num_timesteps,
                "eval_reward_mean": float(np.mean(rewards)),
                "final_pos_err_mean": float(np.nanmean(final_errs)),
                "fail_rate": fails / EVAL_EPISODES,
            }
            self.rows.append(row)
            print(f"[eval@{row['steps']}] reward={row['eval_reward_mean']:.3f} "
                  f"final_pos_err={row['final_pos_err_mean']*100:.2f}cm "
                  f"fail={row['fail_rate']:.0%}", flush=True)
            with open(os.path.join(OUT, "train_log.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
                w.writeheader()
                w.writerows(self.rows)
            self.model.save(os.path.join(OUT, f"ppo_{self.num_timesteps}"))
        return True


def main():
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
            # BC 权重预热：bc net[0,2]→policy_net，net[4]→action_net；log_std 压低使初始策略近确定性
            import torch
            sd = torch.load(bc_init, map_location="cpu")
            pnet = model.policy.mlp_extractor.policy_net
            anet = model.policy.action_net
            with torch.no_grad():
                pnet[0].weight.copy_(sd["net.0.weight"]); pnet[0].bias.copy_(sd["net.0.bias"])
                pnet[2].weight.copy_(sd["net.2.weight"]); pnet[2].bias.copy_(sd["net.2.bias"])
                anet.weight.copy_(sd["net.4.weight"]); anet.bias.copy_(sd["net.4.bias"])
                model.policy.log_std.fill_(-1.0)
            print(f"BC 预热权重已加载（{bc_init}），log_std=-1.0", flush=True)
    t0 = time.time()
    model.learn(total_timesteps=TOTAL_STEPS, callback=EvalCallback(), reset_num_timesteps=not resume)
    model.save(os.path.join(OUT, "ppo_final"))
    print(f"训练完成，用时 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
