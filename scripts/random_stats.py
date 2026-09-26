"""random_stats.py — RandomAgent 有限 episode 统计（baseline 下界，W04）

官方 iter_hybrid_random.py 是无限循环，本脚本加 episode 上限。
随机策略预期：动作越界/物理崩溃 → 提前 terminated 且 reward=-1000。
"""
import numpy as np
from gymtorax import IterHybridEnv, RandomAgent

N_EPISODES = 3
MAX_STEPS = 160

env = IterHybridEnv(render_mode="none", store_history=False, log_level="critical")
agent = RandomAgent(env.action_space)

print(f"action_space: {env.action_space}")
for ep in range(N_EPISODES):
    obs, _ = env.reset()
    cum_reward, n = 0.0, 0
    terminated = False
    while not terminated and n < MAX_STEPS:
        obs, reward, terminated, _, info = env.step(agent.act(obs))
        cum_reward += reward
        n += 1
    print(f"episode {ep}: 存活 {n} 步（t≈{env.current_time:.0f}s/150s），累计奖励 {cum_reward:.1f}，terminated={terminated}")
env.close()
