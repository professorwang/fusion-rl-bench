"""smoke_test.py — FreeGSNKEShapeEnv 冒烟测试（reset + 3 步随机动作）

运行：.venv-gs\\Scripts\\python code\\fusion-rl-bench\\envs\\smoke_test.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from freegsnke_shape_env import FreeGSNKEShapeEnv  # noqa: E402

env = FreeGSNKEShapeEnv(ntheta=16, solver_rtol=1e-6)

t0 = time.perf_counter()
obs, _ = env.reset()
print(f"reset OK（{time.perf_counter()-t0:.1f}s），obs 维度 {obs.shape}，目标 LCFS 已建立")
print(f"action_space: {env.action_space}")

for i in range(3):
    a = env.action_space.sample() * 0.5
    t0 = time.perf_counter()
    obs, r, term, trunc, info = env.step(a)
    dt = time.perf_counter() - t0
    print(f"step {i}: reward={r:.4f} converged={info['converged']} 耗时={dt:.2f}s")
    if term:
        print("求解未收敛，episode 终止")
        break
print("SMOKE_OK")
