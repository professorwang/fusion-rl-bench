"""smoke_test_linear.py — FreeGSNKELinearPosEnv 冒烟测试"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from freegsnke_linear_pos_env import FreeGSNKELinearPosEnv  # noqa: E402

env = FreeGSNKELinearPosEnv()
t0 = time.perf_counter()
obs, info = env.reset(seed=0)
print(f"reset（含基线静态求解）: {time.perf_counter()-t0:.1f}s | obs={obs} | target={info.get('target')}")

ts = []
for i in range(10):
    t0 = time.perf_counter()
    obs, r, term, trunc, info = env.step(env.action_space.sample() * 0.5)
    ts.append(time.perf_counter() - t0)
    print(f"step{i}: r={r:.5f} fail={info['fail_reason']} dt={ts[-1]*1000:.1f}ms")
    if term:
        break
import numpy as np
print(f"步均 {np.mean(ts)*1000:.1f}ms | SMOKE_LINEAR_OK")
