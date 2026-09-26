"""bench_step.py — IterHybridEnv 单步耗时分解实验（W04）

验证 W03 复现报告的发现：xarray 转换（get_state_data）占 step 耗时约 25%。
方法：monkey-patch ToraxApp.run / get_state_data / extract_state_observation 计时。
"""
import time
from collections import defaultdict

import numpy as np
from gymtorax import IterHybridEnv

N_STEPS = 30

env = IterHybridEnv(render_mode="none", store_history=False, log_level="critical")

timings = defaultdict(list)

orig_run = env.torax_app.run
orig_get_state = env.torax_app.get_state_data
orig_extract = env.observation_handler.extract_state_observation


def timed_run():
    t0 = time.perf_counter()
    out = orig_run()
    timings["torax_run(物理步进)"].append(time.perf_counter() - t0)
    return out


def timed_get_state():
    t0 = time.perf_counter()
    out = orig_get_state()
    timings["get_state_data(xarray转换)"].append(time.perf_counter() - t0)
    return out


def timed_extract(state):
    t0 = time.perf_counter()
    out = orig_extract(state)
    timings["extract_observation(观测提取)"].append(time.perf_counter() - t0)
    return out


env.torax_app.run = timed_run
env.torax_app.get_state_data = timed_get_state
env.observation_handler.extract_state_observation = timed_extract

obs, _ = env.reset()
t_all0 = time.perf_counter()
for i in range(N_STEPS):
    action = {"Ip": [3e6 + (i + 1) * 9.5e6 / 100], "NBI": [0.0, 0.25, 0.25], "ECRH": [0.0, 0.35, 0.05]}
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break
total = time.perf_counter() - t_all0

print(f"\n{'='*60}")
print(f"跑了 {len(timings['torax_run(物理步进)'])} 步，总计 {total:.1f}s")
print(f"{'模块':<30}{'合计(s)':>10}{'均值(ms)':>10}{'占比':>8}")
for name, ts in timings.items():
    s = sum(ts)
    print(f"{name:<30}{s:>10.2f}{np.mean(ts)*1000:>10.0f}{s/total*100:>7.1f}%")
env.close()
