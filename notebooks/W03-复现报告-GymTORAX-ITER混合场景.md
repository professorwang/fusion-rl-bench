# W03 复现报告：Gym-TORAX ITER 混合场景 baseline

> 日期：2026-09-17 ｜ 执行：A 角 ｜ 对应《AI团队工作方案》W03 交付物

## 一、复现对象

| 项 | 内容 |
|---|---|
| 环境 | `gymtorax.IterHybridEnv`（Gymnasium 封装 → DeepMind TORAX 1.4.3，1.5D 输运） |
| 场景 | ITER hybrid scenario：150 秒放电；Ip 3 MA → 12.5 MA 斜坡爬升（t<99s 线性，之后保持）；t=99s 起 NBI 33 MW、ECRH 20 MW 注入（r=0.25/0.35，w=0.25/0.05） |
| 智能体 | 脚本化 IterHybridAgent（按场景时间线开环输出动作）——**这正是 RL 要替代的"人工排程"** |
| 代码 | `code/vendor/gymtorax/examples/iter_hybrid.py`（commit: depth-1 clone, 2026-09） |

## 二、运行结果 ✅

- **全场景跑通**：150 秒物理仿真完整结束（进度条 100%），无物理报错（TORAX `check_for_errors` 通过）；
- **耗时**：总 wall-clock **72.1 秒**（pyinstrument 采样 45493 次），其中仿真推进 57 秒 → 约 **2.6× 实时**（CPU，Windows，无 GPU）；
- **性能分解**（pyinstrument，可直接用于优化决策）：

| 模块 | 耗时 | 占比 | 备注 |
|---|---|---|---|
| `IterHybridEnv.step` 合计 | 57.0 s | 79% | 主循环 |
| └ `ToraxApp.run`（物理步进） | 27.4 s | 38% | 其中 **JAX 编译 cache_miss 22.7s**（含 backend_compile 14.9s，一次性成本） |
| └ `get_state_data`（xarray 转换） | 18.2 s | 25% | **每步都在做 `StateHistory.simulation_output_to_xr`——明显可优化点** |
| └ `update_config`（动作注入） | 10.3 s | 14% | pydantic/RuntimeParams 重建 |
| 环境初始化 | 14.8 s | 21% | 含首次 JIT 编译 8.6s |

## 三、排坑记录（新增，已同步 SETUP.md 候选）

7. `iter_hybrid.py` 依赖 `pyinstrument`（dev 依赖，主依赖不带），需手动 `pip install pyinstrument`；
8. `pid_optimization.py` 设置 `text.usetex=True`，无 LaTeX 环境会直接报错——复现前需改为 False；
9. `iter_hybrid_random.py` 是 `while True` 无限循环，做随机 baseline 统计时必须自加 episode 上限。

## 四、对主作品（fusion-rl-bench）的四个启示

1. **控制层级差异 = 我们的补位空间**：Gym-TORAX 的动作是 scenario 级（Ip、NBI、ECRH 功率），**不是线圈电压级**——它服务"放电轨迹优化"，而非 DeepMind TCV 那种"磁体形状控制"。我们的 FreeGSNKE 环境（线圈级自由边界控制）恰好填补后者，两个层级都覆盖才是完整故事。
2. **训练时必须关掉 xarray 序列化**：25% 耗时在 state→xarray 转换上；RL 训练循环里应直接取 ndarray 观测（后续读代码确认 `store_history=False` 是否已跳过——profile 显示仍在调用，需 hack）。
3. **JAX 编译预热**：一次性编译 ~15s，训练前用 dummy step 预热，避免污染计时。
4. **奖励设计是空白**：iter_hybrid.py 完全没看 reward（纯开环排程）；Gym-TORAX 的 reward 定义（见 `pid_optimization.py` 的 reward_breakdown）将是 W04 代码精读重点。

## 五、下周（W04）动作

- [ ] 物理速成 2：输运方程、破裂物理；精读 FRNN 两篇论文（笔记入库）
- [ ] 精读 gymtorax 源码：`base_env.py`（observation/reward 组装）、`action_handler`、`config_loader`
- [ ] 验证 `store_history=False` 路径能否跳过 xarray 转换（预期 step 提速 ~25%）
- [ ] 跑通 RandomAgent 有限 episode 统计（作为 baseline 下界）

## 附：复现命令

```bash
cd code/vendor/gymtorax/examples
/f/claude/nuclear/.venv/Scripts/python iter_hybrid.py   # PROFILE=True, RENDER_MODE="none"
# 日志：/f/claude/nuclear/gymtorax-iterhybrid.log
# 性能报告：code/vendor/gymtorax/examples/profile_output.html（40MB，浏览器打开）
```
