# 环境搭建与排坑记录（SETUP）

> 本文件持续更新，最终将成为《Hello Plasma：AI 人的聚变环境搭建指南》（build log 第 1 篇）的素材。

## 目标环境

| 组件 | 用途 | 安装方式 |
|---|---|---|
| Python 3.13.5（Windows） | 基础 | 系统已有 |
| venv `.venv`（TORAX 主线环境） | TORAX + Gymnasium | `python -m venv .venv` ✅ 已建 |
| venv `.venv-gs`（FreeGSNKE 环境） | 自由边界平衡求解 | `python -m venv .venv-gs` ✅ 已建（**必须分开装，见排坑 1**） |
| **TORAX**（DeepMind） | 可微分 1.5D 输运模拟器 | `.venv` 中 `pip install torax` 🔄 安装中（日志 `pip-install-torax.log`） |
| gymnasium | RL 标准接口 | 两个环境均装 🔄 |
| FreeGSNKE（UKAEA） | 自由边界平衡求解器 | `.venv-gs` 中 `pip install freegsnke` 🔄（日志 `pip-install-gs.log`） |
| Gym-TORAX | TORAX 的 Gym 封装 | 已从 GitHub 克隆到 `code/vendor/gymtorax`（[antoine-mouchamps/gymtorax](https://github.com/antoine-mouchamps/gymtorax)，文档：[gymtorax.readthedocs.io](https://gymtorax.readthedocs.io/)）；待 `pip install -e` 到 `.venv` |

## 上游源码（vendor）

| 仓库 | 位置 | 用途 |
|---|---|---|
| FreeGSNKE | `code/vendor/freegsnke`（[FusionComputingLab/freegsnke](https://github.com/FusionComputingLab/freegsnke)） | 官方示例 notebooks + `machine_configs/`（MAST-U 装置描述数据）；正用 jupyter nbconvert 执行 example02（静态正向求解）验证 |
| Gym-TORAX | `code/vendor/gymtorax` | W03 复现对象的官方 baseline |

## 安装步骤（复现用）

```bash
cd /f/claude/nuclear
# 环境 1：TORAX 主线（numpy>2）
python -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install torax gymnasium
.venv/Scripts/python -c "import torax; print(torax.__version__)"

# 环境 2：FreeGSNKE（numpy 1.26）
python -m venv .venv-gs
.venv-gs/Scripts/python -m pip install --upgrade pip
.venv-gs/Scripts/python -m pip install freegsnke gymnasium matplotlib
```

## 排坑记录

1. **TORAX 与 FreeGSNKE 依赖冲突（不可同环境）**：`torax 0.3.2` 要求 `numpy>2`，而 `freegsnke 3.0.1` 锁死 `numpy~=1.26.4`，pip 直接 ResolutionImpossible。解法：**双 venv**——`.venv` 装 TORAX 主线，`.venv-gs` 装 FreeGSNKE；跨环境协作通过文件交换（配置 JSON / 结果 npz）或后续把 FreeGSNKE 作为子进程调用。
2. **pip 安装超时**：依赖较多（JAX 生态），600s 窗口不够；改后台不限时任务 + 日志落盘。
3. **run_torax 交互式提示**：模拟跑完后 CLI 会询问下一步（r/mc/cc/pr/q），脚本化运行必须 `echo q | run_torax --config=...`，否则 EOFError。
4. **freegsnke 缺 freegs4e**：pip 装完 freegsnke 3.0.1 后 `build_machine` 报 `No module named 'freegs4e'`，需手动 `pip install freegs4e`。
5. **mastu_tools 需要 pyuda**：`freegsnke.mastu_tools` 依赖 UKAEA 内网数据客户端 pyuda（无需安装，仅影响真实 MAST-U 数据拉取，不影响平衡求解）。
6. **PyPI 版与 GitHub 版 API 不一致**：PyPI 的 `freegsnke 3.0.1` 缺少 `NKGSsolver(gs_operator_order=...)` 参数，官方示例 notebook 直接报错。解法：克隆官方仓库后 `pip install -e code/vendor/freegsnke` 以源码版覆盖安装。
7. `gymtorax/examples/iter_hybrid.py` 依赖 `pyinstrument`（dev 依赖，主依赖不带），需手动 `pip install pyinstrument`。
8. `pid_optimization.py` 设置 `text.usetex=True`，无 LaTeX 环境会直接报错——复现前改为 False。
9. `iter_hybrid_random.py` 是 `while True` 无限循环，做随机 baseline 统计时必须自加 episode 上限（参考 `code/fusion-rl-bench/scripts/random_stats.py`）。
10. Windows 控制台 GBK 编码会乱码中文 print——脚本运行加 `PYTHONIOENCODING=utf-8` 或 `python -X utf8`。
11. **reset 性能**：FreeGSNKE 每 episode 重建装置对象极慢（粗网格 ~35–45s）；解法：基线平衡缓存 + `eq.create_auxiliary_equilibrium()` 克隆（reset 0.11s）。这是 RL 训练可行的关键优化。
12. 自建环境封装时，`_solve_and_check` 等内部方法引用的 `self._profiles` 必须先赋值再求解（否则 profiles=None 静默失败为"不收敛"，且伴随 O-point 初始化警告误导排查方向）。
6. **JAX 在 Windows 为 CPU 后端**：日志提示 TPU 不可用、默认 CPU——属正常，1.5D 输运 CPU 足够。

## 跑通验证清单

- [x] `import torax`（1.4.3）/ `import gymnasium`（1.3.0）✅
- [x] TORAX 官方 example（basic_config）跑通：**5 秒物理过程仅用 0.12 秒 wall-clock**（输出 `/tmp/torax_results/state_history_*.nc`）✅
- [x] `import freegsnke`（3.0.1，源码版覆盖安装）✅
- [x] FreeGSNKE 官方 example02（MAST-U 静态正向平衡求解）nbconvert 执行成功 ✅
- [x] Gym-TORAX（`pip install -e code/vendor/gymtorax`）冒烟测试 `test_env.py` 通过：Gymnasium 环境 → TORAX 仿真步进（Ip=3MA，t→9.0/150s）✅ **全链路打通**

## 硬件/算力说明

- 当前：本机 CPU（Windows）；TORAX 基于 JAX，CPU 即可跑 1.5D 输运；
- 后续 RL 训练量大时：按需租云 GPU（AutoDL/腾讯云，4090 级约 2–3 元/小时）。
