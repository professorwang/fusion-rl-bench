"""FreeGSNKELinearPosEnv — MAST-U 等离子体位置控制 Gymnasium 环境（v0.4）

物理路径切换说明（为什么放弃静态牛顿求解做逐步物理）：
静态 NKGSsolver 在 PPO 探索产生的非预期 psi 态下，jtor 剖面的临界点搜索
（diverted_critical）会病态旋转——三次训练翻车实证（见 W05 笔记）。
本环境改用 FreeGSNKE 的**线性演化模式**（nonlinear_solve.nl_solver,
linear_only=True, no_GS=True）：围绕基线平衡的线性化响应，逐步只做矩阵运算
——无牛顿迭代、无临界点搜索，毫秒级一步。这也正是真实托卡马克 RZIP 位置
控制的工程实现方式（线性模型 + 反馈），物理故事更贴近实际。

- 任务：垂直/径向位置保持——智能体输出控制线圈电压，把等离子体电流中心
  （Zcurrent）与内中平面半径（Rin）维持在基线目标值
- 动作：控制线圈电压指令（归一化 [-1,1] → ±max_voltage V），其余主动线圈 0 V
- 观测：Zcurrent、Rin、下 X 点 (Rx, Zx)、等离子体电流 Ip、控制线圈电流（归一化）
- 奖励：-( |Zcur - Zcur*| + |Rin - Rin*| )（米）
- 失败：异常/NaN → 罚 fail_penalty 并终止

运行解释器：.venv-gs
"""
from __future__ import annotations

import os
import pickle

import gymnasium as gym
import numpy as np
from gymnasium import spaces

_REPO = os.environ.get("FREEGSNKE_REPO", os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "freegsnke"))
_MASTU = os.path.join(_REPO, "machine_configs", "MAST-U")
_BASE_CURRENTS = os.path.join(_REPO, "examples", "data", "simple_diverted_currents_PaxisIp.pk")

DEFAULT_CONTROL_COILS = ["P4", "P5", "P6", "D1", "D2", "D3", "Dp"]


def _make_plasma_descriptors():
    """返回 descriptors(eq) 函数：下 X 点位置、电流质心 Z、内中平面半径。"""
    XPT_BOX = [[0.33, -0.88], [0.95, -1.38]]  # MAST-U 下 X 点搜索框（官方 example05b）

    def plasma_descriptors(eq):
        xpt_mask = (
            (eq.xpt[:, 0] >= XPT_BOX[0][0]) & (eq.xpt[:, 0] <= XPT_BOX[1][0])
            & (eq.xpt[:, 1] <= XPT_BOX[0][1]) & (eq.xpt[:, 1] >= XPT_BOX[1][1])
        )
        xpts = eq.xpt[xpt_mask, 0:2].squeeze()
        if xpts.ndim > 1 and xpts.shape[0] > 1:
            opt = eq.opt[0, 0:2]
            idx = np.argmin(np.linalg.norm(xpts - opt, axis=1))
            Rx, Zx = xpts[idx, :]
        else:
            Rx, Zx = np.atleast_1d(xpts)[0], np.atleast_1d(xpts)[-1]
        Zcurrent = eq.Zcurrent()
        Rin = eq.innerOuterSeparatrix()[0]
        return np.array([Rx, Zx, Zcurrent, Rin], dtype=float)

    return plasma_descriptors


class FreeGSNKELinearPosEnv(gym.Env):
    """线性演化位置控制环境。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        control_coils: list[str] | None = None,
        max_voltage: float = 5.0,
        full_timestep: float = 5e-4,
        steps_per_action: int = 4,
        max_steps: int = 50,
        fail_penalty: float = 10.0,
        init_disturb_steps: int = 0,
        init_disturb_voltage: float = 50.0,
    ):
        super().__init__()
        self.control_coils = control_coils or DEFAULT_CONTROL_COILS
        self.max_voltage = max_voltage
        self.full_timestep = full_timestep
        self.steps_per_action = steps_per_action  # 每个动作推进的物理子步数
        self.max_steps = max_steps
        self.fail_penalty = fail_penalty
        self.init_disturb_steps = init_disturb_steps      # reset 后施加的随机扰动电压步数
        self.init_disturb_voltage = init_disturb_voltage  # 扰动电压幅值（V）
        # 观测/奖励归一化：描述符减去基线目标后按尺度缩放（开环垂直不稳定，误差会指数增长）
        self.obs_scale = np.array([0.1, 0.1, 0.1, 0.1])  # Rx, Zx, Zcur, Rin 各 10cm 为 1 个单位
        self.pos_lost_threshold = 0.10  # |Zcur| 或 |Rin| 偏离目标 10cm 判"等离子体丢失"

        n = len(self.control_coils)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float32)
        # 观测：Rx, Zx, Zcur, Rin + dZcur/dt, dRin/dt（归一化偏差）
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
        self._prev_d = None

        self._eq_base = None
        self._profiles_base = None
        self._solver = None            # 静态求解器（仅建基线用一次）
        self._stepping = None
        self._base_currents: dict[str, float] = {}
        self._active_labels: list[str] = []
        self._ctrl_idx: list[int] = []
        self._target = None            # [Rx, Zx, Zcur, Rin] 基线目标
        self._step_count = 0

    # ---------- 内部 ----------

    def _build_baseline(self):
        """构建装置并求基线静态平衡（每个进程只做一次）。"""
        from freegsnke import GSstaticsolver, build_machine, equilibrium_update
        from freegsnke.jtor_update import ConstrainPaxisIp

        tokamak = build_machine.tokamak(
            active_coils_path=os.path.join(_MASTU, "MAST-U_like_active_coils.pickle"),
            passive_coils_path=os.path.join(_MASTU, "MAST-U_like_passive_coils.pickle"),
            limiter_path=os.path.join(_MASTU, "MAST-U_like_limiter.pickle"),
            wall_path=os.path.join(_MASTU, "MAST-U_like_wall.pickle"),
        )
        eq = equilibrium_update.Equilibrium(
            tokamak=tokamak, Rmin=0.1, Rmax=2.0, Zmin=-2.2, Zmax=2.2, nx=65, ny=129
        )
        profiles = ConstrainPaxisIp(eq=eq, paxis=8e3, Ip=6e5, fvac=0.5, alpha_m=1.8, alpha_n=1.2)
        with open(_BASE_CURRENTS, "rb") as f:
            self._base_currents = pickle.load(f)
        for k, v in self._base_currents.items():
            eq.tokamak.set_coil_current(coil_label=k, current_value=v)
        self._solver = GSstaticsolver.NKGSsolver(eq, gs_operator_order=4)
        self._solver.solve(eq=eq, profiles=profiles, constrain=None,
                           target_relative_tolerance=1e-6, max_solving_iterations=50,
                           verbose=False)
        self._eq_base = eq
        self._profiles_base = profiles

    def _init_stepping(self, eq, profiles):
        from freegsnke import nonlinear_solve

        stepping = nonlinear_solve.nl_solver(
            eq=eq,
            profiles=profiles,
            GSStaticSolver=self._solver,
            full_timestep=self.full_timestep,
            plasma_resistivity=1e-6,
            max_mode_frequency=10**2.5,
            n_linearization_workers=1,
            plasma_descriptor_function=_make_plasma_descriptors(),
        )
        stepping.initialize_from_ICs(eq, profiles)
        return stepping

    def _obs(self):
        d_raw = np.asarray(self._stepping.plasma_descriptors_vec, dtype=float)  # [Rx,Zx,Zcur,Rin]
        d = (d_raw - self._target) / self.obs_scale  # 归一化：相对基线目标的偏差
        # 位置偏差的时间导数（垂直稳定是 PD 型问题，ż 是必需观测量——否则策略学不到微分项）
        dt = self.full_timestep * self.steps_per_action
        if self._prev_d is None:
            d_dot = np.zeros(2)
        else:
            d_dot = np.array([(d[2] - self._prev_d[2]) / dt, (d[3] - self._prev_d[3]) / dt])
        self._prev_d = d.copy()  # 存全量 4 维（先前误存 2 维子集导致下一步 IndexError）
        d_dot = np.clip(d_dot * 1e-3, -10, 10)  # 归一化到 O(1) 量级
        # 注：线圈电流观测已移除——stepping.currents_vec 布局为"全金属电流+等离子体本征模"，
        # 用主动线圈索引读取会错位到模态分量（曾读出 1.9e5 的伪值），对本任务亦无增量信息。
        return np.clip(np.concatenate([d, d_dot]), -1e6, 1e6).astype(np.float32)

    # ---------- Gymnasium 接口 ----------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._eq_base is None:
            self._build_baseline()
            self._target = np.asarray(
                _make_plasma_descriptors()(self._eq_base), dtype=float
            )  # [Rx, Zx, Zcur, Rin]

        eq = self._eq_base.create_auxiliary_equilibrium()
        for k, v in self._base_currents.items():
            eq.tokamak.set_coil_current(coil_label=k, current_value=v)

        if self._stepping is None:
            # 首次：完整构建 nl_solver（线性化矩阵计算昂贵，跨 episode 复用）
            self._stepping = self._init_stepping(eq, self._profiles_base)
        else:
            # 之后：仅重置动态状态到基线初值，复用已构建的线性化矩阵
            self._stepping.initialize_from_ICs(eq, self._profiles_base)

        # 主动线圈标签与控制索引（每个 reset 重建，防止实例差异）
        self._active_labels = list(self._base_currents.keys())
        n_active = self._stepping.evol_metal_curr.n_active_coils
        self._active_labels = self._active_labels[:n_active]
        self._ctrl_idx = [self._active_labels.index(c) for c in self.control_coils
                          if c in self._active_labels]
        # 等离子体电流在 currents_vec 中的索引（通常是最后一个主动分量之后）
        self._ip_index = getattr(self._stepping, "Iplasma_index", None)

        # 初始扰动：施加随机方向电压若干步，把等离子体推离平衡点（regulation 任务起点）
        if self.init_disturb_steps > 0:
            try:
                n_active = self._stepping.evol_metal_curr.n_active_coils
                direction = self.np_random.uniform(-1.0, 1.0, size=len(self._ctrl_idx))
                direction /= (np.linalg.norm(direction) + 1e-9)
                for _ in range(self.init_disturb_steps):
                    v = np.zeros(n_active)
                    for d, i in zip(direction, self._ctrl_idx):
                        v[i] = d * self.init_disturb_voltage
                    self._stepping.nlstepper(active_voltage_vec=v, linear_only=True,
                                             no_GS=True, verbose=False)
            except Exception as e:
                # v0.1.1（审核 R6）：不再静默吞错——记录告警并继续（退化为无扰动起点）
                import warnings
                warnings.warn(f"初始扰动施加失败，退化为无扰动起点：{e!r}")

        self._step_count = 0
        self._prev_d = None  # 每个 episode 导数观测从零开始
        try:
            return self._obs(), {"target": self._target.tolist()}
        except Exception as e:
            import warnings
            warnings.warn(f"reset 观测构建失败，返回零观测：{e!r}")
            return np.zeros(self.observation_space.shape, dtype=np.float32), {"obs_error": repr(e)}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        voltages = np.zeros(self._stepping.evol_metal_curr.n_active_coils)
        for a, i in zip(action, self._ctrl_idx):
            voltages[i] = a * self.max_voltage

        fail = None
        try:
            for _ in range(self.steps_per_action):
                self._stepping.nlstepper(
                    active_voltage_vec=voltages, linear_only=True, no_GS=True, verbose=False
                )
        except Exception:
            fail = "stepper_error"

        self._step_count += 1
        try:
            obs = self._obs()
            d = obs[:4]  # 已归一化（相对目标偏差 / 10cm）
            if not np.all(np.isfinite(d)):
                fail = fail or "nan_state"
            elif abs(d[2]) > 1.0 or abs(d[3]) > 1.0:
                fail = fail or "position_lost"  # 垂直/径向位置丢失（>10cm 偏移）
        except Exception as e:
            fail = fail or "obs_error"
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            import warnings
            warnings.warn(f"step 观测构建失败（episode 终止）：{e!r}")

        terminated = fail is not None
        truncated = self._step_count >= self.max_steps

        if terminated:
            reward = -1.0  # 温和终止罚（悬崖式重罚会让优势函数被惩罚主导，学不到稳定律）
        else:
            err_norm = abs(d[2]) + abs(d[3])  # 归一化位置偏差
            # 稠密有界奖励：保持在目标附近 ≈1，漂移指数衰减；小动作惩罚抑制 bang-bang
            reward = float(np.exp(-5.0 * err_norm) - 0.01 * np.sum(action ** 2))

        info = {"fail_reason": fail, "step": self._step_count}
        return obs, reward, terminated, truncated, info
