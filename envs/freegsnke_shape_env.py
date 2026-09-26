"""FreeGSNKEShapeEnv — MAST-U 线圈级等离子体形状控制 Gymnasium 环境（v0.2）

v0.2 加固（W05）：
- 失败分类：solver_nonconvergence / xpoint_lost / limiter_touch / nan_state
- 碰撞检测：LCFS 须严格在 limiter 多边形内；X 点须在壁内存在
- 任务改为"形状保持调节"：reset 时对控制线圈加随机扰动（±reset_perturb_frac），
  智能体须把 LCFS 拉回基线目标形状
- 训练速度档位：nx=33/ny=65 + rtol=1e-5（粗网格快档，供训练；评估用细网格）

- 物理后端：FreeGSNKE 静态正向求解器（NKGSsolver，Newton-Krylov）
- 装置：MAST-U 球形托卡马克（machine_configs/MAST-U）
- 动作：控制线圈电流增量 ΔI（归一化 [-1,1] → 当前值 ±max_delta_frac/步）
- 观测：LCFS 相对磁轴 (dR,dZ) ×ntheta + psi_axis, psi_bndry + 线圈电流归一化
- 奖励：-mean||LCFS − LCFS_target||（米）；失败终止罚 -10

运行解释器：.venv-gs（依赖冲突见 docs/SETUP.md 排坑 1）
"""
from __future__ import annotations

import os
import pickle

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from shapely.geometry import Point, Polygon

_REPO = os.environ.get("FREEGSNKE_REPO", os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "freegsnke"))
_MASTU = os.path.join(_REPO, "machine_configs", "MAST-U")
_BASE_CURRENTS = os.path.join(_REPO, "examples", "data", "simple_diverted_currents_PaxisIp.pk")

DEFAULT_CONTROL_COILS = ["P4", "P5", "P6", "D1", "D2", "D3", "Dp"]


class FreeGSNKEShapeEnv(gym.Env):
    """线圈级自由边界形状控制环境（形状保持任务）。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        control_coils: list[str] | None = None,
        max_delta_frac: float = 0.02,
        reset_perturb_frac: float = 0.02,
        ntheta: int = 16,
        max_steps: int = 20,
        solver_rtol: float = 1e-6,
        coarse_grid: bool = False,
        fail_penalty: float = 10.0,
    ):
        super().__init__()
        self.control_coils = control_coils or DEFAULT_CONTROL_COILS
        self.max_delta_frac = max_delta_frac
        self.reset_perturb_frac = reset_perturb_frac
        self.ntheta = ntheta
        self.max_steps = max_steps
        self.solver_rtol = solver_rtol
        self.coarse_grid = coarse_grid
        self.fail_penalty = fail_penalty
        self.solve_timeout_s = 3.0    # （已弃用：线程看门狗因 BLAS 线程安全问题移除，勿恢复）
        self.min_current_frac = 0.01  # 线圈电流下限（相对基值），防止 jtor 除零退化态
        self.corridor_frac = 0.20     # 动作走廊：电流钳制在基值 ±20% 内，根治病态求解

        n_coils = len(self.control_coils)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_coils,), dtype=np.float32)
        obs_dim = 2 * ntheta + 2 + n_coils
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self._gs = None
        self._eq = None
        self._profiles = None
        self._eq_base = None        # 缓存的基线平衡（克隆源）
        self._profiles_base = None
        self._base_currents: dict[str, float] = {}
        self._currents: dict[str, float] = {}
        self._target_lcfs: np.ndarray | None = None
        self._limiter_poly: Polygon | None = None
        self._wall_poly: Polygon | None = None
        self._step_count = 0

    # ---------- 内部 ----------

    def _build_backend(self):
        """首次：完整构建装置+基线平衡；之后：从基线克隆辅助平衡（秒级）。"""
        from freegsnke import GSstaticsolver, build_machine, equilibrium_update
        from freegsnke.jtor_update import ConstrainPaxisIp

        if self._eq_base is not None:
            # 快路径：克隆基线平衡并重置线圈电流（免重建装置，reset 从 ~35s → ~1s）
            eq = self._eq_base.create_auxiliary_equilibrium()
            for k, v in self._base_currents.items():
                eq.tokamak.set_coil_current(coil_label=k, current_value=v)
            return eq, self._profiles_base

        tokamak = build_machine.tokamak(
            active_coils_path=os.path.join(_MASTU, "MAST-U_like_active_coils.pickle"),
            passive_coils_path=os.path.join(_MASTU, "MAST-U_like_passive_coils.pickle"),
            limiter_path=os.path.join(_MASTU, "MAST-U_like_limiter.pickle"),
            wall_path=os.path.join(_MASTU, "MAST-U_like_wall.pickle"),
        )
        if self._limiter_poly is None:
            self._limiter_poly = Polygon(np.array([tokamak.limiter.R, tokamak.limiter.Z]).T)
            self._wall_poly = Polygon(np.array([tokamak.wall.R, tokamak.wall.Z]).T)

        nx, ny = (33, 65) if self.coarse_grid else (65, 129)
        eq = equilibrium_update.Equilibrium(
            tokamak=tokamak, Rmin=0.1, Rmax=2.0, Zmin=-2.2, Zmax=2.2, nx=nx, ny=ny
        )
        profiles = ConstrainPaxisIp(eq=eq, paxis=8e3, Ip=6e5, fvac=0.5, alpha_m=1.8, alpha_n=1.2)
        with open(_BASE_CURRENTS, "rb") as f:
            self._base_currents = pickle.load(f)
        for k, v in self._base_currents.items():
            eq.tokamak.set_coil_current(coil_label=k, current_value=v)
        if self._gs is None:
            self._gs = GSstaticsolver.NKGSsolver(eq, gs_operator_order=4)
        # 求基线平衡并缓存（目标形状 + 克隆源）
        self._profiles = profiles  # _solve_and_check 依赖 self._profiles
        fail = self._solve_and_check(eq)
        if fail:
            raise RuntimeError(f"基线平衡求解失败：{fail}")
        if self._target_lcfs is None:
            self._target_lcfs = self._lcfs(eq).copy()
        self._eq_base = eq
        self._profiles_base = profiles
        return self._eq_base.create_auxiliary_equilibrium(), profiles

    def _check_state(self, eq) -> str | None:
        """返回 None=正常，否则失败原因字符串。"""
        try:
            psi_axis, psi_bndry = float(eq.psi_axis), float(eq.psi_bndry)
            if not (np.isfinite(psi_axis) and np.isfinite(psi_bndry)):
                return "nan_state"
            sep = np.asarray(eq.separatrix(ntheta=self.ntheta))[:, :2]
            if not np.all(np.isfinite(sep)):
                return "nan_state"
            # X 点须存在且至少一个在壁内
            xpts = getattr(eq, "xpt", None)
            if xpts is None or len(xpts) == 0:
                return "xpoint_lost"
            if not any(self._wall_poly.contains(Point(float(x[0]), float(x[1]))) for x in xpts):
                return "xpoint_lost"
            # LCFS 全部点须在 limiter 内
            for r, z in sep:
                if not self._limiter_poly.contains(Point(float(r), float(z))):
                    return "limiter_touch"
            return None
        except Exception:
            return "xpoint_lost"

    def _solve_and_check(self, eq) -> str | None:
        # 注意：不要在线程中跑求解器——scipy/BLAS 原生代码非线程安全，曾致进程崩溃。
        # 病态态防护三件套：迭代上限 25（快速失败）+ 线圈电流下限（防除零退化）+ 状态检查。
        try:
            self._gs.solve(eq=eq, profiles=self._profiles, constrain=None,
                           target_relative_tolerance=self.solver_rtol,
                           max_solving_iterations=25,
                           verbose=False)
        except Exception:
            return "solver_nonconvergence"
        return self._check_state(eq)

    def _lcfs(self, eq) -> np.ndarray:
        sep = np.asarray(eq.separatrix(ntheta=self.ntheta))[:, :2]
        axis = np.asarray(eq.magneticAxis()[:2])
        return sep - axis

    def _obs(self, eq) -> np.ndarray:
        lcfs = self._lcfs(eq).ravel()
        psi = np.array([eq.psi_axis, eq.psi_bndry])
        cur = np.array([self._currents[c] / (abs(self._base_currents[c]) + 1e-9)
                        for c in self.control_coils])
        return np.concatenate([lcfs, psi, cur]).astype(np.float32)

    # ---------- Gymnasium 接口 ----------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._eq, self._profiles = self._build_backend()  # 首次构建并缓存基线；之后克隆

        # 随机扰动控制线圈（形状保持任务起点）
        self._currents = dict(self._base_currents)
        for c in self.control_coils:
            pert = 1.0 + self.np_random.uniform(-self.reset_perturb_frac, self.reset_perturb_frac)
            self._currents[c] = self._base_currents[c] * pert
            self._eq.tokamak.set_coil_current(coil_label=c, current_value=self._currents[c])

        fail = self._solve_and_check(self._eq)
        if fail:  # 扰动后就不收敛：重新采样（极端扰动直接判负会污染学习）
            return self.reset(seed=seed)

        self._step_count = 0
        return self._obs(self._eq), {"initial_lcfs_err": float(
            np.linalg.norm(self._lcfs(self._eq) - self._target_lcfs, axis=1).mean())}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        for c, da in zip(self.control_coils, action):
            new_i = self._currents[c] * (1.0 + self.max_delta_frac * da)
            # 动作走廊：电流限制在基值 [corr_lo, corr_hi] 区间内——
            # 根治 PPO 探索产生的退化态（曾致 jtor 临界点搜索死循环）
            base = self._base_currents[c]
            lo, hi = sorted([base * (1.0 - self.corridor_frac), base * (1.0 + self.corridor_frac)])
            floor = abs(base) * self.min_current_frac
            new_i = float(np.clip(new_i, min(lo, -floor) if lo < 0 else lo, hi))
            if abs(new_i) < floor:
                new_i = np.sign(base) * floor
            self._currents[c] = new_i
            self._eq.tokamak.set_coil_current(coil_label=c, current_value=new_i)

        fail = self._solve_and_check(self._eq)
        self._step_count += 1

        terminated = fail is not None
        truncated = self._step_count >= self.max_steps

        if terminated:
            reward = -self.fail_penalty
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            err = float(np.linalg.norm(self._lcfs(self._eq) - self._target_lcfs, axis=1).mean())
            reward = -err
            obs = self._obs(self._eq)

        info = {"converged": not terminated, "fail_reason": fail, "step": self._step_count}
        return obs, reward, terminated, truncated, info
