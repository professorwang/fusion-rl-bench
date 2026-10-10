# 外部复现指南（REPRODUCE）— v0.2.0 冻结资产

本文档面向**第三方**：从全新目录开始，独立完成"安装 → 下载冻结模型 → 重出评估结果 →
与冻结数值对照"的全流程。

## 复现范围（诚实声明）

- **包含**：冻结模型评估链——`scripts/fair_eval.py` + 冻结权重（bc_policy.pt、
  ppo_final.zip）+ 冻结参考数据（fair_eval_v013_s3000/s3010.json 等 Release 资产）。
- **不包含**：训练链重跑（B2 增益面扫描、B3 消融训练、B4 多种子训练）。这些实验
  的**产物**（JSON/PNG）随 Release 分发并做哈希核对，但训练本身不在本复现范围内。

## 验收判据（先读）

**判据是排名、量级与结论方向一致，不是逐位一致。** 跨平台数值漂移已证实：同一冻结
权重 + 同种子，Windows 与 macOS 的评估结果相对偏差最大约 ×1.8（开环不稳定通道放大
数值细节，见 `docs/SETUP.md` 排坑 16 与 `notebooks/B3b` 第四节）。同机配对比较不受
影响。

> ⚠️ **上述漂移容差不适用于模型驱动控制器**（LQR/LQI/LQG 等基于辨识模型的反馈）：
> B6 实测 Windows 辨识的 `sysid_model.npz` 在 macOS 上使全状态 LQI 直接失稳（0/10），
> 尽管主导极点一致（A 矩阵 Frobenius 差 77.5%，见 `notebooks/B6-LQG部分观测基线.md`）。
> 模型驱动实验必须在运行平台上重新辨识模型（`scripts/sysid_linear.py`），
> 不得跨平台移植冻结模型。

`scripts/verify_release.py` 的判定逻辑：

| 判定 | 条件 | 退出码 |
|---|---|---|
| 通过 PASS | 资产哈希全部匹配；完整率一致；mae_z_cm 排名一致（PD < BC < PPO）；误差比值在 [1/tol, tol] 内（默认 tol=2.0，覆盖已观测漂移上限 ×1.8） | 0 |
| 漂移 DRIFT | 排名与完整率一致，个别数值超容差带——结论方向不变，属已知跨平台现象，需人工复核 | 0 |
| 失败 FAIL | 哈希不匹配 / 控制器缺失 / 完整率不一致 / 排名改变 | 1 |

## 参考值（冻结，Windows 11 + Python 3.12.15 生成）

`fair_eval_v013`（10 回合配对评估，动作后采样，种子 3000–3009 / 3010–3019）：

| 控制器 | 完整回合 | MAE_Z cm（s3000） | MAE_Z cm（s3010） |
|---|---|---|---|
| PD(Kp=1, Kd=0.3, P6 单通道) | 10/10 | 0.0924 | 0.0823 |
| BC（模仿高增益 PD） | 10/10 | 0.1879 | 0.1838 |
| PPO（BC 热启动微调） | 10/10 | 1.2679 | 1.2801 |

预期结论方向：**PD(1,0.3) < BC < PPO**（误差越小越好），三者全部回合完整、无饱和
（PPO 饱和率约 35% 除外）。更宽的控制器家族排名（含 PD(3,0.3)=0.014cm 低增益盆地、
LQR/LQI）见 README；本脚本只核对评估链覆盖的 PD/BC/PPO 三者。

## 复现步骤

### 0. 前置

- git、Python **3.10–3.12**（推荐 3.12，参考环境为 3.12.15；**3.13 不可用**——
  numba 0.60 尚不支持）；约 3 GB 磁盘（主要是 torch）。
- macOS 用户注意：**不要把 venv 建在外接 exFAT 盘上**（小文件随机写入极慢，
  `docs/SETUP.md` 排坑 13），用本地盘。

### 1. 克隆仓库

```bash
git clone https://github.com/professorwang/fusion-rl-bench.git
cd fusion-rl-bench
```

### 2. 建 venv 并安装依赖

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

平台差异（详见 `requirements.txt` 头部注释）：

- **Intel Mac（x86_64）**：`llvmlite==0.43.0` / `numba==0.60.0` 必须钉死——
  llvmlite ≥0.44 的 macOS 发行仅有 arm64 wheel（排坑 14）；`torch==2.2.2` 是
  Intel Mac 最后一个有 x86_64 wheel 的版本。
- **Windows / Apple Silicon**：钉版同样适用；如需更新的 llvmlite/numba/torch
  可自行放宽（数值漂移风险自担）。

### 3. 安装 FreeGSNKE 源码（PyPI 版不可用）

PyPI 的 `freegsnke 3.0.1` 缺 `NKGSsolver(gs_operator_order=...)` 参数（排坑 6），
必须从 GitHub 源码安装，钉到与本项目验证时相同的 commit：

```bash
git clone https://github.com/FusionComputingLab/freegsnke.git
cd freegsnke && git checkout df5bae82cdbfd733e019b320d15b2d43c7947406 && cd ..
pip install --no-deps -e ./freegsnke    # --no-deps：运行依赖已由 requirements.txt 提供
export FREEGSNKE_REPO="$PWD/freegsnke"  # envs 据此读取 machine_configs/MAST-U 装置数据
```

### 4. 冒烟测试（可选但推荐）

```bash
python envs/smoke_test_linear.py   # 建 MAST-U-like 装置、解基线平衡、步进动作
```

首次 `env.reset` 需重建基线平衡 + numba JIT，约 1–3 分钟（一次性）；之后
reset ~0.5s、step 16–50ms。

### 5. 一键验证

```bash
python scripts/verify_release.py
```

脚本自动完成：下载 Release v0.2.0 全部冻结资产（默认存 `release_assets/v0.2.0/`，
未纳入 git 跟踪，幂等可重跑）→ SHA256 逐文件对照 `SHA256SUMS.txt` → 用冻结
bc_policy.pt / ppo_final.zip 跑 `fair_eval`（种子 3000–3009，PD Kp=1 Kd=0.3）→
与冻结 `fair_eval_v013_s3000.json` 对照，打印"通过/漂移/失败"表并输出
`verify_report.json`。

> 发布现状：Release v0.2.0 当前含 19/24 个 SHA256SUMS.txt 所列文件；B3 第二批
> 消融产物（ablation_obs/horizon/transfer 的 json+png）未上传发布，脚本自动
> 回退到仓库内 `results/`、`assets/` 同名文件核对哈希（报告中标
> "仓库内匹配(发布缺失)"），哈希覆盖率仍为 24/24。

常用选项：`--skip-download`（资产已就位）、`--skip-eval`（只核哈希）、
`--second-batch`（追加种子 3010–3019 对照）、`--tol`（调容差）、
`--assets-dir` / `--out`。

### 6. 手动逐步（不用脚本时的等价命令）

```bash
mkdir -p release_assets/v0.2.0 && cd release_assets/v0.2.0
for f in bc_policy.pt ppo_final.zip fair_eval_v013_s3000.json SHA256SUMS.txt; do
  curl -LO "https://github.com/professorwang/fusion-rl-bench/releases/download/v0.2.0/$f"
done
shasum -a 256 -c SHA256SUMS.txt        # Windows: certutil 或 Git Bash 的 sha256sum
cd ../..
python scripts/fair_eval.py --episodes 10 --seed0 3000 --pd-kp 1 --pd-kd 0.3 \
    --bc release_assets/v0.2.0/bc_policy.pt \
    --ppo release_assets/v0.2.0/ppo_final.zip \
    --out release_assets/v0.2.0/fair_eval_verify_s3000.json
# 对照 release_assets/v0.2.0/fair_eval_v013_s3000.json 的 summary 段
```

## 常见问题

- **评估很慢？** 首次 reset 的一次性开销（基线平衡构建 + numba JIT）约 1–3 分钟，
  属正常；整批 30 回合（3 控制器 × 10 种子）约 3–5 分钟。
- **数值和冻结表不完全一样？** 跨平台/跨 BLAS 实现漂移已证实（上限约 ×1.8）；
  只要排名 PD < BC < PPO、量级一致、全部回合完整，即判一致。
- **中文乱码（Windows）**：`set PYTHONIOENCODING=utf-8` 或 `python -X utf8`（排坑 10）。
- **matplotlib 中文缺字形（macOS）**：设 `plt.rcParams["font.sans-serif"]=["PingFang SC"]`（排坑 17）。
- 更多环境排坑见 `docs/SETUP.md`（17 条实录）。

## 复现不成功时

请在 GitHub issue 附上 `release_assets/v0.2.0/verify_report.json`（含环境信息与
逐文件哈希结果）与终端输出。
