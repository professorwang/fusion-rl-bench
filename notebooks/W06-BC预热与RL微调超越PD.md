# W06 笔记：模仿→超越 —— BC 预热 + RL 微调超过经典 PD（里程碑）

> 日期：2026-09-26 ｜ 对应 fusion-rl-bench 里程碑 M1：**RL 控制器在真实物理环境上超过经典控制器**
> 本笔记与 `W05-环境加固与PPO训练实录.md` 共同构成白皮书第 5 章"实测数据"主体。

## 一、结果总览（任务：MAST-U 垂直位置保持，带随机初始扰动，233/s 不稳定模）

| 控制器 | 存活 | 平均/最终位置偏差 | 备注 |
|---|---|---|---|
| 纯 PPO（无预热，16k 步） | ❌ 全败 | — | 稀疏存活信号下学不会（W05 已记录） |
| 经典 PD（Kp=100, Kd=100） | ✅ 50/50 | 2.43cm | 网格扫描 11/16 组增益可稳定 |
| **BC（模仿 PD）** | ✅ 50/50 | **1.54cm** | 7500 专家样本，800 epoch |
| **BC + PPO 微调（16k 步）** | ✅ **0% 失败** | **最低 1.05cm，多数评估点 < PD** | **RL 超过经典 PD ✅** |

对比曲线：`runs/bc_rl_vs_pd.png`。

## 二、决定性技术点（按影响排序）

1. **观测必须含时间导数**：PD 是 (z, ż) 的函数；只给 z 的观测使 obs→action 映射不可学（MSE 地板 ~0.12 不死不活两轮）。环境加入 dZcur/dt、dRin/dt 后 BC 一轮即达标。**这是本轮最重要的教训，也是 RL 控制的常识但容易被忽视的点。**
2. **模仿→超越范式**：纯 PPO 在 ms 级不稳定系统上探索成本过高；先用 7500 条 PD 数据 BC 到 1.54cm，再 PPO 微调（log_std=-1.0 近确定性起步）→ 稳定超过专家。
3. **控制权限匹配物理**：P6 灵敏度扫描（200 倍于其他线圈）+ ±500V 权限 + 0.5ms 控制周期（233/s 增长率决定）。
4. **稠密有界奖励**：exp(-5·err) − 0.01a² + 温和终止罚（-1），避免优势函数被惩罚主导。
5. **初始扰动起点**：reset 时施加随机方向电压 3 步，任务从无扰动保持变为 regulation——更接近真实工况，且 PD 在此任务上 2.43cm 是公平对照。

## 三、排坑新增

13. `currents_vec` 布局为"全金属电流+等离子体本征模"，用主动线圈索引读取会错位到模态分量（读出 1.9e5 伪值且含复数）——观测别碰它，位置描述符已够用。
14. `_prev_d` 只存 2 维子集但按 4 维索引——IndexError 被静默 catch 成 obs_error，导致 episode 全灭。**教训：except 吞错时把 fail_reason 打全，调试成本差 10 倍。**

## 四、复现命令

```bash
# 1. BC 预热（含 PD 专家数据采集）
.venv-gs/Scripts/python -X utf8 -u code/fusion-rl-bench/scripts/bc_warmup.py
# 2. BC 权重初始化 PPO 微调（8k 步 ×2 轮）
BC_INIT=code/fusion-rl-bench/runs/bc_policy.pt .venv-gs/Scripts/python -X utf8 -u code/fusion-rl-bench/scripts/train_ppo.py
RESUME=code/fusion-rl-bench/runs/ppo_final .venv-gs/Scripts/python -X utf8 -u code/fusion-rl-bench/scripts/train_ppo.py
# 3. PD baseline 对照
.venv-gs/Scripts/python -X utf8 -u code/fusion-rl-bench/scripts/pd_baseline.py
```

## 五、下一步（W07–W08 候选）

1. **跨工况泛化**：变初始扰动强度/方向、变 Ip 设定值，验证策略鲁棒性（白皮书"泛化"一节）；
2. **多任务扩展**：加入 Rx/Zx（X 点）控制通道，向形状控制过渡；
3. **开源发布**：整理 README（中英）、演示 GIF（渲染 LCFS 轨迹）、arXiv 短文框架；
4. **写 build log 第 2 篇**：《从三次翻车到超过经典控制器：我们在真实托卡马克物理上训练 RL》（素材：W05+W06 全部实录）。
