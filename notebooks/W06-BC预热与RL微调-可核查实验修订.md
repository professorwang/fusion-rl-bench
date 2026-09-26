# W06 笔记：BC 预热 + RL 微调 —— 从"吹过头的结论"到可核查的实验（v2.0 修订）

> 日期：2026-09-26（v2.0 按发表前审核修订）｜ 本笔记与 W05 共同构成工程实录。
> **重要**：v1.0 中"RL 超过经典 PD（里程碑 M1）"的结论**已撤回**——当时三种控制器使用了三种不同误差口径，交叉比较不成立。v2.0 以统一配对评估重新表述。

## 一、当前可证实的结果（统一口径配对评估，scripts/fair_eval.py）

协议：相同 10 个初始条件种子、相同初始扰动、同一模拟器实例、动作后采样；误差先按回合计算再跨回合取均值；失败回合单独计数、不混入误差。

| 控制器 | 完整回合 | 全过程垂直误差 MAE |
|---|---|---|
| 经典 PD（Kp=100, Kd=100，仅 P6） | 10/10 | 2.42 cm |
| BC（模仿 PD） | 10/10 | 1.74 cm |
| BC + PPO 微调（16k 步） | 10/10 | **1.27 cm** |

**结论（限定版）**：在当前任务设置与指标下，训练出的 PPO 模型误差低于我们调过的 PD——这是**积极信号，值得继续验证**。
**不能由此主张**：RL 已优于经典控制器。限制条件：单训练种子、线性化 MAST-U-like 模型（上游声明非精确副本）、25ms 仿真回合、PD 仅粗网格调参且单通道（RL 三通道权限不对等）、未做 LQR 等强基线对比、非真机验证。

逐回合原始数据：`results/fair_eval_v011.json`；对比图：`assets/fair_eval_v011.png`（替代已撤回的 bc_rl_vs_pd.png）。

## 二、审核暴露的问题与本轮修复（对应审核 R1–R6）

| 问题 | 修复 |
|---|---|
| 三控制器三种误差口径（R1） | `scripts/fair_eval.py` 统一配对评估；撤回口径混用的对比图 |
| PD 调参 `best[1]` 字段 bug（R2） | 已修（比较 best[2]）；PD 补齐初始扰动与多 episode 评估 |
| "50/50""0% 失败"可靠性误述（R3） | 文章改述；评估失败与有效轨迹误差分开报告；固定评估种子 |
| BC→PPO 权重迁移改变策略函数（R4） | 代码注释明示；log_std=-1 对应归一化标准差≈0.368（500V 尺度下约 184V），不再称"接近确定性" |
| MAST-U 表述越界（R5） | 全文改"MAST-U-like 线性化模型"；233/s 标注为模型诊断而非实测 |
| 静默吞错（R6） | env 改为 warnings 显式记录；"必须有导数"限定为当前设定下的经验 |

## 三、保留的工程经验（未受审核影响）

1. **观测含时间导数**：在当前无记忆策略拟合 PD 的设定下，显式加入 dZcur/dt 使模仿学习可收敛（限定表述；帧堆叠/RNN 为候选未做隔离对比）。
2. **控制权限扫描**：本模型、基线电流、100V/5ms 脉冲、33×65 网格下，P6 对垂直位置灵敏度约为其他主动线圈的两个数量级（仅适用该模型与测量方式）。
3. **三次翻车实录**（归因措辞已按审核收紧）：①病态位形下全牛顿求解打转——静态求解不适合毫秒级动态稳定任务，并非静态求解本身错误；②线程超时方案曾致进程异常退出，根因未经最小复现确认，已弃用；③异常观测伪值系状态向量布局索引不当，已移除该观测项。
4. **环境工程**：克隆 reset（0.5s）、线性化矩阵跨 episode 复用、52ms/步（同机同任务实测，非与静态求解的公平对比）。

## 四、复现命令（v0.1.1 修订后）

```bash
# 1. PD baseline（修复挑参 bug + 初始扰动对齐 + 10 episode/组）
.venv-gs/Scripts/python -X utf8 -u code/fusion-rl-bench/scripts/pd_baseline.py
# 2. BC 预热（固定种子、自动建目录）
.venv-gs/Scripts/python -X utf8 -u code/fusion-rl-bench/scripts/bc_warmup.py
# 3. BC 权重初始化 PPO 微调（每次运行独立目录，CSV 追加不覆盖）
BC_INIT=code/fusion-rl-bench/runs/bc_policy.pt .venv-gs/Scripts/python -X utf8 -u code/fusion-rl-bench/scripts/train_ppo.py
# 4. 统一口径配对评估（逐回合原始数据输出）
.venv-gs/Scripts/python -X utf8 -u code/fusion-rl-bench/scripts/fair_eval.py \
    --episodes 10 --seed0 3000 --bc code/fusion-rl-bench/runs/bc_policy.pt \
    --ppo <run_dir>/ppo_final.zip --out code/fusion-rl-bench/results/fair_eval.json
```

## 五、下一步（承接审核阶段 B/C）

1. PD 重新调优 + LQR/LQG 经典基线；单 P6 与三通道分开对比；
2. ≥5 独立训练种子、100–200 冻结测试初始条件、置信区间；
3. 鲁棒性：观测噪声、执行延迟、参数误差、持续扰动；非线性演化模型复核；
4. 真机路径（装置合作前提下）：模型校验→仿真闭环→影子运行→装置方批准的受控实验；
5. 竞品差异分析：FPDT（FreeGSNKE+虚拟 PCS）、RL4F（DIII-D 离线 RL）、星环 TokaPCS——我们的定位是统一 RL 基准 + 可复现评估 + 开放数据。
