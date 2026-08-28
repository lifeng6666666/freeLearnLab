# 强化学习全面教程 · Jupyter Notebook 版

本目录是《强化学习全面教程》（v1.0 · 2026.08，共 14 章）的配套 Notebook 包，包含 12 章可交互代码 + 1 个工具模块。每个 Notebook 都可以独立运行，配套 PDF 教程同步阅读。

## 📁 目录结构

| 文件 | 对应 PDF 章节 | 算法 | 环境 |
|------|------|------|------|
| `utils.ipynb` | — | 通用工具：经验回放、神经网络基类、绘图 | — |
| `ch01_env_loop.ipynb` | 第一章 强化学习入门与核心概念 | 环境交互循环 | CartPole-v1 |
| `ch02_mdp_frozen_lake.ipynb` | 第二章 数学基础：MDP 与贝尔曼方程 | MDP 分析与策略评估 | FrozenLake-v1 |
| `ch03_dp.ipynb` | 第三章 基于值的方法：动态规划 | 策略迭代 + 值迭代 | FrozenLake-v1 |
| `ch04_mc_control.ipynb` | 第四章 蒙特卡洛方法 | Monte Carlo 控制 | Blackjack-v1 |
| `ch05_td_sarsa_q.ipynb` | 第五章 时序差分学习：SARSA 与 Q-Learning | SARSA + Q-Learning | CliffWalking-v0 |
| `ch06_dqn.ipynb` | 第六章 函数逼近与 DQN | DQN 完整实现 | CartPole-v1 |
| `ch07_reinforce.ipynb` | 第七章 策略梯度方法：REINFORCE | REINFORCE 策略梯度 | CartPole-v1 |
| `ch08_a2c.ipynb` | 第八章 Actor-Critic 与 A2C/A3C | A2C 同步 Actor-Critic + GAE | CartPole-v1 |
| `ch09_ppo.ipynb` | 第九章 PPO：现代 RL 工业标准 | PPO 完整实现 | CartPole-v1 |
| `ch10_sac.ipynb` | 第十章 SAC 与连续控制 | SAC 连续控制 | Pendulum-v1 |
| `ch11_rainbow.ipynb` | 第十一章 Rainbow DQN 与改进技巧 | Double DQN + Dueling + PER | CartPole-v1 |
| `ch12_rlhf_concept.ipynb` | 第十二章 前沿主题：RLHF、DPO、世界模型、离线 RL、MARL | RLHF 简化概念演示 | ToyEnv |
| — | 第十三章 算法对比与工程实践 | 算法选择决策树、对比与调参（无 Notebook，见 PDF） | — |
| — | 第十四章 调试、调参与常见问题排查 | 训练问题排查 + 调试 checklist（无 Notebook，见 PDF） | — |

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch gymnasium numpy matplotlib tqdm jupyter
```

### 2. 启动 Jupyter Lab

```bash
cd reinforcement-learning
jupyter lab
```

### 3. 按章节顺序运行

建议从 `ch01_env_loop.ipynb` 开始，每章运行后再读下一章。每个 Notebook 顶部都有依赖导入单元，按顺序执行即可。

## 📚 学习路径建议

### 入门路径（5 天）
- Day 1: Ch01 + Ch02（建立 RL 直觉与 MDP 概念）
- Day 2: Ch03 + Ch04（动态规划 + 蒙特卡洛）
- Day 3: Ch05 + Ch06（TD + DQN）
- Day 4: Ch07 + Ch08（策略梯度 + A2C）
- Day 5: Ch09（PPO 工业级算法）

### 进阶路径（额外 3 天）
- Day 6: Ch10（SAC 连续控制）
- Day 7: Ch11（Rainbow DQN 改进）
- Day 8: Ch12（RLHF、DPO、世界模型、离线 RL、MARL 等前沿主题）

### 收尾（阅读 PDF）
- 第十三章：算法对比与工程实践（算法选择决策树、超参数经验、训练曲线分析）
- 第十四章：调试、调参与常见问题排查（训练不收敛、Q 值爆炸、策略熵坍缩等）

## ⚙️ 环境要求

- Python 3.9+
- PyTorch 2.0+（CPU 即可，GPU 可选）
- Gymnasium 0.29+
- matplotlib 3.5+

### GPU 加速说明

CartPole、FrozenLake 等小环境用 CPU 即可（每个 Notebook 1-10 分钟）。若处理 Atari 等大状态空间，强烈建议使用 GPU。Notebook 顶部会自动检测并使用 GPU。

## 🔍 常见问题

### Q1: 中文字体显示为方块

Notebook 顶部已配置 Noto Sans SC 字体（Linux 路径）。若你在 macOS 或 Windows 上运行，请修改字体路径：
- macOS: `Heiti SC` 或 `PingFang SC`
- Windows: `Microsoft YaHei`

### Q2: 训练不收敛

参考 PDF 教程第十四章（调试、调参与常见问题排查）。常见原因：随机种子未固定、学习率过大、batch_size 太小、Q 值爆炸、策略熵过早坍缩。第 14.8 节提供完整调试 checklist。

### Q3: GPU 显存不足

减小 batch_size（64 → 32）或减小网络隐藏层大小（128 → 64）。

## 📖 与 PDF 教程的关系

本 Notebook 包与《强化学习全面教程》PDF 配套使用：

- **PDF**：系统讲解理论、数学推导、算法对比、工程实践
- **Notebook**：可运行代码、可视化训练曲线、交互式调参

建议学习方式：
1. 先读 PDF 对应章节，理解原理
2. 打开 Notebook，按顺序运行代码单元
3. 修改超参数，观察行为变化
4. 回到 PDF 复习理论，加深理解

## 🎯 下一步

完成本教程后，建议：
- 阅读 CleanRL / Stable-Baselines3 源码
- 在 MuJoCo 等更复杂环境上实践
- 关注 RLHF / Decision Transformer 等前沿方向
- 尝试复现一篇 RL 论文（如 PPO、SAC 原论文）

祝你学习愉快！🚀
