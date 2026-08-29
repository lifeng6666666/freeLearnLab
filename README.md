# 博学实验室

> 在 AI 时代，博学万物。一个不断拓展的跨领域课程库。

## 项目介绍

博学实验室是一个不断拓展的跨领域课程库。项目利用 AI 协助制作广阔、轻量、注重实践的课程，把分散在不同学科中的知识整理成可以阅读、运行、练习和迁移的学习材料。

这里将持续探索值得学习的主题：从机器学习、编程与数据，到人文、社会、商业、艺术和日常生活中的各种问题，都可以成为课程。

诸君可在此自由、广博的学习。

项目的课程强调：

- **跨领域**：不以单一专业或职业为边界，鼓励建立不同知识之间的连接；
- **轻量化**：用清晰的解释、必要的数学和短小的实践降低入门门槛；
- **重实践**：配套 Notebook、案例、练习或小项目，让知识落到可运行的结果上；
- **AI 辅助制作**：利用 AI 做资料梳理、讲解设计、代码示例和迭代校对，同时保留人工判断与验证；
- **持续演进**：课程会随着实践反馈不断修订，目录也会持续扩展。

## 课程

### 机器学习全面教程

目录：[macheine-learning](macheine-learning/)

这是一套从基础概念到实际应用的机器学习课程。课程使用 PDF 讲义配合 Jupyter Notebook，兼顾理论理解、算法原理、从零实现和数据实验。每章尽量以轻量、清晰、可运行的方式讲解一个核心主题，并通过案例观察算法在真实数据上的表现。

课程特点：**简明**的讲解、**直观配图**的概念说明、每个算法**一步步从零实现**。

#### 课程内容

| 文件            | 主题         | 从零实现                          | 应用数据           |
| ------------- | ---------- | ----------------------------- | -------------- |
| `utils.ipynb` | 工具函数       | —                             | —              |
| `ch01.ipynb`  | 机器学习概览     | —                             | 鸢尾花完整流程        |
| `ch02.ipynb`  | 线性回归       | `MyLinearRegression`          | 加州房价           |
| `ch03.ipynb`  | 决策树        | `MyDecisionTreeClassifier`    | 贷款审批           |
| `ch04.ipynb`  | 聚类         | `MyKMeans`                    | 客户分群、图像压缩、异常检测 |
| `ch05.ipynb`  | 集成学习       | `MyRandomForest`、`MyAdaBoost` | 乳腺癌分类          |
| `ch06.ipynb`  | 支持向量机（SVM） | `MySVM`（线性、hinge loss）        | 手写数字           |
| `ch07.ipynb`  | 神经网络       | `MyMLP`（手写反向传播）               | MNIST、手写数字     |
| `ch08.ipynb`  | 降维         | `MyPCA`                       | t-SNE 可视化      |

课程资料包括：

- [机器学习全面教程（PDF）](macheine-learning/机器学习全面教程.pdf)：理论知识、核心概念、公式、算法步骤和伪代码；
- `ch01.ipynb` 至 `ch08.ipynb`：对应章节的代码实现、实验和应用案例；
### 强化学习全面教程

目录：[reinforcement-learning](reinforcement-learning/)

这是一套以讲义 PDF 为主线、Jupyter Notebook 为配套实践的强化学习课程。PDF 负责解释核心概念、算法原理、数学背景和学习路径；Notebook 则用于运行环境交互、验证实现、观察实验结果，并帮助学习者把理论转化为可执行代码。

课程特点：核心公式**一步步推导**、不跳步；各算法**前后连贯**，每个算法都由前一方法的局限自然引出，形成完整演进脉络。

#### 课程内容

| 文件                           | 主题                    | 关键内容                            |
| ---------------------------- | --------------------- | ------------------------------- |
| `utils.ipynb`                | 通用工具库                 | 环境可视化、训练辅助函数                    |
| `ch01_env_loop.ipynb`        | 强化学习入门与环境交互循环         | Agent、Environment、Reward、Policy |
| `ch02_mdp_frozen_lake.ipynb` | MDP 与 FrozenLake      | 状态、动作、转移、奖励                     |
| `ch03_dp.ipynb`              | 动态规划                  | 值迭代、策略迭代                        |
| `ch04_mc_control.ipynb`      | 蒙特卡洛控制                | MC prediction / control         |
| `ch05_td_sarsa_q.ipynb`      | TD、SARSA 与 Q-Learning | 时序差分与离线学习                       |
| `ch06_dqn.ipynb`             | 深度 Q 网络               | DQN、经验回放、目标网络                   |
| `ch07_reinforce.ipynb`       | REINFORCE             | 策略梯度                            |
| `ch08_a2c.ipynb`             | A2C                   | Actor-Critic                    |
| `ch09_ppo.ipynb`             | PPO                   | 近端策略优化                          |
| `ch10_sac.ipynb`             | SAC                   | 最大熵强化学习                         |
| `ch11_rainbow.ipynb`         | Rainbow               | DQN 变体与提升方法                     |
| `ch12_rlhf_concept.ipynb`    | RLHF 概念               | PPO + RM + KL 惩罚                |

课程资料包括：

- 主线讲义：PDF 讲义为课程主体，负责理论梳理、公式与算法理解；
- Notebook 代码：`ch01_env_loop.ipynb` 至 `ch12_rlhf_concept.ipynb`，用于配套实验与代码实践；
- [通用工具模块](reinforcement-learning/utils.ipynb)：训练与可视化辅助函数。

后续课程将按相同方式持续加入，覆盖更多学科与实践主题。
