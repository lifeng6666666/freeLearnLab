"""CartPole 训练效果可视化工具。

统一接口：传入 policy_fn(state) -> int，生成动画并内嵌显示。
所有 CartPole notebook 训练完成后调用 visualize_cartpole(policy_fn) 即可。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import gymnasium as gym


class CartPoleVisualizer:
    """CartPole 训练效果可视化器。"""

    def __init__(self, env_id='CartPole-v1', seed=999, max_steps=500):
        self.env_id = env_id
        self.seed = seed
        self.max_steps = max_steps

    def rollout(self, policy_fn):
        """用 policy_fn 跑一回合，记录轨迹。"""
        env = gym.make(self.env_id)
        s, _ = env.reset(seed=self.seed)
        states, actions, rewards = [], [], []
        total_r = 0
        for t in range(self.max_steps):
            a = int(policy_fn(s))
            states.append(s.copy())
            actions.append(a)
            s_next, r, terminated, truncated, _ = env.step(a)
            rewards.append(r)
            total_r += r
            s = s_next
            if terminated or truncated:
                # 记录终止状态（杆子倒下的画面）
                states.append(s_next.copy())
                actions.append(a)
                rewards.append(0.0)
                break
        env.close()
        return np.array(states), np.array(actions), np.array(rewards), total_r

    def animate(self, policy_fn, title='CartPole 策略可视化'):
        """生成动画并返回 HTML 对象（Jupyter 内嵌播放）。"""
        states, actions, rewards, total_r = self.rollout(policy_fn)
        n_frames = len(states)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.5, 1.6)
        ax.set_aspect('equal')
        ax.set_title(f'{title}  |  总奖励 = {total_r:.0f}  |  步数 = {n_frames}',
                     fontsize=12, fontweight='bold')

        # 地面
        ax.axhline(0, color='#888', linewidth=2, zorder=1)
        # 轨道刻度
        for tick in np.arange(-2.4, 2.5, 0.6):
            ax.plot([tick, tick], [-0.05, 0], color='#aaa', linewidth=0.8)

        cart_w, cart_h = 0.4, 0.2
        pole_len = 0.6

        cart = plt.Rectangle((0, 0), cart_w, cart_h,
                             fc='#3B82F6', ec='#1E40AF', zorder=3)
        ax.add_patch(cart)
        pole, = ax.plot([], [], color='#EF4444', linewidth=4, zorder=4)
        pole_fallen, = ax.plot([], [], color='#9CA3AF', linewidth=4, zorder=4)
        tip_traj, = ax.plot([], [], color='#F59E0B', alpha=0.4, linewidth=1.2, zorder=2)
        info = ax.text(-2.9, 1.45, '', fontsize=9, family='monospace')

        tip_x, tip_y = [], []

        def init():
            cart.set_xy((-cart_w / 2, 0))
            pole.set_data([], [])
            pole_fallen.set_data([], [])
            tip_traj.set_data([], [])
            info.set_text('')
            return cart, pole, pole_fallen, tip_traj, info

        def update(frame):
            x, _, theta, theta_dot = states[frame]
            a = actions[frame]
            r = rewards[frame]

            cart.set_xy((x - cart_w / 2, 0))
            # 终止时放大角度，让杆子明显倒下（视觉化倒下）
            is_last = (frame == n_frames - 1) and (abs(np.degrees(theta)) > 12)
            disp_theta = theta
            if is_last:
                # 视觉化倒下：放大到 70 度
                sign = 1 if theta > 0 else -1
                disp_theta = sign * np.radians(70)

            pole_x = [x, x + pole_len * np.sin(disp_theta)]
            pole_y = [cart_h, cart_h + pole_len * np.cos(disp_theta)]
            if is_last:
                pole_fallen.set_data(pole_x, pole_y)
                pole.set_data([], [])
            else:
                pole.set_data(pole_x, pole_y)
                pole_fallen.set_data([], [])

            tip_x.append(pole_x[1])
            tip_y.append(pole_y[1])
            tip_traj.set_data(tip_x, tip_y)

            arrow = '\u2192' if a == 1 else '\u2190'
            status = '  <<< FALLEN' if is_last else ''
            info.set_text(
                f'step={frame + 1:3d}  action={arrow}\n'
                f'x={x:+.2f}  \u03b8={np.degrees(theta):+6.1f}\u00b0\n'
                f'reward={r:.1f}  total={rewards[:frame + 1].sum():.0f}{status}'
            )
            return cart, pole, pole_fallen, tip_traj, info

        anim = animation.FuncAnimation(
            fig, update, frames=n_frames, init_func=init,
            interval=30, blit=True, repeat=False)
        plt.close(fig)
        return HTML(anim.to_jshtml())


def visualize_cartpole(policy_fn, title='CartPole 策略可视化', seed=999, max_steps=500):
    """便捷函数：一步生成 CartPole 动画。

    参数:
        policy_fn: callable, 输入 state(np.ndarray) 返回 action(int)
        title:     动画标题
        seed:      随机种子
        max_steps: 最大步数
    """
    viz = CartPoleVisualizer(seed=seed, max_steps=max_steps)
    return viz.animate(policy_fn, title=title)


# ──────────────────────────────────────────────────────────────
# FrozenLake 可视化（ch02 策略评估 / ch03 动态规划）
# ──────────────────────────────────────────────────────────────

_FROZEN_LAKE_DESC_4x4 = [
    'SFFF',
    'FHFH',
    'FFFH',
    'HFFG',
]

_FL_CELL_COLORS = {'S': '#86efac', 'F': '#dbeafe', 'H': '#fca5a5', 'G': '#fde047'}
_FL_CELL_LABELS = {'S': 'S', 'F': 'F', 'H': 'H', 'G': 'G'}
_FL_ARROWS = {0: '\u2190', 1: '\u2193', 2: '\u2192', 3: '\u2191'}


def _draw_frozen_lake_grid(ax, desc):
    """在 ax 上绘制 FrozenLake 4x4 网格背景。"""
    rows, cols = len(desc), len(desc[0])
    for r in range(rows):
        for c in range(cols):
            cell = desc[r][c]
            ax.add_patch(plt.Rectangle((c, rows - 1 - r), 1, 1,
                                       facecolor=_FL_CELL_COLORS[cell],
                                       edgecolor='#6b7280', linewidth=1.5, zorder=1))
            ax.text(c + 0.5, rows - 1 - r + 0.5, _FL_CELL_LABELS[cell],
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    color='#374151', zorder=3)
    ax.set_xlim(-0.05, cols + 0.05)
    ax.set_ylim(-0.05, rows + 0.05)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])


def _draw_frozen_lake_content(ax, V, policy, desc):
    """在 ax 上绘制 FrozenLake 网格、价值数值和策略箭头。"""
    rows, cols = len(desc), len(desc[0])
    _draw_frozen_lake_grid(ax, desc)

    # 价值数值
    for r in range(rows):
        for c in range(cols):
            s = r * cols + c
            ax.text(c + 0.5, rows - 1 - r + 0.85, f'{V[s]:.2f}',
                    ha='center', va='center', fontsize=8, color='#1e3a5f', zorder=4)

    # 策略箭头
    if policy is not None:
        for r in range(rows):
            for c in range(cols):
                s = r * cols + c
                cell = desc[r][c]
                if cell in ('H', 'G'):
                    continue
                a = int(policy[s].argmax())
                ax.text(c + 0.5, rows - 1 - r + 0.2, _FL_ARROWS[a],
                        ha='center', va='center', fontsize=18, fontweight='bold',
                        color='#7c2d12', zorder=5)


def visualize_frozen_lake(V, policy=None, title='FrozenLake 价值函数', desc=None):
    """可视化 FrozenLake 状态价值函数和策略。

    参数:
        V:       np.ndarray, shape=(16,), 状态价值
        policy:  np.ndarray, shape=(16,4) 或 None, 策略概率分布
        title:   标题
        desc:    地图描述, 默认标准 4x4
    """
    if desc is None:
        desc = _FROZEN_LAKE_DESC_4x4
    fig, ax = plt.subplots(figsize=(5.5, 5.5), constrained_layout=True)
    _draw_frozen_lake_content(ax, V, policy, desc)
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.show()


def visualize_frozen_lake_comparison(V_slip, pi_slip, V_nonslip, pi_nonslip,
                                      title='FrozenLake 最优策略对比'):
    """并排对比湿滑/非湿滑 FrozenLake 的最优策略与价值。

    参数:
        V_slip:    np.ndarray, 湿滑环境 V*
        pi_slip:   np.ndarray, 湿滑环境最优策略
        V_nonslip: np.ndarray, 非湿滑环境 V*
        pi_nonslip: np.ndarray, 非湿滑环境最优策略
        title:     总标题
    """
    desc = _FROZEN_LAKE_DESC_4x4
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)
    sub_titles = ['is_slippery=True (湿滑, 反直觉)', 'is_slippery=False (非湿滑, 直觉)']
    Vs = [V_slip, V_nonslip]
    pis = [pi_slip, pi_nonslip]
    for ax, V, pi, sub in zip(axes, Vs, pis, sub_titles):
        _draw_frozen_lake_content(ax, V, pi, desc)
        ax.set_title(sub, fontsize=11, fontweight='bold')
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.show()


# ──────────────────────────────────────────────────────────────
# Blackjack 策略可视化（ch04 蒙特卡洛）
# ──────────────────────────────────────────────────────────────


def visualize_blackjack_value(Q, title='Blackjack 状态价值 V(s)'):
    """可视化 Blackjack 状态价值 V(s)=max_a Q(s,a)。

    参数:
        Q: defaultdict, key=(player_sum, dealer, usable_ace), value=np.array([Q_停, Q_要])
        title: 标题
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    titles_sub = ['无可用 A (Hard)', '有可用 A (Soft)']

    for idx, usable_ace in enumerate([False, True]):
        ax = axes[idx]
        V = np.full((10, 10), np.nan)  # rows=player 12-21, cols=dealer 1-10
        for pi, player in enumerate(range(12, 22)):
            for di, dealer in enumerate(range(1, 11)):
                state = (player, dealer, usable_ace)
                if state in Q:
                    V[pi, di] = float(Q[state].max())

        im = ax.imshow(V, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto',
                       extent=[0.5, 10.5, 11.5, 21.5], origin='lower')

        for x in np.arange(0.5, 11.5, 1):
            ax.axvline(x, color='white', linewidth=0.5)
        for y in np.arange(11.5, 22.5, 1):
            ax.axhline(y, color='white', linewidth=0.5)

        for pi, player in enumerate(range(12, 22)):
            for di, dealer in enumerate(range(1, 11)):
                state = (player, dealer, usable_ace)
                if state in Q:
                    v = float(Q[state].max())
                    ax.text(dealer, player, f'{v:+.2f}', ha='center', va='center',
                            fontsize=7, color='#1f2937')

        ax.set_xlabel('庄家明牌'); ax.set_ylabel('玩家点数')
        ax.set_title(titles_sub[idx], fontsize=11, fontweight='bold')
        ax.set_xticks(range(1, 11)); ax.set_yticks(range(12, 22))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    plt.show()


def visualize_blackjack_policy(Q, title='Blackjack 学到的策略'):
    """可视化 Blackjack 策略：玩家点数 vs 庄家明牌。

    参数:
        Q: defaultdict, key=(player_sum, dealer, usable_ace), value=np.array([Q_停, Q_要])
        title: 标题
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    titles_sub = ['无可用 A (Hard)', '有可用 A (Soft)']

    for idx, usable_ace in enumerate([False, True]):
        ax = axes[idx]
        # 玩家点数 12-21, 庄家明牌 1-10
        grid = np.full((10, 10), -1)  # rows=player, cols=dealer
        for pi, player in enumerate(range(12, 22)):
            for di, dealer in enumerate(range(1, 11)):
                state = (player, dealer, usable_ace)
                if state in Q:
                    grid[pi, di] = int(Q[state].argmax())

        # 画热力图
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['#3B82F6', '#EF4444'])  # 0=停(蓝) 1=要(红)
        ax.imshow(grid, cmap=cmap, vmin=-1, vmax=1, aspect='auto',
                  extent=[0.5, 10.5, 11.5, 21.5], origin='lower')

        # 网格线
        for x in np.arange(0.5, 11.5, 1):
            ax.axvline(x, color='white', linewidth=0.5)
        for y in np.arange(11.5, 22.5, 1):
            ax.axhline(y, color='white', linewidth=0.5)

        ax.set_xlabel('庄家明牌'); ax.set_ylabel('玩家点数')
        ax.set_title(titles_sub[idx], fontsize=11, fontweight='bold')
        ax.set_xticks(range(1, 11)); ax.set_yticks(range(12, 22))

        # 标注文字
        for pi, player in enumerate(range(12, 22)):
            for di, dealer in enumerate(range(1, 11)):
                state = (player, dealer, usable_ace)
                if state in Q:
                    a = int(Q[state].argmax())
                    label = '\u505c' if a == 0 else '\u8981'
                    ax.text(dealer, player, label, ha='center', va='center',
                            fontsize=8, color='white', fontweight='bold')

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3B82F6', label='\u505c\u724c (Stick)'),
                        Patch(facecolor='#EF4444', label='\u8981\u724c (Hit)')]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=10)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    plt.show()


# ──────────────────────────────────────────────────────────────
# CliffWalking 路径可视化（ch05 SARSA vs Q-Learning）
# ──────────────────────────────────────────────────────────────


def _cliffwalking_path(Q, env):
    """从 Q 表提取贪婪路径。"""
    s, _ = env.reset()
    path = [s]
    visited = {s}
    for _ in range(200):
        a = int(Q[s].argmax())
        s, _, terminated, truncated, _ = env.step(a)
        path.append(s)
        if s in visited:
            break
        visited.add(s)
        if terminated or truncated:
            break
    return path


def visualize_cliffwalking(Q_sarsa, Q_ql, title='CliffWalking: SARSA vs Q-Learning 路径'):
    """并排可视化 SARSA 和 Q-Learning 在 CliffWalking 上学到的路径。

    参数:
        Q_sarsa: np.ndarray, shape=(48,4), SARSA 的 Q 表
        Q_ql:    np.ndarray, shape=(48,4), Q-Learning 的 Q 表
        title:   标题
    """
    env = gym.make('CliffWalking-v1')
    path_sarsa = _cliffwalking_path(Q_sarsa, env)
    path_ql = _cliffwalking_path(Q_ql, env)
    env.close()

    rows, cols = 4, 12
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.5), constrained_layout=True)
    labels = ['SARSA (on-policy)', 'Q-Learning (off-policy)']
    paths = [path_sarsa, path_ql]
    colors = ['#3B82F6', '#EF4444']

    for idx, (ax, path, label, color) in enumerate(zip(axes, paths, labels, colors)):
        # 背景色
        for r in range(rows):
            for c in range(cols):
                s = r * cols + c
                if r == 3 and 1 <= c <= 10:
                    fc = '#fca5a5'  # 悬崖
                elif s == 36:
                    fc = '#86efac'  # 起点
                elif s == 47:
                    fc = '#fde047'  # 终点
                else:
                    fc = '#f0f9ff'
                ax.add_patch(plt.Rectangle((c, rows - 1 - r), 1, 1,
                             facecolor=fc, edgecolor='#cbd5e1', linewidth=0.8))
        # 路径连线
        coords = [(s % cols + 0.5, rows - 1 - s // cols + 0.5) for s in path]
        xs, ys = zip(*coords)
        ax.plot(xs, ys, color=color, linewidth=2.5, marker='o',
                markersize=4, zorder=5)

        # 标注起点终点
        ax.text(0.5, 0.5, 'S', ha='center', va='center', fontsize=11, fontweight='bold')
        ax.text(11.5, 0.5, 'G', ha='center', va='center', fontsize=11, fontweight='bold')
        if 1 <= len(path) <= 50:
            ax.text(6, 4.5, f'{len(path)-1} \u6b65', ha='center', fontsize=9, color=color)

        ax.set_xlim(-0.1, cols + 0.1); ax.set_ylim(-0.1, rows + 0.6)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, fontsize=11, fontweight='bold')

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.show()
