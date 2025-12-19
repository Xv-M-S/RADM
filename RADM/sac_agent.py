"""
Soft Actor-Critic (SAC) Agent for Layout Optimization
适用于连续控制的布局优化强化学习算法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import deque
import random
import copy


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done: bool):
        """存储经验"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """采样批次经验"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32),
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """SAC Actor网络 - 策略网络"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # 初始化输出层
        self.mu_head.weight.data.uniform_(-3e-3, 3e-3)
        self.mu_head.bias.data.uniform_(-3e-3, 3e-3)
        self.log_std_head.weight.data.uniform_(-3e-3, 3e-3)
        self.log_std_head.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        x = self.net(state)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -20, 2)  # 限制log_std范围

        return mu, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """从策略中采样动作"""
        mu, log_std = self.forward(state)
        std = log_std.exp()

        # 重参数化技巧
        eps = torch.randn_like(std)
        action = mu + eps * std

        # 计算log概率 (用于SAC的熵项)
        log_prob = self._normal_log_prob(mu, std, action)

        # 应用tanh激活函数 (动作范围限制)
        action = torch.tanh(action)

        # 调整log概率 (tanh的雅可比行列式)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)

        return action, log_prob

    def _normal_log_prob(self, mu: torch.Tensor, std: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """计算正态分布的对数概率"""
        var = std.pow(2)
        log_prob = -((x - mu) ** 2) / (2 * var) - torch.log(std) - np.log(np.sqrt(2 * np.pi))
        return log_prob.sum(dim=-1, keepdim=True)


class Critic(nn.Module):
    """SAC Critic网络 - 价值网络"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class SACAgent:
    """Soft Actor-Critic Agent"""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 lr: float = 3e-4,
                 batch_size: int = 256,
                 buffer_capacity: int = 100000,
                 device: str = 'cuda'):
        """
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度
            gamma: 折扣因子
            tau: 软更新系数
            alpha: 熵温度系数
            lr: 学习率
            batch_size: 批次大小
            buffer_capacity: 经验缓冲区容量
            device: 计算设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size

        # 网络初始化
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        # 自动熵调整 (可选)
        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=self.device)
        self.alpha_optimizer = Adam([self.log_alpha], lr=lr)

        # 优化器
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=lr)

        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # 训练统计
        self.train_step = 0

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            if deterministic:
                # 确定性策略：使用均值
                mu, _ = self.actor(state_tensor)
                action = torch.tanh(mu)
            else:
                # 随机策略：采样
                action, _ = self.actor.sample(state_tensor)

            return action.cpu().numpy().flatten()

    def store_transition(self, state: np.ndarray, action: np.ndarray,
                        reward: float, next_state: np.ndarray, done: bool):
        """存储经验"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Dict[str, float]:
        """更新网络参数"""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # 采样批次
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 更新Critic网络
        critic_loss = self._update_critics(states, actions, rewards, next_states, dones)

        # 更新Actor网络
        actor_loss = self._update_actor(states)

        # 更新熵温度系数
        alpha_loss = self._update_alpha(states)

        # 软更新目标网络
        self._soft_update_targets()

        self.train_step += 1

        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.log_alpha.exp().item()
        }

    def _update_critics(self, states: torch.Tensor, actions: torch.Tensor,
                       rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor) -> float:
        """更新Critic网络"""

        with torch.no_grad():
            # 从目标策略中采样下一个动作
            next_actions, next_log_probs = self.actor.sample(next_states)

            # 计算目标Q值
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # 当前Q值
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        # 计算损失
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = critic1_loss + critic2_loss

        # 反向传播
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        return critic_loss.item()

    def _update_actor(self, states: torch.Tensor) -> float:
        """更新Actor网络"""

        # 从当前策略采样动作
        actions, log_probs = self.actor.sample(states)

        # 计算Q值
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        min_q = torch.min(q1, q2)

        # Actor损失 = E[α * logπ - Q]
        actor_loss = (self.alpha * log_probs - min_q).mean()

        # 反向传播
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def _update_alpha(self, states: torch.Tensor) -> float:
        """更新熵温度系数"""
        with torch.no_grad():
            _, log_probs = self.actor.sample(states)

        # Alpha损失 = E[-α * (logπ + H_target)]
        alpha_loss = (-self.log_alpha * (log_probs + self.target_entropy)).mean()

        # 反向传播
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return alpha_loss.item()

    def _soft_update_targets(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])
        self.log_alpha = checkpoint['log_alpha']
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])


class LayoutSACTrainer:
    """布局优化专用SAC训练器"""

    def __init__(self,
                 env,
                 agent: SACAgent,
                 max_episodes: int = 1000,
                 max_steps_per_episode: int = 50,
                 update_interval: int = 1,
                 eval_interval: int = 100,
                 save_interval: int = 500,
                 log_dir: str = './logs'):
        """
        Args:
            env: 布局RL环境
            agent: SAC智能体
            max_episodes: 最大训练回合数
            max_steps_per_episode: 每回合最大步数
            update_interval: 更新间隔
            eval_interval: 评估间隔
            save_interval: 保存间隔
            log_dir: 日志目录
        """
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.update_interval = update_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.log_dir = log_dir

        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.train_losses = []

    def train(self, initial_layouts: List[Dict], constraints_list: List[List[Dict]]):
        """
        训练SAC智能体

        Args:
            initial_layouts: 初始布局列表，每个包含boxes和text_features
            constraints_list: 对应的约束列表
        """
        print("开始SAC训练...")

        for episode in range(self.max_episodes):
            # 随机选择一个训练布局
            layout_idx = np.random.randint(len(initial_layouts))
            layout_data = initial_layouts[layout_idx]
            constraints = constraints_list[layout_idx]

            # 重置环境
            state = self.env.reset(
                layout_data['boxes'],
                constraints,
                layout_data.get('text_features'),
                layout_data.get('target_boxes')
            )

            episode_reward = 0
            episode_steps = 0

            # 回合内循环
            while episode_steps < self.max_steps_per_episode:
                # 选择动作
                action = self.agent.select_action(state)

                # 执行动作
                next_state, reward, done, info = self.env.step(action)

                # 存储经验
                self.agent.store_transition(state, action, reward, next_state, done)

                # 更新智能体
                if episode_steps % self.update_interval == 0:
                    loss_info = self.agent.update()
                    if loss_info:
                        self.train_losses.append(loss_info)

                episode_reward += reward
                episode_steps += 1
                state = next_state

                if done:
                    break

            # 记录统计信息
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)

            # 打印训练信息
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.3f}, Avg Length: {avg_length:.1f}")

            # 评估
            if episode % self.eval_interval == 0:
                self._evaluate()

            # 保存模型
            if episode % self.save_interval == 0:
                self.agent.save(f"{self.log_dir}/sac_episode_{episode}.pth")

        print("训练完成！")

    def _evaluate(self):
        """评估当前智能体性能"""
        eval_episodes = 10
        eval_rewards = []

        for _ in range(eval_episodes):
            # 使用确定性策略进行评估
            state = self.env.reset()  # 使用默认布局
            episode_reward = 0
            done = False

            while not done:
                action = self.agent.select_action(state, deterministic=True)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state

            eval_rewards.append(episode_reward)

        avg_eval_reward = np.mean(eval_rewards)
        print(f"评估结果 - 平均奖励: {avg_eval_reward:.3f}")

        return avg_eval_reward

    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'train_losses': self.train_losses
        }
