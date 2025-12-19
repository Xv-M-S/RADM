"""
联合训练管理器
协调RADM扩散模型和RL智能体的联合训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import os
import time
from pathlib import Path
import json
from collections import defaultdict
import matplotlib.pyplot as plt

from .rl_diffusion_integrator import RLGuidedRADM, CurriculumRLGuidance, RLDataCollector
from .sac_agent import SACAgent, LayoutSACTrainer
from .data_collection import LayoutPreferenceDataset, PreferenceLearningModel
from .rl_layout_env import LayoutRLEnvironment


class JointTrainingDataset(Dataset):
    """
    联合训练数据集
    包含布局生成数据和偏好学习数据
    """

    def __init__(self,
                 layout_data: List[Dict],
                 preference_dataset: Optional['LayoutPreferenceDataset'] = None,
                 preference_weight: float = 0.3):
        """
        Args:
            layout_data: 布局生成训练数据
            preference_dataset: 偏好学习数据集
            preference_weight: 偏好学习在训练中的权重
        """
        self.layout_data = layout_data
        self.preference_dataset = preference_dataset
        self.preference_weight = preference_weight

        # 计算采样概率
        self.layout_prob = 1.0 - preference_weight
        self.preference_prob = preference_weight

    def __len__(self):
        # 返回更大的长度以支持混合采样
        base_len = len(self.layout_data)
        if self.preference_dataset and len(self.preference_dataset.preferences) > 0:
            pref_len = len(self.preference_dataset.preferences)
            return max(base_len, pref_len) * 2  # 混合采样
        return base_len

    def __getitem__(self, idx):
        # 随机决定采样类型
        if self.preference_dataset and np.random.random() < self.preference_prob:
            # 采样偏好学习数据
            return self._get_preference_sample()
        else:
            # 采样布局生成数据
            layout_idx = idx % len(self.layout_data)
            return self._get_layout_sample(layout_idx)

    def _get_layout_sample(self, idx):
        """获取布局生成样本"""
        data = self.layout_data[idx].copy()

        # 添加数据类型标识
        data['data_type'] = 'layout_generation'

        return data

    def _get_preference_sample(self):
        """获取偏好学习样本"""
        if not self.preference_dataset:
            return self._get_layout_sample(0)

        # 随机采样偏好对
        pref_idx = np.random.randint(len(self.preference_dataset.preferences))
        pref = self.preference_dataset.preferences[pref_idx]

        layout_a = self.preference_dataset.layouts[pref['layout_a_idx']]
        layout_b = self.preference_dataset.layouts[pref['layout_b_idx']]

        return {
            'data_type': 'preference_learning',
            'layout_a': layout_a,
            'layout_b': layout_b,
            'preference': pref['preference'],
            'constraints': pref.get('constraints', []),
            'text_desc': pref.get('text_desc', '')
        }


class JointTrainer:
    """
    RL-Diffusion联合训练器
    协调扩散模型和RL智能体的端到端训练
    """

    def __init__(self,
                 radm_model,
                 sac_agent: SACAgent,
                 preference_model: Optional[PreferenceLearningModel] = None,
                 device: str = 'cuda',
                 config: Dict[str, Any] = None):
        """
        Args:
            radm_model: RADM扩散模型
            sac_agent: SAC智能体
            preference_model: 偏好学习模型 (可选)
            device: 计算设备
            config: 训练配置
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 模型
        self.radm_model = radm_model.to(self.device)
        self.sac_agent = sac_agent
        self.preference_model = preference_model.to(self.device) if preference_model else None

        # RL引导的RADM
        self.rl_guided_radm = RLGuidedRADM(
            self.radm_model,
            self.sac_agent,
            config.get('guidance_config', {})
        ).to(self.device)

        # 优化器
        self.radm_optimizer = optim.AdamW(
            self.radm_model.parameters(),
            lr=config.get('radm_lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-4)
        )

        if self.preference_model:
            self.preference_optimizer = optim.Adam(
                self.preference_model.parameters(),
                lr=config.get('preference_lr', 1e-3)
            )

        # 课程学习
        self.curriculum = CurriculumRLGuidance(
            initial_guidance_scale=config.get('initial_guidance_scale', 0.0),
            final_guidance_scale=config.get('final_guidance_scale', 0.2),
            curriculum_steps=config.get('curriculum_steps', 10000)
        )

        # 数据收集器
        self.rl_env = LayoutRLEnvironment(max_boxes=config.get('max_boxes', 10))
        self.data_collector = RLDataCollector(
            self.rl_env,
            self.sac_agent,
            collection_interval=config.get('collection_interval', 10)
        )

        # 训练配置
        self.config = config or {}
        self.global_step = 0

        # 训练统计
        self.stats = defaultdict(list)
        self.best_reward = -float('inf')

        # 创建输出目录
        self.output_dir = Path(self.config.get('output_dir', './outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self,
              train_dataset: JointTrainingDataset,
              val_dataset: Optional[JointTrainingDataset] = None,
              num_epochs: int = 100,
              batch_size: int = 16,
              log_interval: int = 100,
              eval_interval: int = 1000,
              save_interval: int = 5000):
        """
        执行联合训练

        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            num_epochs: 训练轮数
            batch_size: 批次大小
            log_interval: 日志间隔
            eval_interval: 评估间隔
            save_interval: 保存间隔
        """

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )

        print("🚀 开始RL-Diffusion联合训练...")
        print(f"📊 训练配置: {self.config}")
        print(f"📈 总训练步数: {len(train_loader) * num_epochs}")

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            # 训练一个epoch
            self._train_epoch(train_loader, epoch, log_interval)

            epoch_time = time.time() - epoch_start_time

            # 评估
            if val_dataset and (epoch + 1) % (eval_interval // len(train_loader)) == 0:
                self._evaluate(val_dataset, epoch)

            # 保存模型
            if (epoch + 1) % (save_interval // len(train_loader)) == 0:
                self._save_checkpoint(epoch)

            print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")

        print("🎉 联合训练完成！")

    def _train_epoch(self, train_loader, epoch, log_interval):
        """训练一个epoch"""
        self.radm_model.train()
        if self.preference_model:
            self.preference_model.train()

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            self.global_step += 1

            # 更新课程学习进度
            self.curriculum.update_step()
            current_guidance_scale = self.curriculum.get_current_scale()
            self.rl_guided_radm.set_guidance_scale(current_guidance_scale)

            # 处理不同类型的数据
            data_type = batch.get('data_type', ['layout_generation'] * len(batch))

            if isinstance(data_type, list) and data_type[0] == 'layout_generation':
                # 布局生成训练
                loss = self._train_layout_generation(batch)
            elif isinstance(data_type, list) and data_type[0] == 'preference_learning':
                # 偏好学习训练
                loss = self._train_preference_learning(batch)
            else:
                # 混合批次 - 分别处理
                loss = self._train_mixed_batch(batch)

            # 反向传播
            if isinstance(loss, dict):
                total_loss = sum(loss.values())
            else:
                total_loss = loss

            total_loss.backward()

            # 梯度裁剪
            if self.config.get('grad_clip', 1.0) > 0:
                nn.utils.clip_grad_norm_(self.radm_model.parameters(), self.config['grad_clip'])
                if self.preference_model:
                    nn.utils.clip_grad_norm_(self.preference_model.parameters(), self.config['grad_clip'])

            # 优化器步骤
            self.radm_optimizer.step()
            if self.preference_model and 'preference' in str(total_loss):
                self.preference_optimizer.step()

            # 清零梯度
            self.radm_optimizer.zero_grad()
            if self.preference_model:
                self.preference_optimizer.zero_grad()

            # 更新RL智能体
            if self.global_step % self.config.get('rl_update_interval', 1) == 0:
                rl_loss = self.sac_agent.update()
                if rl_loss:
                    for k, v in rl_loss.items():
                        self.stats[f'rl_{k}'].append(v)

            # 收集RL数据
            self._collect_rl_data_from_batch(batch)

            # 记录统计
            epoch_loss += total_loss.item()
            num_batches += 1

            # 日志输出
            if self.global_step % log_interval == 0:
                avg_loss = epoch_loss / num_batches
                current_lr = self.radm_optimizer.param_groups[0]['lr']

                log_info = {
                    'epoch': epoch,
                    'step': self.global_step,
                    'loss': avg_loss,
                    'guidance_scale': current_guidance_scale,
                    'lr': current_lr
                }

                # 添加RL统计
                if self.stats['rl_actor_loss']:
                    log_info['rl_actor_loss'] = np.mean(self.stats['rl_actor_loss'][-10:])
                    log_info['rl_critic_loss'] = np.mean(self.stats['rl_critic_loss'][-10:])

                print(f"Step {self.global_step}: {log_info}")

                # 记录到统计
                for k, v in log_info.items():
                    if isinstance(v, (int, float)):
                        self.stats[k].append(v)

    def _train_layout_generation(self, batch) -> Dict[str, torch.Tensor]:
        """训练布局生成任务"""
        # 将批次数据移到设备
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
            elif isinstance(value, list):
                batch[key] = [v.to(self.device) if isinstance(v, torch.Tensor) else v for v in value]

        # 前向传播 (使用RL引导的RADM)
        outputs = self.rl_guided_radm(batch['x_t'], batch['t'], batch['cond'])

        # 计算扩散损失 (根据RADM的具体实现调整)
        # 这里是简化版本，实际需要根据RADM的损失函数实现
        diffusion_loss = self._compute_diffusion_loss(outputs, batch)

        return {'diffusion': diffusion_loss}

    def _train_preference_learning(self, batch) -> Dict[str, torch.Tensor]:
        """训练偏好学习任务"""
        if not self.preference_model:
            return {'preference': torch.tensor(0.0, device=self.device)}

        layout_a = batch['layout_a'].to(self.device)
        layout_b = batch['layout_b'].to(self.device)
        preferences = batch['preference'].to(self.device)

        # 前向传播
        pred_logits, quality_a, quality_b = self.preference_model(layout_a, layout_b)

        # 计算损失
        preference_loss = F.cross_entropy(pred_logits, preferences + 1)  # 转换为0,1,2

        # 质量预测损失 (可选)
        if 'quality_a' in batch and 'quality_b' in batch:
            quality_a_target = batch['quality_a'].to(self.device)
            quality_b_target = batch['quality_b'].to(self.device)
            quality_loss = F.mse_loss(quality_a.squeeze(), quality_a_target) + \
                           F.mse_loss(quality_b.squeeze(), quality_b_target)
            total_loss = preference_loss + 0.1 * quality_loss
        else:
            total_loss = preference_loss
            quality_loss = torch.tensor(0.0, device=self.device)

        return {
            'preference': preference_loss,
            'quality': quality_loss
        }

    def _train_mixed_batch(self, batch):
        """处理混合批次"""
        # 分离不同类型的数据
        layout_mask = [dt == 'layout_generation' for dt in batch['data_type']]
        preference_mask = [dt == 'preference_learning' for dt in batch['data_type']]

        total_loss = {}

        if any(layout_mask):
            # 布局生成子批次
            layout_batch = {k: v[layout_mask] if isinstance(v, (list, torch.Tensor)) else v
                           for k, v in batch.items()}
            layout_loss = self._train_layout_generation(layout_batch)
            total_loss.update({f'layout_{k}': v for k, v in layout_loss.items()})

        if any(preference_mask):
            # 偏好学习子批次
            pref_batch = {k: v[preference_mask] if isinstance(v, (list, torch.Tensor)) else v
                         for k, v in batch.items()}
            pref_loss = self._train_preference_learning(pref_batch)
            total_loss.update({f'pref_{k}': v for k, v in pref_loss.items()})

        return total_loss

    def _compute_diffusion_loss(self, outputs, batch) -> torch.Tensor:
        """计算扩散模型损失 (需要根据RADM具体实现调整)"""
        # 这里是简化实现
        # 实际需要根据RADM的损失计算逻辑
        return torch.mean(outputs ** 2)

    def _collect_rl_data_from_batch(self, batch):
        """从训练批次收集RL数据"""
        # 从当前布局状态收集RL交互数据
        if 'current_layout' in batch:
            layout = batch['current_layout']
            cond = batch.get('cond', {})

            # 随机采样一些布局进行RL数据收集
            if np.random.random() < 0.1:  # 10%概率收集
                self.data_collector.collect_from_diffusion_step(
                    layout, cond
                )

    def _evaluate(self, val_dataset, epoch):
        """评估模型性能"""
        self.radm_model.eval()
        if self.preference_model:
            self.preference_model.eval()

        val_loader = DataLoader(val_dataset, batch_size=self.config.get('eval_batch_size', 32))

        val_losses = []
        rl_rewards = []

        with torch.no_grad():
            for batch in val_loader:
                # 计算验证损失
                if batch.get('data_type', ['layout_generation'])[0] == 'layout_generation':
                    loss = self._train_layout_generation(batch)
                    val_losses.append(sum(loss.values()).item())
                else:
                    loss = self._train_preference_learning(batch)
                    val_losses.append(sum(loss.values()).item())

                # 评估RL性能
                if hasattr(batch, 'current_layout'):
                    # 简单的RL评估
                    reward = self._evaluate_rl_performance(batch)
                    rl_rewards.append(reward)

        avg_val_loss = np.mean(val_losses)
        avg_rl_reward = np.mean(rl_rewards) if rl_rewards else 0.0

        print(f"📊 验证结果 - Loss: {avg_val_loss:.4f}, RL Reward: {avg_rl_reward:.4f}")

        # 保存最佳模型
        if avg_rl_reward > self.best_reward:
            self.best_reward = avg_rl_reward
            self._save_checkpoint(epoch, suffix='best')

        self.stats['val_loss'].append(avg_val_loss)
        self.stats['val_rl_reward'].append(avg_rl_reward)

    def _evaluate_rl_performance(self, batch) -> float:
        """评估RL性能"""
        # 简化实现：返回固定的评估分数
        return 0.5

    def _save_checkpoint(self, epoch, suffix=''):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'radm_model': self.radm_model.state_dict(),
            'radm_optimizer': self.radm_optimizer.state_dict(),
            'sac_agent': self.sac_agent.__dict__ if hasattr(self.sac_agent, '__dict__') else {},
            'curriculum': self.curriculum.__dict__,
            'stats': dict(self.stats),
            'config': self.config
        }

        if self.preference_model:
            checkpoint['preference_model'] = self.preference_model.state_dict()
            checkpoint['preference_optimizer'] = self.preference_optimizer.state_dict()

        filename = f'checkpoint_epoch_{epoch}'
        if suffix:
            filename += f'_{suffix}'
        filename += '.pth'

        torch.save(checkpoint, self.output_dir / filename)
        print(f"💾 保存检查点: {filename}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path)

        self.radm_model.load_state_dict(checkpoint['radm_model'])
        self.radm_optimizer.load_state_dict(checkpoint['radm_optimizer'])
        self.global_step = checkpoint['global_step']
        self.stats.update(checkpoint['stats'])

        if self.preference_model and 'preference_model' in checkpoint:
            self.preference_model.load_state_dict(checkpoint['preference_model'])
            self.preference_optimizer.load_state_dict(checkpoint['preference_optimizer'])

        print(f"📂 加载检查点: {checkpoint_path}")

    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss曲线
        if self.stats['loss']:
            axes[0, 0].plot(self.stats['loss'])
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Loss')

        # RL损失曲线
        if self.stats['rl_actor_loss']:
            axes[0, 1].plot(self.stats['rl_actor_loss'], label='Actor Loss')
            axes[0, 1].plot(self.stats['rl_critic_loss'], label='Critic Loss')
            axes[0, 1].set_title('RL Training Losses')
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()

        # 引导强度曲线
        if self.stats['guidance_scale']:
            axes[1, 0].plot(self.stats['guidance_scale'])
            axes[1, 0].set_title('RL Guidance Scale (Curriculum)')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Guidance Scale')

        # 验证性能曲线
        if self.stats['val_rl_reward']:
            axes[1, 1].plot(self.stats['val_rl_reward'])
            axes[1, 1].set_title('Validation RL Reward')
            axes[1, 1].set_xlabel('Epochs')
            axes[1, 1].set_ylabel('Reward')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()


class TrainingConfig:
    """训练配置管理"""

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """获取默认训练配置"""
        return {
            # 模型配置
            'max_boxes': 10,
            'hidden_dim': 256,
            'num_heads': 8,

            # RL配置
            'rl_state_dim': 200,  # 需要根据实际状态维度调整
            'rl_action_dim': 20,  # max_boxes * 2
            'rl_hidden_dim': 256,
            'rl_buffer_capacity': 100000,
            'rl_batch_size': 256,

            # 训练配置
            'radm_lr': 1e-4,
            'preference_lr': 1e-3,
            'weight_decay': 1e-4,
            'grad_clip': 1.0,

            # 课程学习
            'initial_guidance_scale': 0.0,
            'final_guidance_scale': 0.2,
            'curriculum_steps': 10000,

            # 训练控制
            'rl_update_interval': 1,
            'collection_interval': 10,
            'num_workers': 4,
            'eval_batch_size': 32,

            # 输出配置
            'output_dir': './outputs/joint_training',

            # 引导配置
            'guidance_config': {
                'guidance_scale': 0.1,
                'rl_steps': 3,
                'guidance_start_step': 10,
                'guidance_end_step': 40,
                'enable_rl_guidance': True
            }
        }

    @staticmethod
    def get_experiment_config(experiment_name: str) -> Dict[str, Any]:
        """获取特定实验的配置"""
        base_config = TrainingConfig.get_default_config()

        if experiment_name == 'baseline':
            # 基线实验：只训练扩散模型
            base_config['guidance_config']['enable_rl_guidance'] = False
            base_config['initial_guidance_scale'] = 0.0
            base_config['final_guidance_scale'] = 0.0

        elif experiment_name == 'rl_light':
            # 轻度RL引导
            base_config['final_guidance_scale'] = 0.1
            base_config['rl_steps'] = 2

        elif experiment_name == 'rl_medium':
            # 中等RL引导
            base_config['final_guidance_scale'] = 0.2
            base_config['rl_steps'] = 3

        elif experiment_name == 'rl_heavy':
            # 强RL引导
            base_config['final_guidance_scale'] = 0.3
            base_config['rl_steps'] = 5
            base_config['guidance_start_step'] = 5
            base_config['guidance_end_step'] = 45

        elif experiment_name == 'preference_learning':
            # 包含偏好学习的实验
            base_config['preference_weight'] = 0.3

        return base_config


# 使用示例
def create_joint_trainer(radm_model, config_name: str = 'rl_medium'):
    """创建联合训练器的便捷函数"""
    from .sac_agent import SACAgent

    config = TrainingConfig.get_experiment_config(config_name)

    # 创建SAC智能体
    sac_agent = SACAgent(
        state_dim=config['rl_state_dim'],
        action_dim=config['rl_action_dim'],
        hidden_dim=config['rl_hidden_dim'],
        batch_size=config['rl_batch_size'],
        buffer_capacity=config['rl_buffer_capacity']
    )

    # 创建联合训练器
    trainer = JointTrainer(
        radm_model=radm_model,
        sac_agent=sac_agent,
        device='cuda',
        config=config
    )

    return trainer
