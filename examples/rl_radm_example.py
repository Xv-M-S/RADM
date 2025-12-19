#!/usr/bin/env python3
"""
RL增强RADM使用示例
演示如何使用强化学习增强的RADM进行布局生成
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# 导入RL增强的RADM组件
from ..RADM.rl_layout_env import LayoutRLEnvironment
from ..RADM.sac_agent import SACAgent, LayoutSACTrainer
from ..RADM.data_collection import LayoutPreferenceDataset, InteractiveDataCollector, AutomatedPreferenceGenerator
from ..RADM.rl_diffusion_integrator import RLGuidedRADM
from ..RADM.joint_trainer import JointTrainer, TrainingConfig, JointTrainingDataset


def create_sample_layout_data():
    """创建示例布局数据"""
    layouts = []

    # 示例1: 移动端个人主页布局
    layout1 = {
        'boxes': torch.tensor([
            [0.1, 0.05, 0.8, 0.15],  # Navbar
            [0.4, 0.25, 0.2, 0.2],   # Avatar
            [0.1, 0.5, 0.8, 0.1],    # Name
            [0.1, 0.65, 0.8, 0.08],  # Bio
            [0.1, 0.78, 0.35, 0.1],  # Stats
            [0.55, 0.78, 0.35, 0.1], # Gallery
            [0.1, 0.9, 0.8, 0.08],   # Button
        ]),
        'constraints': [
            {'type': 'full_width', 'target': 0},  # Navbar
            {'type': 'below', 'source': 1, 'target': 0},  # Avatar below Navbar
            {'type': 'below', 'source': 2, 'target': 1},  # Name below Avatar
            {'type': 'below', 'source': 3, 'target': 2},  # Bio below Name
            {'type': 'below', 'source': 4, 'target': 3},  # Stats below Bio
            {'type': 'below', 'source': 5, 'target': 3},  # Gallery below Bio
            {'type': 'below', 'source': 6, 'target': 4},  # Button below Stats
            {'type': 'full_width', 'target': 6},  # Button full width
        ],
        'text_features': torch.randn(7, 768),  # 模拟文本特征
        'target_boxes': None
    }

    # 示例2: 仪表盘布局
    layout2 = {
        'boxes': torch.tensor([
            [0.05, 0.05, 0.5, 0.6],   # Main Chart
            [0.6, 0.05, 0.35, 0.25],   # KPI 1
            [0.6, 0.35, 0.35, 0.25],   # KPI 2
            [0.6, 0.65, 0.35, 0.25],   # KPI 3
            [0.05, 0.7, 0.9, 0.25],    # Console
        ]),
        'constraints': [
            {'type': 'right_of', 'source': 1, 'target': 0},  # KPI1 right of Main
            {'type': 'below', 'source': 2, 'target': 1},     # KPI2 below KPI1
            {'type': 'below', 'source': 3, 'target': 2},     # KPI3 below KPI2
            {'type': 'below', 'source': 4, 'target': 0},     # Console below Main
            {'type': 'full_width', 'target': 4},            # Console full width
        ],
        'text_features': torch.randn(5, 768),
        'target_boxes': None
    }

    layouts.extend([layout1, layout2])

    # 生成更多变体
    for i in range(10):
        # 随机扰动
        base_layout = layouts[i % 2].copy()
        noise = torch.randn_like(base_layout['boxes']) * 0.1
        base_layout['boxes'] = torch.clamp(base_layout['boxes'] + noise, 0, 1)
        layouts.append(base_layout)

    return layouts


def example_1_rl_training():
    """示例1: RL智能体训练"""
    print("🚀 示例1: RL智能体训练")
    print("=" * 50)

    # 创建RL环境
    env = LayoutRLEnvironment(max_boxes=10)

    # 创建SAC智能体
    sac_agent = SACAgent(
        state_dim=200,  # 状态维度
        action_dim=20,  # 动作维度 (10 boxes * 2)
        hidden_dim=256,
        batch_size=64,
        buffer_capacity=10000
    )

    # 准备训练数据
    layout_data = create_sample_layout_data()
    constraints_list = [data['constraints'] for data in layout_data]

    # 创建训练器
    trainer = LayoutSACTrainer(
        env=env,
        agent=sac_agent,
        max_episodes=50,  # 简短训练用于演示
        max_steps_per_episode=30,
        update_interval=5
    )

    # 执行训练
    print("开始RL训练...")
    trainer.train(layout_data, constraints_list)

    # 获取训练统计
    stats = trainer.get_training_stats()
    print("训练完成！")
    print(f"最终平均奖励: {np.mean(stats['episode_rewards'][-10:]):.3f}")

    return sac_agent


def example_2_preference_learning():
    """示例2: 偏好学习数据收集"""
    print("\n🎯 示例2: 偏好学习数据收集")
    print("=" * 50)

    # 创建偏好数据集
    preference_dataset = LayoutPreferenceDataset()

    # 创建交互式数据收集器
    collector = InteractiveDataCollector(preference_dataset)

    # 创建自动偏好生成器
    auto_generator = AutomatedPreferenceGenerator(preference_dataset)

    # 生成一些示例布局对
    layout_pairs = []
    constraints_list = []

    for i in range(5):
        layout_a = torch.rand(5, 4) * 0.8 + 0.1  # 随机布局
        layout_b = layout_a + torch.randn_like(layout_a) * 0.05  # 轻微变体
        layout_pairs.append((layout_a, layout_b))
        constraints_list.append([
            {'type': 'below', 'source': 1, 'target': 0},
            {'type': 'right_of', 'source': 2, 'target': 1}
        ])

    # 自动生成偏好数据
    print("自动生成偏好数据...")
    auto_generator.generate_batch_preferences(
        layout_pairs, constraints_list, batch_size=20
    )

    print(f"生成了 {len(preference_dataset.preferences)} 个偏好样本")

    # 显示数据集统计
    stats = preference_dataset.get_statistics()
    print("数据集统计:", stats)

    return preference_dataset


def example_3_rl_guided_generation():
    """示例3: RL引导的布局生成"""
    print("\n🎨 示例3: RL引导的布局生成")
    print("=" * 50)

    # 这里需要一个训练好的RADM模型
    # 为了演示，我们创建一个模拟的模型
    class MockRADM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

        def forward(self, x, t, cond):
            # 模拟扩散模型输出
            return torch.randn_like(x)

    radm_model = MockRADM()

    # 创建SAC智能体 (使用示例1训练的或创建新的)
    sac_agent = SACAgent(
        state_dim=200,
        action_dim=20,
        hidden_dim=256
    )

    # 创建RL引导的RADM
    rl_guided_radm = RLGuidedRADM(
        radm_model,
        sac_agent,
        guidance_config={
            'guidance_scale': 0.2,
            'rl_steps': 3,
            'enable_rl_guidance': True
        }
    )

    print("创建了RL引导的RADM模型")
    print(f"引导强度: {rl_guided_radm.guidance_config['guidance_scale']}")
    print(f"RL步骤数: {rl_guided_radm.guidance_config['rl_steps']}")

    return rl_guided_radm


def example_4_joint_training():
    """示例4: 联合训练流程"""
    print("\n🔄 示例4: 联合训练流程")
    print("=" * 50)

    # 创建训练数据
    layout_data = create_sample_layout_data()

    # 创建偏好数据集 (可选)
    preference_dataset = LayoutPreferenceDataset()

    # 创建联合训练数据集
    joint_dataset = JointTrainingDataset(
        layout_data=layout_data,
        preference_dataset=preference_dataset,
        preference_weight=0.2
    )

    print(f"创建了联合训练数据集:")
    print(f"- 布局样本数: {len(layout_data)}")
    print(f"- 偏好样本数: {len(preference_dataset.preferences)}")
    print(f"- 偏好学习权重: {joint_dataset.preference_weight}")

    # 创建训练配置
    config = TrainingConfig.get_experiment_config('rl_medium')
    print(f"使用训练配置: {config['guidance_config']}")

    return joint_dataset, config


def example_5_complete_pipeline():
    """示例5: 完整训练和推理流程"""
    print("\n🏭 示例5: 完整训练和推理流程")
    print("=" * 50)

    print("步骤1: 数据准备")
    layout_data = create_sample_layout_data()
    print(f"✓ 准备了 {len(layout_data)} 个布局样本")

    print("\n步骤2: RL环境和智能体设置")
    env = LayoutRLEnvironment(max_boxes=10)
    sac_agent = SACAgent(state_dim=200, action_dim=20, hidden_dim=256)
    print("✓ 创建了RL环境和SAC智能体")

    print("\n步骤3: 偏好学习设置")
    preference_dataset = LayoutPreferenceDataset()
    collector = InteractiveDataCollector(preference_dataset)
    print("✓ 设置了偏好学习数据收集器")

    print("\n步骤4: 扩散模型集成")
    # 这里应该加载真实的RADM模型
    print("✓ 准备了扩散模型集成 (需要实际的RADM模型)")

    print("\n步骤5: 联合训练器配置")
    config = TrainingConfig.get_experiment_config('rl_medium')
    print(f"✓ 配置了联合训练器 (引导强度: {config['final_guidance_scale']})")

    print("\n🎉 完整流程配置完成！")
    print("\n接下来可以运行:")
    print("1. trainer.train(train_dataset, val_dataset)")
    print("2. 生成RL引导的布局")
    print("3. 评估生成质量")

    return {
        'layout_data': layout_data,
        'env': env,
        'sac_agent': sac_agent,
        'preference_dataset': preference_dataset,
        'config': config
    }


def run_all_examples():
    """运行所有示例"""
    print("🎯 RL增强RADM系统 - 完整示例演示")
    print("=" * 60)

    try:
        # 示例1: RL训练
        sac_agent = example_1_rl_training()

        # 示例2: 偏好学习
        preference_dataset = example_2_preference_learning()

        # 示例3: RL引导生成
        rl_guided_radm = example_3_rl_guided_generation()

        # 示例4: 联合训练
        joint_dataset, config = example_4_joint_training()

        # 示例5: 完整流程
        pipeline_components = example_5_complete_pipeline()

        print("\n🎊 所有示例运行完成！")
        print("\n📚 接下来你可以:")
        print("1. 调整超参数进行更长的训练")
        print("2. 集成真实的RADM模型")
        print("3. 添加更多类型的布局约束")
        print("4. 实现人类偏好学习的交互界面")

    except Exception as e:
        print(f"❌ 运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 运行所有示例
    run_all_examples()
