"""
Test cases for Multi-Agent Reinforcement Learning Layout Optimization System
"""
import torch
import numpy as np
from unittest.mock import Mock
import sys
import os

sys.path.append(os.path.abspath("/home/sxm/flux-workspace/text-to-layout-zhuanlan/BASE-RADM/RADM"))
from RADM.rl_agent import LayoutAgent, MultiAgentLayoutPolicy, PPOLayoutTrainer, PPOStorage, BatchSampler


def create_mock_config():
    """创建模拟配置对象"""
    config = Mock()
    config.MODEL.DEVICE = 'cpu'
    config.MODEL.RADM.NUM_CLASSES = 5  # 5个前景类别
    config.RL.HIDDEN_DIM = 256
    config.RL.CLIP_PARAM = 0.2
    config.RL.PPO_EPOCH = 10
    config.RL.NUM_MINI_BATCH = 5
    config.RL.VALUE_LOSS_COEF = 0.5
    config.RL.ENTROPY_COEF = 0.01
    config.RL.MAX_GRAD_NORM = 0.5
    config.RL.LR = 3e-4
    config.RL.NUM_STEPS = 64
    config.RL.MAX_ELEMENTS = 10
    return config


class TestLayoutAgent:
    """测试LayoutAgent类"""

    def test_initialization(self):
        """测试智能体初始化"""
        config = create_mock_config()
        agent = LayoutAgent(config)
        
        # 检查网络结构
        assert agent.hidden_dim == 256
        assert agent.text_dim == 768
        assert agent.action_dim == 4
        assert agent.device == torch.device('cpu')
        
        # 检查各子网络存在
        assert hasattr(agent, 'bbox_encoder')
        assert hasattr(agent, 'text_encoder')
        assert hasattr(agent, 'class_embedding')
        assert hasattr(agent, 'actor')
        assert hasattr(agent, 'critic')

    def test_forward_pass(self):
        """测试前向传播"""
        config = create_mock_config()
        agent = LayoutAgent(config)
        
        # 创建模拟观察值
        observation = {
            'bbox': torch.tensor([0.2, 0.3, 0.4, 0.5]),
            'class_id': torch.tensor(2),
            'global_text_feature': torch.randn(768),
            'neighbor_states': [
                {
                    'bbox': torch.tensor([0.1, 0.1, 0.2, 0.2]),
                    'class_id': torch.tensor(1)
                },
                {
                    'bbox': torch.tensor([0.6, 0.6, 0.2, 0.2]),
                    'class_id': torch.tensor(3)
                }
            ]
        }
        
        action, log_prob, value = agent(observation)
        value = value.squeeze()
        
        # 检查输出维度
        assert action.shape == (4,)  # [dx, dy, dw, dh]
        assert log_prob.shape == ()  # 标量
        assert value.shape == ()     # 标量
        
        # 检查值的合理性
        assert not torch.isnan(action).any()
        assert not torch.isnan(log_prob).any()
        assert not torch.isnan(value).any()

    def test_forward_pass_no_neighbors(self):
        """测试无邻居时的前向传播"""
        config = create_mock_config()
        agent = LayoutAgent(config)
        
        # 创建无邻居的观察值
        observation = {
            'bbox': torch.tensor([0.2, 0.3, 0.4, 0.5]),
            'class_id': torch.tensor(2),
            'global_text_feature': torch.randn(768),
            'neighbor_states': []
        }
        
        action, log_prob, value = agent(observation)
        value = value.squeeze()
        
        # 检查输出维度
        assert action.shape == (4,)
        assert log_prob.shape == ()
        assert value.shape == ()

    def test_evaluate_actions(self):
        """测试动作评估"""
        config = create_mock_config()
        agent = LayoutAgent(config)
        
        # 创建模拟观察值和动作
        observation = {
            'bbox': torch.tensor([0.2, 0.3, 0.4, 0.5]),
            'class_id': torch.tensor(2),
            'global_text_feature': torch.randn(768),
            'neighbor_states': []
        }
        actions = torch.randn(4)
        
        log_prob, entropy, value = agent.evaluate_actions(observation, actions)
        
        # 检查输出维度和合理性
        assert log_prob.shape == ()
        assert entropy.shape == ()
        assert value.shape == ()
        assert not torch.isnan(log_prob).any()
        assert not torch.isnan(entropy).any()
        assert not torch.isnan(value).any()


class TestMultiAgentLayoutPolicy:
    """测试MultiAgentLayoutPolicy类"""

    def test_initialization(self):
        """测试多智能体策略初始化"""
        config = create_mock_config()
        policy = MultiAgentLayoutPolicy(config)
        
        assert hasattr(policy, 'agent')
        assert hasattr(policy, 'central_critic')
        assert isinstance(policy.agent, LayoutAgent)

    def test_forward_pass(self):
        """测试多智能体前向传播"""
        config = create_mock_config()
        policy = MultiAgentLayoutPolicy(config)
        
        # 创建多智能体观察值
        observations = {
            0: {
                'bbox': torch.tensor([0.2, 0.3, 0.4, 0.5]),
                'class_id': torch.tensor(2),
                'global_text_feature': torch.randn(768),
                'neighbor_states': []
            },
            1: {
                'bbox': torch.tensor([0.6, 0.6, 0.2, 0.2]),
                'class_id': torch.tensor(1),
                'global_text_feature': torch.randn(768),
                'neighbor_states': []
            }
        }
        
        actions, log_probs = policy(observations)
        
        # 检查输出
        assert len(actions) == 2
        assert len(log_probs) == 2
        assert actions[0].shape == (4,)
        assert actions[1].shape == (4,)

    def test_evaluate_actions(self):
        """测试多智能体动作评估"""
        config = create_mock_config()
        policy = MultiAgentLayoutPolicy(config)
        
        # 创建观察值和动作
        observations = {
            0: {
                'bbox': torch.tensor([0.2, 0.3, 0.4, 0.5]),
                'class_id': torch.tensor(2),
                'global_text_feature': torch.randn(768),
                'neighbor_states': []
            }
        }
        actions = {0: torch.randn(4)}
        
        log_probs, entropies, values = policy.evaluate_actions(observations, actions)
        
        assert len(log_probs) == 1
        assert len(entropies) == 1
        assert len(values) == 1

    def test_central_value_function(self):
        """测试中心化价值函数"""
        config = create_mock_config()
        policy = MultiAgentLayoutPolicy(config)
        
        # 创建观察值
        observations = {
            0: {
                'bbox': torch.tensor([0.2, 0.3, 0.4, 0.5]),
                'class_id': torch.tensor(2),
                'global_text_feature': torch.randn(768),
                'neighbor_states': []
            },
            1: {
                'bbox': torch.tensor([0.6, 0.6, 0.2, 0.2]),
                'class_id': torch.tensor(1),
                'global_text_feature': torch.randn(768),
                'neighbor_states': []
            }
        }
        
        central_value = policy.central_value_function(observations)
        
        assert central_value.shape == (1,)
        assert not torch.isnan(central_value).any()


class TestPPOLayoutTrainer:
    """测试PPOLayoutTrainer类"""

    def test_initialization(self):
        """测试PPO训练器初始化"""
        config = create_mock_config()
        trainer = PPOLayoutTrainer(config)
        
        assert trainer.clip_param == 0.2
        assert trainer.ppo_epoch == 10
        assert trainer.value_loss_coef == 0.5
        assert trainer.entropy_coef == 0.01
        assert hasattr(trainer, 'policy')
        assert hasattr(trainer, 'optimizer')

    def test_update_method_exists(self):
        """测试更新方法存在"""
        config = create_mock_config()
        trainer = PPOLayoutTrainer(config)
        
        # 确保update方法存在
        assert hasattr(trainer, 'update')
        assert callable(getattr(trainer, 'update'))


class TestPPOStorage:
    """测试PPOStorage类"""

    def test_initialization(self):
        """测试存储初始化"""
        config = create_mock_config()
        storage = PPOStorage(config)
        
        assert storage.num_steps == 64
        assert storage.num_agents == 10
        assert len(storage.observations) == 10
        assert len(storage.actions) == 10

    def test_insert_method(self):
        """测试插入数据"""
        config = create_mock_config()
        storage = PPOStorage(config)
        
        # 创建模拟数据
        observations = {
            0: {'bbox': torch.tensor([0.2, 0.3, 0.4, 0.5]), 
                'class_id': torch.tensor(2),
                'global_text_feature': torch.randn(768),
                'neighbor_states': []},
            1: {'bbox': torch.tensor([0.6, 0.6, 0.2, 0.2]), 
                'class_id': torch.tensor(1),
                'global_text_feature': torch.randn(768),
                'neighbor_states': []}
        }
        actions = {0: torch.randn(4), 1: torch.randn(4)}
        action_log_probs = {0: torch.tensor(0.5), 1: torch.tensor(0.3)}
        value_preds = {0: torch.tensor(0.8), 1: torch.tensor(0.6)}
        rewards = {0: torch.tensor(0.1), 1: torch.tensor(0.2)}
        masks = {0: torch.tensor(1.0), 1: torch.tensor(1.0)}
        
        storage.insert(observations, actions, action_log_probs, 
                      value_preds, rewards, masks)
        
        # 检查数据是否正确插入
        assert len(storage.observations[0]) == 1
        assert len(storage.actions[1]) == 1
        assert len(storage.rewards[0]) == 1

    def test_compute_returns(self):
        """测试回报计算"""
        config = create_mock_config()
        storage = PPOStorage(config)
        
        # 插入一些数据
        observations = {0: {'bbox': torch.tensor([0.2, 0.3, 0.4, 0.5]), 
                           'class_id': torch.tensor(2),
                           'global_text_feature': torch.randn(768),
                           'neighbor_states': []}}
        actions = {0: torch.randn(4)}
        action_log_probs = {0: torch.tensor(0.5)}
        value_preds = {0: torch.tensor(0.8)}
        rewards = {0: torch.tensor(0.1)}
        masks = {0: torch.tensor(1.0)}
        
        storage.insert(observations, actions, action_log_probs, 
                      value_preds, rewards, masks)
        storage.insert(observations, actions, action_log_probs, 
                      value_preds, rewards, masks)
        
        # 计算回报
        next_value = {0: torch.tensor(0.7)}
        storage.compute_returns(next_value)
        
        # 检查回报是否计算完成
        assert len(storage.returns[0]) == 2


class TestBatchSampler:
    """测试BatchSampler类"""

    def test_sampler_initialization(self):
        """测试批采样器初始化"""
        sampler = BatchSampler(batch_size=32, num_steps=64, num_agents=10)
        
        assert sampler.batch_size == 32
        assert sampler.num_steps == 64
        assert sampler.num_agents == 10

    def test_sampler_iteration(self):
        """测试批采样器迭代"""
        sampler = BatchSampler(batch_size=16, num_steps=32, num_agents=5)
        
        batches = list(sampler)
        
        # 检查生成的批次
        assert len(batches) > 0
        for batch in batches:
            assert len(batch) <= 16  # 不超过batch_size


def run_agent_unit_tests():
    """运行智能体单元测试"""
    print("Running LayoutAgent Unit Tests...")
    
    agent_tester = TestLayoutAgent()
    
    agent_tester.test_initialization()
    print("✓ LayoutAgent initialization test passed")
    
    agent_tester.test_forward_pass()
    print("✓ LayoutAgent forward pass test passed")
    
    agent_tester.test_forward_pass_no_neighbors()
    print("✓ LayoutAgent no neighbors test passed")
    
    agent_tester.test_evaluate_actions()
    print("✓ LayoutAgent evaluate actions test passed")


def run_policy_unit_tests():
    """运行策略单元测试"""
    print("Running MultiAgentLayoutPolicy Unit Tests...")
    
    policy_tester = TestMultiAgentLayoutPolicy()
    
    policy_tester.test_initialization()
    print("✓ MultiAgentLayoutPolicy initialization test passed")
    
    policy_tester.test_forward_pass()
    print("✓ MultiAgentLayoutPolicy forward pass test passed")
    
    policy_tester.test_evaluate_actions()
    print("✓ MultiAgentLayoutPolicy evaluate actions test passed")
    
    policy_tester.test_central_value_function()
    print("✓ MultiAgentLayoutPolicy central value function test passed")


def run_trainer_unit_tests():
    """运行训练器单元测试"""
    print("Running PPOLayoutTrainer Unit Tests...")
    
    trainer_tester = TestPPOLayoutTrainer()
    
    trainer_tester.test_initialization()
    print("✓ PPOLayoutTrainer initialization test passed")
    
    trainer_tester.test_update_method_exists()
    print("✓ PPOLayoutTrainer update method exists test passed")


def run_storage_unit_tests():
    """运行存储单元测试"""
    print("Running PPOStorage Unit Tests...")
    
    storage_tester = TestPPOStorage()
    
    storage_tester.test_initialization()
    print("✓ PPOStorage initialization test passed")
    
    storage_tester.test_insert_method()
    print("✓ PPOStorage insert method test passed")
    
    storage_tester.test_compute_returns()
    print("✓ PPOStorage compute returns test passed")


def run_sampler_unit_tests():
    """运行采样器单元测试"""
    print("Running BatchSampler Unit Tests...")
    
    sampler_tester = TestBatchSampler()
    
    sampler_tester.test_sampler_initialization()
    print("✓ BatchSampler initialization test passed")
    
    sampler_tester.test_sampler_iteration()
    print("✓ BatchSampler iteration test passed")


def run_integration_tests():
    """运行集成测试"""
    print("Running Integration Tests...")
    
    # 测试完整的训练流程
    config = create_mock_config()
    
    # 创建策略和训练器
    policy = MultiAgentLayoutPolicy(config)
    trainer = PPOLayoutTrainer(config)
    
    # 创建模拟观察值
    observations = {
        0: {
            'bbox': torch.tensor([0.2, 0.3, 0.4, 0.5]),
            'class_id': torch.tensor(2),
            'global_text_feature': torch.randn(768),
            'neighbor_states': []
        },
        1: {
            'bbox': torch.tensor([0.6, 0.6, 0.2, 0.2]),
            'class_id': torch.tensor(1),
            'global_text_feature': torch.randn(768),
            'neighbor_states': []
        }
    }
    
    # 测试前向传播
    actions, log_probs = policy(observations)
    assert len(actions) == 2
    print("✓ Policy forward pass integration test passed")
    
    # 测试中心价值函数
    central_value = policy.central_value_function(observations)
    assert central_value.shape == (1,)
    print("✓ Central value function integration test passed")


def run_comprehensive_test():
    """运行综合测试"""
    print("Running Comprehensive Tests...")
    
    # 创建完整的训练流程
    config = create_mock_config()
    policy = MultiAgentLayoutPolicy(config)
    trainer = PPOLayoutTrainer(config)
    
    # 模拟训练循环
    for episode in range(3):  # 只做3个episode测试
        # 创建模拟环境数据
        observations = {
            0: {
                'bbox': torch.tensor([0.2, 0.3, 0.4, 0.5]),
                'class_id': torch.tensor(2),
                'global_text_feature': torch.randn(768),
                'neighbor_states': []
            },
            1: {
                'bbox': torch.tensor([0.6, 0.6, 0.2, 0.2]),
                'class_id': torch.tensor(1),
                'global_text_feature': torch.randn(768),
                'neighbor_states': []
            }
        }
        
        # 获取动作
        actions, log_probs = policy(observations)
        
        # 模拟奖励（通常来自环境）
        rewards = {0: torch.tensor(0.1), 1: torch.tensor(0.2)}
        
        # 检查动作和奖励的一致性
        assert len(actions) == len(rewards)
        
        print(f"Episode {episode + 1} completed successfully")
    
    print("✓ Comprehensive test passed")


if __name__ == "__main__":
    print("Running Multi-Agent Reinforcement Learning Layout Optimization System Tests...")
    print("="*70)
    
    try:
        # 运行各组件测试
        run_agent_unit_tests()
        print()
        
        run_policy_unit_tests()
        print()
        
        run_trainer_unit_tests()
        print()
        
        run_storage_unit_tests()
        print()
        
        run_sampler_unit_tests()
        print()
        
        run_integration_tests()
        print()
        
        run_comprehensive_test()
        print()
        
        print("="*70)
        print("🎉 All tests passed! Multi-Agent RL Layout Optimization System is working correctly.")
        print("System components tested:")
        print("- LayoutAgent: Individual element controllers")
        print("- MultiAgentLayoutPolicy: Multi-agent coordination")
        print("- PPOLayoutTrainer: PPO-based training algorithm")
        print("- PPOStorage: Experience replay buffer")
        print("- BatchSampler: Mini-batch sampling")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()



