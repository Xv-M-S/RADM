
"""
Test cases for Multi-Agent Reinforcement Learning Environment for Layout Generation
"""
import torch
import numpy as np
import pytest
from unittest.mock import Mock
import sys
import os

sys.path.append(os.path.abspath("/home/sxm/flux-workspace/text-to-layout-zhuanlan/BASE-RADM/RADM"))
from RADM.rl_layout_env import LayoutElement, LayoutEnvironment


def create_mock_config():
    """创建模拟配置对象"""
    config = Mock()
    config.MODEL.DEVICE = 'cpu'
    config.MODEL.RADM.NUM_CLASSES = 5  # 5个前景类别，索引0-4
    config.RL.MAX_ELEMENTS = 20
    config.RL.ACTION_SCALE = 0.1
    config.RL.MAX_STEPS = 50
    return config


class TestLayoutElement:
    """测试LayoutElement类"""

    def test_initialization(self):
        """测试初始化"""
        bbox = torch.tensor([0.1, 0.2, 0.3, 0.4])
        text_feature = torch.randn(512)
        
        element = LayoutElement(element_id=0, bbox=bbox, class_id=1, text_feature=text_feature)
        
        assert element.id == 0
        assert torch.equal(element.bbox, bbox)
        assert element.class_id == 1
        assert torch.equal(element.text_feature, text_feature)
        assert element.neighbors == []
        assert element.reward_history == []

    def test_update_bbox(self):
        """测试边界框更新"""
        bbox = torch.tensor([0.1, 0.2, 0.3, 0.4])
        element = LayoutElement(element_id=0, bbox=bbox, class_id=1, text_feature=torch.randn(512))
        
        # 测试正常更新
        new_bbox = torch.tensor([0.2, 0.3, 0.4, 0.5])
        element.update_bbox(new_bbox)
        assert torch.equal(element.bbox, new_bbox)
        
        # 测试边界约束
        out_of_bounds_bbox = torch.tensor([-0.1, 1.2, 0.3, 0.4])
        element.update_bbox(out_of_bounds_bbox)
        expected = torch.tensor([0.0, 1.0, 0.3, 0.4])
        assert torch.allclose(element.bbox, expected)

    def test_get_state(self):
        """测试获取状态"""
        bbox = torch.tensor([0.1, 0.2, 0.3, 0.4])
        text_feature = torch.randn(512)
        element = LayoutElement(element_id=0, bbox=bbox, class_id=1, text_feature=text_feature)
        element.neighbors = [1, 2]
        
        state = element.get_state()
        
        assert torch.equal(state['bbox'], bbox)
        assert state['class_id'] == 1
        assert torch.equal(state['text_feature'], text_feature)
        assert state['neighbors'] == [1, 2]


class TestLayoutEnvironment:
    """测试LayoutEnvironment类"""

    def setup_method(self):
        """设置测试环境"""
        self.config = create_mock_config()
        self.env = LayoutEnvironment(self.config)

    def test_initialization(self):
        """测试环境初始化"""
        assert self.env.device == torch.device('cpu')
        assert self.env.max_elements == 20
        assert self.env.action_dim == 4
        assert self.env.action_scale == 0.1
        assert self.env.max_steps == 50
        assert len(self.env.reward_weights) == 5

    def test_reset(self):
        """测试环境重置"""
        # 创建模拟的RADM输出，确保有有效的前景类
        num_proposals = 5
        num_classes = 6  # 5个前景类 + 1个背景类
        
        # 创建预测框，使用随机值
        pred_boxes = torch.randn(1, num_proposals, 4)
        
        # 创建预测logits，确保一些预测为前景类（0-4），一些为背景类（5）
        pred_logits = torch.randn(1, num_proposals, num_classes)
        
        # 手动设置一些类别，确保有前景类
        pred_logits[0, :, :5] = 10  # 前5类设为高分
        pred_logits[0, :, 5] = -10  # 背景类设为低分
        
        radm_output = {
            'pred_boxes': pred_boxes,
            'pred_logits': pred_logits
        }
        
        text_feature = torch.randn(512)
        image_size = (480, 640)
        
        observations = self.env.reset(radm_output, text_feature, image_size)
        
        # 计算预期的有效元素数量
        classes = torch.argmax(pred_logits[0], dim=-1)  # [num_proposals]
        valid_mask = classes < self.config.MODEL.RADM.NUM_CLASSES
        expected_num_elements = valid_mask.sum().item()
        
        # 检查环境状态
        assert len(self.env.elements) == expected_num_elements
        assert self.env.global_text_feature is not None
        assert self.env.image_size == image_size
        assert self.env.step_count == 0
        
        # 检查观测格式
        assert isinstance(observations, dict)
        for agent_id, obs in observations.items():
            assert 'bbox' in obs
            assert 'class_id' in obs
            assert 'text_feature' in obs
            assert 'neighbor_states' in obs

    def test_reset_with_background_filtering(self):
        """测试重置时的背景过滤"""
        # 创建所有都是背景类的输出
        num_proposals = 3
        num_classes = 6  # 5个前景类 + 1个背景类
        
        pred_boxes = torch.randn(1, num_proposals, 4)
        
        # 创建所有预测都是背景类的logits
        pred_logits = torch.randn(1, num_proposals, num_classes)
        pred_logits[0, :, :5] = -10  # 前5类设为低分
        pred_logits[0, :, 5] = 10   # 背景类设为高分
        
        radm_output = {
            'pred_boxes': pred_boxes,
            'pred_logits': pred_logits
        }
        
        text_feature = torch.randn(512)
        image_size = (480, 640)
        
        observations = self.env.reset(radm_output, text_feature, image_size)
        
        # 所有元素都应该是背景类，会被过滤掉
        assert len(self.env.elements) == 0

    def test_step_basic(self):
        """测试基本步骤功能"""
        # 先重置环境，创建一些元素
        num_proposals = 3
        num_classes = 6
        
        pred_boxes = torch.tensor([[[0.1, 0.1, 0.2, 0.2], 
                                   [0.4, 0.4, 0.2, 0.2], 
                                   [0.7, 0.7, 0.2, 0.2]]])
        
        pred_logits = torch.randn(1, num_proposals, num_classes)
        pred_logits[0, :, :5] = 10  # 确保都是前景类
        pred_logits[0, :, 5] = -10
        
        radm_output = {
            'pred_boxes': pred_boxes,
            'pred_logits': pred_logits
        }
        
        text_feature = torch.randn(512)
        image_size = (480, 640)
        self.env.reset(radm_output, text_feature, image_size)
        
        # 验证元素被正确创建
        assert len(self.env.elements) == 3
        
        # 创建动作字典
        actions = {
            0: torch.tensor([0.1, 0.1, 0.1, 0.1]),
            1: torch.tensor([-0.1, -0.1, -0.1, -0.1]),
            2: torch.tensor([0.0, 0.0, 0.0, 0.0])
        }
        
        observations, rewards, done, info = self.env.step(actions)
        
        # 检查返回值
        assert isinstance(observations, dict)
        assert isinstance(rewards, dict)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        # 检查步数增加
        assert self.env.step_count == 1
        
        # 检查奖励计算
        assert len(rewards) == 3
        for agent_id, reward in rewards.items():
            assert isinstance(reward.item(), (int, float))

    def test_execute_action(self):
        """测试动作执行"""
        # 创建单个元素
        bbox = torch.tensor([0.5, 0.5, 0.2, 0.2])
        element = LayoutElement(element_id=0, bbox=bbox, class_id=1, text_feature=torch.randn(512))
        self.env.elements = {0: element}
        
        # 执行动作
        action = torch.tensor([0.1, 0.1, 0.1, 0.1])
        self.env._execute_action(0, action)
        
        # 检查边界框是否更新
        expected_x = 0.5 + 0.1 * 0.1  # 0.5 + 0.01 = 0.51
        expected_y = 0.5 + 0.1 * 0.1  # 0.5 + 0.01 = 0.51
        expected_w = 0.2 * (1 + 0.1 * 0.1)  # 0.2 * 1.01 = 0.202
        expected_h = 0.2 * (1 + 0.1 * 0.1)  # 0.2 * 1.01 = 0.202
        
        updated_bbox = self.env.elements[0].bbox
        assert torch.allclose(updated_bbox, torch.tensor([expected_x, expected_y, expected_w, expected_h]))

    def test_calculate_overlap_penalty(self):
        """测试重叠惩罚计算"""
        # 创建两个重叠的元素
        bbox1 = torch.tensor([0.1, 0.1, 0.3, 0.3])  # [x, y, w, h]
        bbox2 = torch.tensor([0.2, 0.2, 0.3, 0.3])
        
        element1 = LayoutElement(0, bbox1, 0, torch.randn(512))
        element2 = LayoutElement(1, bbox2, 0, torch.randn(512))
        self.env.elements = {0: element1, 1: element2}
        
        # 计算重叠惩罚
        penalty1 = self.env._calculate_overlap_penalty(0)
        penalty2 = self.env._calculate_overlap_penalty(1)
        
        # 两个元素的重叠惩罚应该相同
        assert penalty1 == penalty2
        assert penalty1 > 0  # 应该有重叠

    def test_calculate_alignment_bonus(self):
        """测试对齐奖励计算"""
        # 测试边缘对齐
        bbox_edge = torch.tensor([0.0, 0.0, 0.2, 0.3])  # 左上角
        element_edge = LayoutElement(0, bbox_edge, 0, torch.randn(512))
        self.env.elements = {0: element_edge}
        
        bonus = self.env._calculate_alignment_bonus(0)
        assert bonus >= 0.5  # 边缘对齐应该有奖励

    def test_calculate_balance_bonus(self):
        """测试平衡奖励计算"""
        # 测试中心对齐
        bbox_center = torch.tensor([0.4, 0.4, 0.2, 0.2])  # 中心附近
        element_center = LayoutElement(0, bbox_center, 0, torch.randn(512))
        self.env.elements = {0: element_center}
        
        balance_score = self.env._calculate_balance_bonus()
        assert 0 <= balance_score <= 1  # 平衡分数应该在0-1之间

    def test_calculate_semantic_coherence(self):
        """测试语义连贯性计算"""
        # 创建两个相同类别的元素
        text_feature = torch.randn(512)
        bbox1 = torch.tensor([0.1, 0.1, 0.2, 0.2])
        bbox2 = torch.tensor([0.4, 0.4, 0.2, 0.2])
        
        element1 = LayoutElement(0, bbox1, 1, text_feature)
        element2 = LayoutElement(1, bbox2, 1, text_feature)
        element1.neighbors = [1]
        element2.neighbors = [0]
        self.env.elements = {0: element1, 1: element2}
        
        coherence = self.env._calculate_semantic_coherence(0)
        assert coherence >= 0.5  # 相同类别应该有较高的连贯性

    def test_calculate_aesthetic_score(self):
        """测试美学评分计算"""
        # 测试黄金比例
        bbox_golden = torch.tensor([0.2, 0.2, 0.324, 0.2])  # 接近黄金比例
        element_golden = LayoutElement(0, bbox_golden, 0, torch.randn(512))
        self.env.elements = {0: element_golden}
        
        aesthetic_score = self.env._calculate_aesthetic_score(0)
        assert 0 <= aesthetic_score <= 1  # 美学分数应该在0-1之间

    def test_get_layout_boxes_and_classes(self):
        """测试获取布局边界框和类别"""
        # 创建几个元素
        bbox1 = torch.tensor([0.1, 0.1, 0.2, 0.2])
        bbox2 = torch.tensor([0.4, 0.4, 0.3, 0.3])
        
        element1 = LayoutElement(0, bbox1, 1, torch.randn(512))
        element2 = LayoutElement(1, bbox2, 2, torch.randn(512))
        self.env.elements = {0: element1, 1: element2}
        
        boxes = self.env.get_layout_boxes()
        classes = self.env.get_layout_classes()
        
        assert boxes.shape == (2, 4)
        assert classes.shape == (2,)
        assert torch.equal(boxes[0], bbox1)
        assert torch.equal(boxes[1], bbox2)
        assert classes[0] == 1
        assert classes[1] == 2

    def test_termination_condition(self):
        """测试终止条件"""
        # 重置环境
        num_proposals = 2
        num_classes = 6
        
        pred_boxes = torch.tensor([[[0.1, 0.1, 0.2, 0.2], 
                                   [0.4, 0.4, 0.2, 0.2]]])
        pred_logits = torch.randn(1, num_proposals, num_classes)
        pred_logits[0, :, :5] = 10  # 确保都是前景类
        pred_logits[0, :, 5] = -10
        
        radm_output = {
            'pred_boxes': pred_boxes,
            'pred_logits': pred_logits
        }
        
        text_feature = torch.randn(512)
        image_size = (480, 640)
        self.env.reset(radm_output, text_feature, image_size)
        
        # 设置最大步数为2
        self.env.max_steps = 2
        
        # 执行2步
        actions = {0: torch.zeros(4), 1: torch.zeros(4)}
        _, _, done, _ = self.env.step(actions)  # 第1步
        assert not done
        
        _, _, done, _ = self.env.step(actions)  # 第2步
        assert done

    def test_empty_environment(self):
        """测试空环境的边界情况"""
        # 测试空环境的平衡计算
        # 创建一个新环境，确保是完全干净的
        clean_env = LayoutEnvironment(create_mock_config())
        
        # 测试空环境的平衡计算
        balance_score = clean_env._calculate_balance_bonus()
        assert balance_score == 0.0
        
        # 测试空环境的布局质量
        quality = clean_env._calculate_layout_quality()
        assert quality['r_shm'] == 0.0
        assert quality['balance'] == 0.0
        assert quality['alignment'] == 0.0


def test_box_iou_calculation():
    """测试边界框IOU计算"""
    env = LayoutEnvironment(create_mock_config())
    
    # 测试完全重叠
    box1 = torch.tensor([0.0, 0.0, 1.0, 1.0])
    box2 = torch.tensor([0.0, 0.0, 1.0, 1.0])
    iou = env._calculate_box_iou(box1, box2)
    assert iou == 1.0
    
    # 测试无重叠
    box1 = torch.tensor([0.0, 0.0, 0.5, 0.5])
    box2 = torch.tensor([0.6, 0.6, 0.3, 0.3])
    iou = env._calculate_box_iou(box1, box2)
    assert iou == 0.0
    
    # 测试部分重叠
    box1 = torch.tensor([0.0, 0.0, 0.5, 0.5])
    box2 = torch.tensor([0.25, 0.25, 0.5, 0.5])
    iou = env._calculate_box_iou(box1, box2)
    assert 0.0 < iou < 1.0


def test_neighbor_update():
    """测试邻居关系更新"""
    env = LayoutEnvironment(create_mock_config())
    
    # 创建几个元素
    bbox1 = torch.tensor([0.1, 0.1, 0.1, 0.1])
    bbox2 = torch.tensor([0.2, 0.2, 0.1, 0.1])
    bbox3 = torch.tensor([0.8, 0.8, 0.1, 0.1])
    
    element1 = LayoutElement(0, bbox1, 0, torch.randn(512))
    element2 = LayoutElement(1, bbox2, 0, torch.randn(512))
    element3 = LayoutElement(2, bbox3, 0, torch.randn(512))
    env.elements = {0: element1, 1: element2, 2: element3}
    
    env._update_neighbor_relationships()
    
    # 检查邻居关系
    assert 1 in env.elements[0].neighbors  # 元素0应该有元素1作为邻居
    assert 0 in env.elements[1].neighbors  # 元素1应该有元素0作为邻居
    # 元素2应该远离其他元素，可能没有邻居或只有很远的邻居


def test_empty_elements_update_neighbors():
    """测试空元素列表时的邻居更新"""
    env = LayoutEnvironment(create_mock_config())
    env.elements = {}  # 空元素字典
    
    # 这个调用应该不会出错
    env._update_neighbor_relationships()
    
    # 验证环境状态仍然有效
    assert len(env.elements) == 0


def run_comprehensive_test():
    """运行综合测试"""
    print("Running comprehensive tests for Layout Environment...")
    
    # 创建测试环境
    config = create_mock_config()
    env = LayoutEnvironment(config)
    
    # 测试完整的工作流程
    pred_boxes = torch.tensor([[[0.1, 0.1, 0.2, 0.2], 
                               [0.4, 0.4, 0.2, 0.2], 
                               [0.7, 0.7, 0.2, 0.2]]])
    pred_logits = torch.randn(1, 3, 6)
    pred_logits[0, :, :5] = 10  # 确保都是前景类
    pred_logits[0, :, 5] = -10
    
    radm_output = {
        'pred_boxes': pred_boxes,
        'pred_logits': pred_logits
    }
    text_feature = torch.randn(512)
    image_size = (480, 640)
    
    # 重置环境
    obs = env.reset(radm_output, text_feature, image_size)
    print(f"Reset successful. Number of elements: {len(env.elements)}")
    
    # 执行几个步骤
    for step in range(5):
        actions = {i: torch.randn(4) * 0.1 for i in range(3)}
        obs, rewards, done, info = env.step(actions)
        print(f"Step {step + 1}: Total reward = {sum(rewards.values()):.3f}")
        
        if done:
            break
    
    # 检查最终状态
    final_boxes = env.get_layout_boxes()
    final_classes = env.get_layout_classes()
    quality = env._calculate_layout_quality()
    
    print(f"Final layout quality: {quality}")
    print(f"Final boxes shape: {final_boxes.shape}")
    print(f"Final classes shape: {final_classes.shape}")
    
    print("Comprehensive test completed successfully!")


if __name__ == "__main__":
    # 运行所有测试
    print("Running Layout Environment Tests...")
    
    # 创建测试实例
    test_element = TestLayoutElement()
    test_env = TestLayoutEnvironment()
    
    # 运行测试方法
    test_element.test_initialization()
    print("✓ LayoutElement initialization test passed")
    
    test_element.test_update_bbox()
    print("✓ LayoutElement update_bbox test passed")
    
    test_element.test_get_state()
    print("✓ LayoutElement get_state test passed")
    
    test_env.setup_method()
    test_env.test_initialization()
    print("✓ LayoutEnvironment initialization test passed")
    
    test_env.test_reset()
    print("✓ LayoutEnvironment reset test passed")
    
    test_env.test_reset_with_background_filtering()
    print("✓ LayoutEnvironment reset with background filtering test passed")
    
    test_env.test_step_basic()
    print("✓ LayoutEnvironment step test passed")
    
    test_env.test_execute_action()
    print("✓ Execute action test passed")
    
    test_env.test_calculate_overlap_penalty()
    print("✓ Overlap penalty test passed")
    
    test_env.test_calculate_alignment_bonus()
    print("✓ Alignment bonus test passed")
    
    test_env.test_calculate_balance_bonus()
    print("✓ Balance bonus test passed")
    
    test_env.test_calculate_semantic_coherence()
    print("✓ Semantic coherence test passed")
    
    test_env.test_calculate_aesthetic_score()
    print("✓ Aesthetic score test passed")
    
    test_env.test_get_layout_boxes_and_classes()
    print("✓ Layout boxes and classes test passed")
    
    test_env.test_termination_condition()
    print("✓ Termination condition test passed")
    
    test_env.test_empty_environment()
    print("✓ Empty environment test passed")
    
    test_box_iou_calculation()
    print("✓ Box IOU calculation test passed")
    
    test_neighbor_update()
    print("✓ Neighbor update test passed")
    
    test_empty_elements_update_neighbors()
    print("✓ Empty elements neighbor update test passed")
    
    run_comprehensive_test()
    
    print("\n🎉 All tests passed! Layout Environment is working correctly.")



