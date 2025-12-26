#!/usr/bin/env python3
"""
DDPO组件测试脚本
"""
import torch
import torch.nn as nn
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

# 简单的测试，不依赖detectron2
def test_preference_model():
    """测试偏好模型"""
    from RADM.layers import PreferenceModel

    # 创建模型
    model = PreferenceModel(layout_dim=256, text_dim=768, hidden_dim=512)

    # 创建测试输入
    batch_size, num_proposals = 2, 10
    layout_features = torch.randn(batch_size, num_proposals, 256)
    text_features = torch.randn(batch_size, 768)

    # 前向传播
    scores = model(layout_features, text_features)

    print(f"PreferenceModel test passed!")
    print(f"Input shape: layout={layout_features.shape}, text={text_features.shape}")
    print(f"Output shape: {scores.shape}")
    print(f"Output range: [{scores.min():.4f}, {scores.max():.4f}]")

    return True

def test_ddpo_trainer():
    """测试DDPO训练器"""
    from RADM.layers import DDPOTrainer, PreferenceModel

    # 创建虚拟模型 (用简单的MLP代替RADM)
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 4)

        def forward(self, x):
            return self.linear(x)

    # 创建组件
    dummy_model = DummyModel()
    preference_model = PreferenceModel()
    ddpo_trainer = DDPOTrainer(dummy_model, preference_model, beta=0.1, sample_size=4)

    # 创建测试数据
    sample_size, batch_size, num_proposals = 4, 2, 10
    layout_samples = torch.randn(sample_size, batch_size, num_proposals, 4)
    text_features = torch.randn(batch_size, 768)
    rewards = torch.randn(sample_size, batch_size)

    # 计算损失
    loss = ddpo_trainer.compute_preference_loss(layout_samples, text_features, rewards)

    print(f"DDPOTrainer test passed!")
    print(f"Loss value: {loss.item():.4f}")

    return True

def main():
    print("Testing DDPO components...")

    try:
        test_preference_model()
        print()
        test_ddpo_trainer()
        print("\nAll DDPO tests passed! ✅")
        return True
    except Exception as e:
        print(f"\nDDPO test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

