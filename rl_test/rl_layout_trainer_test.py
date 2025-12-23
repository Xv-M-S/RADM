import pytest
import torch
import os
import tempfile
import json
from pathlib import Path

# 导入被测试模块
from RADM.rl_trainer import RLLayoutTrainer, RLLayoutInference
from RADM.config import get_config

@pytest.fixture
def temp_output_dir():
    """创建临时输出目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def minimal_config(temp_output_dir):
    """创建最小配置"""
    config = get_config()
    config.OUTPUT_DIR = temp_output_dir
    config.RL.NUM_ENV_STEPS = 1000  # 减少训练步数
    config.RL.NUM_STEPS = 32
    config.RL.LOG_INTERVAL = 1
    config.RL.SAVE_INTERVAL = 10
    config.MODEL.DEVICE = 'cpu' if not torch.cuda.is_available() else 'cuda'
    return config

def test_training_loop(minimal_config):
    """测试训练循环能正常运行"""
    trainer = RLLayoutTrainer(minimal_config)
    
    # 创建极小数据集
    from test_utils import create_dummy_dataloader
    data_loader = create_dummy_dataloader(batch_size=1, num_samples=2)
    
    # 运行少量训练步骤
    trainer.train(data_loader)
    
    # 验证模型文件已保存
    model_dir = os.path.join(minimal_config.OUTPUT_DIR, 'rl_training', 'models')
    assert os.path.exists(model_dir)
    assert len(os.listdir(model_dir)) > 0

def test_inference_pipeline(minimal_config):
    """测试推理流程"""
    # 首先训练一个小型模型
    trainer = RLLayoutTrainer(minimal_config)
    data_loader = create_dummy_dataloader(batch_size=1, num_samples=2)
    trainer.train(data_loader)
    
    # 获取最新模型
    model_dir = Path(minimal_config.OUTPUT_DIR) / 'rl_training' / 'models'
    latest_model = sorted(model_dir.glob('*.pth'))[-1]
    
    # 初始化推理器
    inference = RLLayoutInference(minimal_config, str(latest_model))
    
    # 创建测试批次
    test_batch = next(iter(data_loader))
    
    # 运行推理
    result = inference(test_batch)
    
    # 验证结果
    assert 'final_boxes' in result
    assert 'final_classes' in result
    assert result['final_boxes'].shape[0] > 0  # 应有至少一个框
    assert torch.all(result['final_boxes'] >= 0)  # 坐标应为非负
    assert torch.all(result['final_boxes'] <= 1)  # 归一化坐标应在[0,1]范围内

if __name__ == "__main__":
    pytest.main([__file__, "-v"])