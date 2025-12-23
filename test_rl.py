#!/usr/bin/env python3
"""
Test script for RL components
"""
import torch
import numpy as np
from detectron2.config import get_cfg

from RADM import add_radm_config, add_rl_config
from RADM.util.model_ema import add_model_ema_configs
from RADM.rl_layout_env import LayoutEnvironment
from RADM.rl_agent import MultiAgentLayoutPolicy


def test_environment():
    """Test layout environment"""
    print("Testing Layout Environment...")

    # Create config
    cfg = get_cfg()
    add_radm_config(cfg)
    add_rl_config(cfg)
    add_model_ema_configs(cfg)

    # Override RL config for testing
    cfg.MODEL.DEVICE = 'cpu'  # Force CPU for testing
    cfg.RL.MAX_ELEMENTS = 5
    cfg.RL.MAX_STEPS = 10
    cfg.RL.ACTION_SCALE = 0.1

    # Create environment
    env = LayoutEnvironment(cfg)

    # Create mock RADM output
    batch_size, num_proposals = 1, 5
    radm_output = {
        'pred_boxes': torch.rand(batch_size, num_proposals, 4),  # [x, y, w, h]
        'pred_logits': torch.randn(batch_size, num_proposals, cfg.MODEL.RADM.NUM_CLASSES)
    }

    text_features = torch.randn(1, 768)
    image_size = (800, 600)

    # Reset environment
    observations = env.reset(radm_output, text_features, image_size)
    print(f"Environment reset successful. Created {len(observations)} agents.")

    # Test step
    actions = {}
    for agent_id in observations.keys():
        actions[agent_id] = torch.randn(4) * 0.1  # Random action

    next_observations, rewards, done, info = env.step(actions)
    print(f"Environment step successful. Rewards: {list(rewards.values())}")

    # Test layout extraction
    final_boxes = env.get_layout_boxes()
    final_classes = env.get_layout_classes()
    print(f"Final layout: {len(final_boxes)} boxes, {len(final_classes)} classes")

    print("✓ Environment test passed!")
    return True


def test_policy():
    """Test RL policy"""
    print("Testing RL Policy...")

    # Create config
    cfg = get_cfg()
    add_radm_config(cfg)
    add_rl_config(cfg)
    add_model_ema_configs(cfg)
    cfg.MODEL.DEVICE = 'cpu'

    # Override RL config for testing
    cfg.RL.HIDDEN_DIM = 64  # Smaller for testing

    # Create policy
    policy = MultiAgentLayoutPolicy(cfg)

    # Create mock observations
    observations = {}
    for i in range(3):  # 3 agents
        obs = {
            'bbox': torch.rand(4),
            'class_id': torch.randint(0, cfg.MODEL.RADM.NUM_CLASSES, (1,)).item(),
            'text_feature': torch.randn(768),
            'global_text_feature': torch.randn(768),
            'neighbor_states': [
                {
                    'bbox': torch.rand(4),
                    'class_id': torch.randint(0, cfg.MODEL.RADM.NUM_CLASSES, (1,)).item(),
                    'text_feature': torch.randn(768)
                } for _ in range(2)  # 2 neighbors
            ],
            'image_size': (800, 600),
            'step_count': 0
        }
        observations[i] = obs

    # Test forward pass
    actions, log_probs = policy(observations)
    print(f"Policy forward pass successful. Actions shape: {len(actions)}")

    # Test action evaluation
    log_probs_eval, entropies, values = policy.evaluate_actions(observations, actions)
    print(f"Policy evaluation successful. Values shape: {len(values)}")

    print("✓ Policy test passed!")
    return True


def test_integration():
    """Test RL components integration"""
    print("Testing RL Components Integration...")

    # Create config
    cfg = get_cfg()
    add_radm_config(cfg)
    add_rl_config(cfg)
    add_model_ema_configs(cfg)
    cfg.MODEL.DEVICE = 'cpu'

    # Override RL config for testing
    cfg.RL.MAX_ELEMENTS = 3
    cfg.RL.MAX_STEPS = 5
    cfg.RL.ACTION_SCALE = 0.1
    cfg.RL.HIDDEN_DIM = 64

    # Initialize components
    env = LayoutEnvironment(cfg)
    policy = MultiAgentLayoutPolicy(cfg)

    # Create mock RADM output
    radm_output = {
        'pred_boxes': torch.rand(1, 3, 4),
        'pred_logits': torch.randn(1, 3, cfg.MODEL.RADM.NUM_CLASSES)
    }
    text_features = torch.randn(1, 768)
    image_size = (800, 600)

    # Test full episode
    observations = env.reset(radm_output, text_features, image_size)


    total_reward = 0
    for step in range(cfg.RL.MAX_STEPS):
        # Sample actions
        actions, _ = policy(observations)

        # Environment step
        next_observations, rewards, done, info = env.step(actions)

        # Accumulate reward
        step_reward = sum(rewards.values())
        total_reward += step_reward

        print(f"Step {step}: Reward = {step_reward:.4f}")

        if done:
            break

        observations = next_observations

    print(f"Episode completed. Total reward: {total_reward:.4f}")
    print(f"Final layout quality: {info.get('layout_quality', {})}")

    print("✓ Integration test passed!")
    return True


def main():
    """Run all tests"""
    print("=" * 50)
    print("RL COMPONENTS TEST SUITE")
    print("=" * 50)

    try:
        # Test individual components
        test_environment()
        print()
        test_policy()
        print()

        # Test integration
        test_integration()
        print()

        print("=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("RL components are ready for training.")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
