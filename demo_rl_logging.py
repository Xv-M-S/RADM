#!/usr/bin/env python3
"""
Demo script showing RL training logging capabilities
"""
import logging
import sys
from detectron2.config import get_cfg

from RADM import add_radm_config, add_rl_config
from RADM.rl_trainer import RLLayoutTrainer


def setup_logging(log_level='INFO'):
    """Setup logging configuration"""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level))

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger


def demo_config_loading():
    """Demo configuration loading and validation"""
    print("=" * 60)
    print("DEMO: RL Configuration Loading")
    print("=" * 60)

    # Load config
    cfg = get_cfg()
    add_radm_config(cfg)
    add_rl_config(cfg)
    cfg.merge_from_file('configs/radm.yaml')

    print("✓ Configuration loaded successfully")
    print(f"  RL Steps: {cfg.RL.NUM_STEPS}")
    print(f"  Log Level: {cfg.RL.LOG_LEVEL}")
    print(f"  Reward Weights: {cfg.RL.OVERLAP_PENALTY}, {cfg.RL.ALIGNMENT_BONUS}, {cfg.RL.BALANCE_BONUS}")
    print()


def demo_trainer_initialization():
    """Demo trainer initialization with different log levels"""
    print("=" * 60)
    print("DEMO: Trainer Initialization with Different Log Levels")
    print("=" * 60)

    for log_level in ['WARNING', 'INFO', 'DEBUG']:
        print(f"\n--- Testing {log_level} Log Level ---")

        # Setup logging
        logger = setup_logging(log_level)

        # Create config
        cfg = get_cfg()
        add_radm_config(cfg)
        add_rl_config(cfg)
        cfg.merge_from_file('configs/radm.yaml')
        cfg.RL.LOG_LEVEL = log_level

        try:
            # Initialize trainer (without actual training data)
            trainer = RLLayoutTrainer(cfg)
            print(f"✓ Trainer initialized successfully with {log_level} logging")
            print("  This would show detailed initialization logs in DEBUG mode")
        except Exception as e:
            print(f"✗ Failed to initialize trainer: {e}")

    print()


def demo_log_levels():
    """Demo different logging levels and their outputs"""
    print("=" * 60)
    print("DEMO: Logging Level Examples")
    print("=" * 60)

    # Test different log levels
    test_logger = logging.getLogger('test_logger')

    print("\n1. DEBUG Level (shows everything):")
    test_logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('    %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    test_logger.addHandler(handler)

    test_logger.debug("This is a DEBUG message - very detailed info")
    test_logger.info("This is an INFO message - standard progress info")
    test_logger.warning("This is a WARNING message - potential issues")

    print("\n2. INFO Level (standard training logs):")
    handler.setLevel(logging.INFO)
    test_logger.info("Training step completed")
    test_logger.debug("This debug message won't show")

    print("\n3. WARNING Level (only important messages):")
    handler.setLevel(logging.WARNING)
    test_logger.warning("This warning will show")
    test_logger.info("This info won't show")
    test_logger.debug("This debug won't show")

    print()


def demo_training_output_format():
    """Demo the training output format"""
    print("=" * 60)
    print("DEMO: Training Output Format")
    print("=" * 60)

    print("\nTypical training log output:")
    print("  [15.2%] Step 3072/20000 | Value Loss: 0.0124 | Action Loss: -0.0345 | Entropy: 0.0234 | Rewards: 4.231±1.234 (max: 6.789) | Episode Length: 45.6 (max: 50) | r_shm: 0.8345 | Speed: 45.2 steps/sec | ETA: 12.3 min")

    print("\nField explanations:")
    print("  [15.2%]         - Training progress percentage")
    print("  Step 3072/20000 - Current step / total steps")
    print("  Value Loss      - PPO value function loss")
    print("  Action Loss     - PPO policy loss")
    print("  Entropy         - Action distribution randomness")
    print("  Rewards         - Mean ± std (max reward)")
    print("  Episode Length  - Average episode length (max)")
    print("  Quality Metrics - Layout quality scores")
    print("  Speed & ETA     - Training speed and estimated completion time")

    print("\nPeriodic detailed reports (every 10 log intervals):")
    print("  ========================================================")
    print("  DETAILED TRAINING STATISTICS:")
    print("  Total Steps: 3072")
    print("  Training Progress: 15.23%")
    print("    value_loss: 0.012432")
    print("    action_loss: -0.034567")
    print("    dist_entropy: 0.023456")
    print("    Episode count: 8")
    print("    Reward distribution: min=-1.234, median=4.321, max=6.789")
    print("  ========================================================")

    print()


def main():
    """Run all demos"""
    print("RL-RADM Training Logging Demonstration")
    print("This script shows the logging capabilities added to the RL training system\n")

    demo_config_loading()
    demo_trainer_initialization()
    demo_log_levels()
    demo_training_output_format()

    print("=" * 60)
    print("🎉 DEMO COMPLETED!")
    print("=" * 60)
    print("\nTo use these logging features in actual training:")
    print("1. Set LOG_LEVEL in configs/radm.yaml (DEBUG/INFO/WARNING)")
    print("2. Run: python train_rl.py --config-file configs/radm.yaml --device cuda")
    print("3. Monitor the detailed training progress and statistics")


if __name__ == "__main__":
    main()
