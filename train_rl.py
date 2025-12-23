#!/usr/bin/env python3
"""
RL Training Script for Layout Optimization
"""
import torch
import argparse
import os
import logging
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, MetadataCatalog

from RADM import add_radm_config,add_rl_config,RADMDatasetMapper
from RADM.util.model_ema import add_model_ema_configs
from RADM.rl_trainer import RLLayoutTrainer
from detectron2.data import build_detection_train_loader
from detectron2.data.datasets.coco import load_coco_json


def register_layout(cfg):
    """Register layout dataset"""
    DATASET_ROOT = cfg.DATASETS.DATASET_PATH
    ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
    # TRAIN_JSON = os.path.join(ANN_ROOT, 'train.json')
    TRAIN_JSON = os.path.join(ANN_ROOT, 'test.json')
    VAL_JSON = os.path.join(ANN_ROOT, 'test.json')

    IMAGE_ROOT = os.path.join(DATASET_ROOT, 'images')
    # TRAIN_PATH = os.path.join(IMAGE_ROOT, 'train')
    TRAIN_PATH = os.path.join(IMAGE_ROOT, 'test')
    VAL_PATH = os.path.join(IMAGE_ROOT, 'test')

    element_category = ["Logo", "文字", "衬底", "符号元素", "强调突出子部分文字"]

    DatasetCatalog.register("layout_train", lambda: load_coco_json(TRAIN_JSON, image_root=TRAIN_PATH, dataset_name="layout_train"))
    MetadataCatalog.get("layout_train").set(thing_classes=element_category,
                                            json_file=TRAIN_JSON,
                                            image_root=TRAIN_PATH)

    DatasetCatalog.register("layout_val", lambda: load_coco_json(VAL_JSON, image_root=VAL_PATH, dataset_name="layout_val"))
    MetadataCatalog.get("layout_val").set(thing_classes=element_category,
                                          json_file=VAL_JSON,
                                          image_root=VAL_PATH)


def setup_cfg(args):
    """Setup configuration"""
    cfg = get_cfg()
    add_radm_config(cfg)
    add_rl_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Set device
    cfg.MODEL.DEVICE = args.device

    # RL specific settings
    if not hasattr(cfg, 'RL'):
        # Add default RL config if not present
        cfg.RL = type('RLConfig', (), {})()
        cfg.RL.NUM_ENV_STEPS = 10000000
        cfg.RL.NUM_STEPS = 2048
        cfg.RL.NUM_PROCESSES = 1
        cfg.RL.LOG_INTERVAL = 1
        cfg.RL.SAVE_INTERVAL = 100
        cfg.RL.EVAL_INTERVAL = 10
        cfg.RL.CLIP_PARAM = 0.2
        cfg.RL.PPO_EPOCH = 10
        cfg.RL.NUM_MINI_BATCH = 5
        cfg.RL.VALUE_LOSS_COEF = 0.5
        cfg.RL.ENTROPY_COEF = 0.01
        cfg.RL.MAX_GRAD_NORM = 0.5
        cfg.RL.LR = 0.0003
        cfg.RL.MAX_ELEMENTS = 20
        cfg.RL.MAX_STEPS = 50
        cfg.RL.ACTION_SCALE = 0.1
        cfg.RL.HIDDEN_DIM = 256

    cfg.freeze()
    return cfg


def main(args):
    # Setup logger
    logger = setup_logger()
    logger.info("Starting RL training for layout optimization...")

    # Setup configuration
    cfg = setup_cfg(args)

    # Register dataset
    register_layout(cfg)

    # Create data loader
    mapper = RADMDatasetMapper(cfg, is_train=True)
    data_loader = build_detection_train_loader(cfg, mapper=mapper)

    # Initialize RL trainer
    rl_trainer = RLLayoutTrainer(cfg)

    # Load checkpoint if provided
    if args.resume:
        if os.path.exists(args.resume):
            start_step = rl_trainer.load_model(args.resume)
            logger.info(f"Resumed training from step {start_step}")
        else:
            logger.warning(f"Checkpoint {args.resume} not found, starting from scratch")

    # Start training
    try:
        rl_trainer.train(data_loader)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        # Save final model
        final_checkpoint = os.path.join(rl_trainer.model_dir, 'final_model.pth')
        rl_trainer.save_model(cfg.RL.NUM_ENV_STEPS)
        logger.info(f"Final model saved to {final_checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Training for Layout Optimization")
    parser.add_argument('--config-file', required=True, help='Path to config file')
    parser.add_argument('--device', default='cuda', help='Device for training')
    parser.add_argument('--resume', default='', help='Path to checkpoint to resume from')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    main(args)
