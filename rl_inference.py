#!/usr/bin/env python3
"""
RL-Enhanced Layout Generation Inference Script
"""
import torch
import argparse
import os
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, MetadataCatalog

from RADM import add_radm_config
from RADM.rl_trainer import RLLayoutInference
from RADM.util.model_ema import add_model_ema_configs, may_get_ema_checkpointer, EMADetectionCheckpointer
from detectron2.checkpoint import DetectionCheckpointer


def setup_cfg(args):
    """Setup configuration"""
    cfg = get_cfg()
    add_radm_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Set device
    cfg.MODEL.DEVICE = args.device
    cfg.freeze()
    return cfg


def load_image(image_path):
    """Load and preprocess image"""
    image = Image.open(image_path).convert('RGB')
    return image


def prepare_input(image_path, text_features_path=None):
    """
    Prepare input data for inference
    """
    # Load image
    image = load_image(image_path)

    # Convert to tensor
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

    # Load or create text features
    if text_features_path and os.path.exists(text_features_path):
        text_features = torch.load(text_features_path)
    else:
        # Use random text features for demo
        text_features = torch.randn(1, 768)
        print("Warning: Using random text features. Provide text_features_path for better results.")

    # Create batch
    batch = {
        'image': image_tensor.unsqueeze(0),  # Add batch dimension
        'text_features': text_features,
        'height': image.height,
        'width': image.width
    }

    return batch, image


def visualize_layout(image, boxes, classes, class_names, save_path=None):
    """
    Visualize layout on image
    """
    # Create a copy for drawing
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)

    # Colors for different classes
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']

    # Draw boxes
    for i, (box, class_id) in enumerate(zip(boxes, classes)):
        # Convert normalized coordinates to image coordinates
        x1, y1, x2, y2 = box
        x1, x2 = x1 * image.width, x2 * image.width
        y1, y2 = y1 * image.height, y2 * image.height

        # Draw rectangle
        color = colors[class_id % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        draw.text((x1, y1 - 20), f"{class_name}", fill=color)

    if save_path:
        vis_image.save(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        return vis_image


def main(args):
    # Setup logger
    logger = setup_logger()

    # Setup configuration
    cfg = setup_cfg(args)

    # Register layout dataset (for class names)
    element_category = ["Logo", "文字", "衬底", "符号元素", "强调突出子部分文字"]

    # Initialize RL inference model
    if not os.path.exists(args.rl_checkpoint):
        raise FileNotFoundError(f"RL checkpoint not found: {args.rl_checkpoint}")

    rl_inference = RLLayoutInference(cfg, args.rl_checkpoint)

    # Prepare input
    batch, original_image = prepare_input(args.image_path, args.text_features)

    logger.info(f"Processing image: {args.image_path}")
    logger.info(f"Image size: {original_image.size}")

    # Run inference
    with torch.no_grad():
        result = rl_inference(batch)

    # Extract results
    radm_boxes = result['radm_output']['pred_boxes'][0]  # RADM predictions
    radm_classes = result['radm_output']['pred_logits'][0].argmax(-1)  # RADM class predictions

    rl_boxes = result['final_boxes']  # RL optimized boxes
    rl_classes = result['final_classes']  # RL optimized classes

    quality_metrics = result['rl_optimized']['quality']
    final_reward = result['rl_optimized']['final_reward']

    logger.info(f"RADM detected {len(radm_boxes)} elements")
    logger.info(f"RL optimized layout quality: {quality_metrics}")
    logger.info(f"Final RL reward: {final_reward:.4f}")

    # Visualize and save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        # Save RADM result
        radm_vis = visualize_layout(
            original_image, radm_boxes.cpu().numpy(),
            radm_classes.cpu().numpy(), element_category,
            os.path.join(args.output_dir, 'radm_layout.png')
        )

        # Save RL optimized result
        rl_vis = visualize_layout(
            original_image, rl_boxes.cpu().numpy(),
            rl_classes.cpu().numpy(), element_category,
            os.path.join(args.output_dir, 'rl_optimized_layout.png')
        )

        # Save comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.imshow(radm_vis)
        ax1.set_title('RADM Generated Layout')
        ax1.axis('off')

        ax2.imshow(rl_vis)
        ax2.set_title('RL Optimized Layout')
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'layout_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Save metrics
        metrics = {
            'quality_metrics': {k: float(v) for k, v in quality_metrics.items()},
            'final_reward': float(final_reward),
            'num_elements_before': len(radm_boxes),
            'num_elements_after': len(rl_boxes)
        }

        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Results saved to {args.output_dir}")

    # Print summary
    print("\n" + "="*50)
    print("LAYOUT OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Original RADM elements: {len(radm_boxes)}")
    print(f"RL optimized elements: {len(rl_boxes)}")
    print(f"Layout quality metrics:")
    for metric, value in quality_metrics.items():
        print(f"{metric}: {value:.4f}")
    print(f"Final RL reward: {final_reward:.4f}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL-Enhanced Layout Generation Inference")
    parser.add_argument('--config-file', required=True, help='Path to config file')
    parser.add_argument('--rl-checkpoint', required=True, help='Path to RL checkpoint')
    parser.add_argument('--image-path', required=True, help='Path to input image')
    parser.add_argument('--text-features', default=None, help='Path to text features')
    parser.add_argument('--output-dir', default='./rl_output', help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device to run inference')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    main(args)
