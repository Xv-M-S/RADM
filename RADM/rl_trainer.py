"""
RL Trainer for Layout Optimization
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import time
from tqdm import tqdm
import os
import json

from .rl_layout_env import LayoutEnvironment
from .rl_agent import MultiAgentLayoutPolicy, PPOLayoutTrainer, PPOStorage
from .detector import RADM


class RLLayoutTrainer:
    """
    Main trainer for RL-enhanced layout optimization
    """

    def __init__(self, config, radm_model=None):
        print("=== RL TRAINER INIT START ===")
        print(f"Config device: {config.MODEL.DEVICE}")
        try:
            self.config = config
            self.device = torch.device(config.MODEL.DEVICE)
        except Exception as e:
            print(f"Error in RLLayoutTrainer.__init__: {e}")
            import traceback
            traceback.print_exc()
            raise
        self.logger = logging.getLogger(__name__)

        print("Setting up logging level...")
        # Set up logging level based on config
        log_level = getattr(config.RL, 'LOG_LEVEL', 'INFO')
        if log_level == 'DEBUG':
            self.logger.setLevel(logging.DEBUG)
        elif log_level == 'INFO':
            self.logger.setLevel(logging.INFO)
        elif log_level == 'WARNING':
            self.logger.setLevel(logging.WARNING)
        else:
            self.logger.setLevel(logging.INFO)

        # ===== 关键：加 Handler =====
        if not self.logger.handlers:               # 防止重复
            ch = logging.StreamHandler()           # 默认输出到 sys.stderr
            ch.setLevel(self.logger.level)         # 用跟 logger 一样的级别
            ch.setFormatter(logging.Formatter(
                '[%(asctime)s][%(name)s][%(levelname)s] %(message)s'))
            self.logger.addHandler(ch)
            self.logger.propagate = True           # 允许传播到 root logger

        print("Initializing RADM model...")
        if radm_model is not None:
            # Use the provided pre-trained RADM model
            self.radm = radm_model
            print("Using provided RADM model")
        else:
            # Initialize RADM model (same as in train_net.py)
            from detectron2.checkpoint import DetectionCheckpointer
            from detectron2.modeling import build_model

            self.radm = build_model(config)
            self.radm.eval()

            # Load RADM weights if specified
            if hasattr(config.MODEL, 'WEIGHTS') and config.MODEL.WEIGHTS:
                checkpointer = DetectionCheckpointer(self.radm, save_dir=config.OUTPUT_DIR)
                checkpointer.resume_or_load(config.MODEL.WEIGHTS, resume=False)
                print("RADM weights loaded")

        # Ensure RADM is in eval mode
        self.radm.eval()  # Keep RADM in eval mode

        print("Initializing RL components...")
        try:
            print("Creating LayoutEnvironment...")
            self.env = LayoutEnvironment(config)
            print("LayoutEnvironment created")

            print("Creating MultiAgentLayoutPolicy...")
            self.policy = MultiAgentLayoutPolicy(config)
            print("MultiAgentLayoutPolicy created")

            print("Creating PPOLayoutTrainer...")
            self.ppo_trainer = PPOLayoutTrainer(config)
            print("PPOLayoutTrainer created")

            print("=== RL TRAINER INIT COMPLETE ===")
            print("FINAL CHECK: RLLayoutTrainer initialized successfully")
        except Exception as e:
            print(f"Error during RL trainer initialization: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Training parameters
        self.num_env_steps = config.RL.NUM_ENV_STEPS if hasattr(config.RL, 'NUM_ENV_STEPS') else 10000000
        self.num_steps = config.RL.NUM_STEPS if hasattr(config.RL, 'NUM_STEPS') else 2048
        self.num_processes = config.RL.NUM_PROCESSES if hasattr(config.RL, 'NUM_PROCESSES') else 1
        self.log_interval = config.RL.LOG_INTERVAL if hasattr(config.RL, 'LOG_INTERVAL') else 1
        self.save_interval = config.RL.SAVE_INTERVAL if hasattr(config.RL, 'SAVE_INTERVAL') else 100
        self.eval_interval = config.RL.EVAL_INTERVAL if hasattr(config.RL, 'EVAL_INTERVAL') else 10

        # Data loader iterator (will be set in train method)
        self.data_loader_iter = None

        # Create directories
        self.output_dir = os.path.join(config.OUTPUT_DIR, 'rl_training')
        os.makedirs(self.output_dir, exist_ok=True)
        self.model_dir = os.path.join(self.output_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self, data_loader: DataLoader):
        """
        Main training loop
        """
        print("=== RL TRAINER STARTED ===")
        print(f"DEBUG: train called with data_loader type: {type(data_loader)}")
        print("DEBUG: About to call logger.info")
        self.logger.info("Starting RL training for layout optimization...")
        print("DEBUG: logger.info called successfully")

        # Initialize data loader iterator
        self.data_loader_iter = iter(data_loader)
        print("DEBUG: Data loader iterator initialized")

        self.logger.info(f"Training configuration:")
        self.logger.info(f"  Total steps: {self.num_env_steps}")
        self.logger.info(f"  Steps per rollout: {self.num_steps}")
        self.logger.info(f"  Log interval: {self.log_interval}")
        self.logger.info(f"  Save interval: {self.save_interval}")
        self.logger.info(f"  Eval interval: {self.eval_interval}")

        # Initialize timing
        self.start_time = time.time()
        total_steps = 0

        print("=== ENTERING TRAINING LOOP ===")
        print(f"Total steps needed: {self.num_env_steps}, steps per rollout: {self.num_steps}")
        step_count = 0
        while total_steps < self.num_env_steps:
            step_count += 1
            print(f"=== TRAINING STEP {step_count} (total_steps: {total_steps}) ===")
            self.logger.debug(f"Starting rollout collection at step {total_steps}")

            try:
                # Collect rollouts and get episode statistics
                rollouts, episode_stats = self.collect_rollouts(data_loader)
                self.logger.debug(f"Collected rollouts with {len(rollouts.returns)} agents")
                print(f"Rollouts collected successfully")
            except Exception as e:
                print(f"Error in collect_rollouts: {e}")
                import traceback
                traceback.print_exc()
                break

            # Update policy
            self.logger.debug("Updating PPO policy...")
            train_stats = self.ppo_trainer.update(rollouts)
            self.logger.debug("Policy update completed")

            # Log statistics
            total_steps += self.num_steps

            if total_steps % self.log_interval == 0:
                self.log_training_stats(train_stats, total_steps, **episode_stats)

            # Save model
            if total_steps % self.save_interval == 0:
                self.save_model(total_steps)

            # Evaluate
            if total_steps % self.eval_interval == 0:
                eval_stats = self.evaluate(data_loader)
                self.logger.info(f"Evaluation at step {total_steps}: {eval_stats}")

    def collect_rollouts(self, data_loader: DataLoader):
        """
        Collect rollouts from the environment
        """
        print("DEBUG: collect_rollouts called")
        try:
            self.policy.eval()
        except Exception as e:
            print(f"Error in policy.eval(): {e}")
            raise

        # Get a batch of data
        try:
            print("Getting batch data...")
            batch_data = next(self.data_loader_iter)
            print("Batch data obtained")
            # Log some info about the batch to verify we're getting different data
            if isinstance(batch_data, list) and len(batch_data) > 0:
                sample = batch_data[0] if isinstance(batch_data, list) else batch_data
                if hasattr(sample, 'get') and 'image_id' in sample:
                    print(f"Processing image_id: {sample['image_id']}")
        except StopIteration:
            print("Reached end of dataset, restarting from beginning")
            self.data_loader_iter = iter(data_loader)
            batch_data = next(self.data_loader_iter)
            print("Data loader iterator restarted")
        except Exception as e:
            print(f"Error getting batch data: {e}")
            raise

        # Handle detectron2 batch format (list of samples)
        if isinstance(batch_data, list):
            # Take the first sample from the batch for RL training
            batch = batch_data[0]
            self.logger.debug(f"Processing batch with {len(batch_data)} samples, using first sample")
        else:
            batch = batch_data
            self.logger.debug("Processing single sample batch")

        # Move to device
        print(f"Batch keys: {list(batch.keys())}")
        for k, v in batch.items():
            print(f"  {k}: {type(v)}, shape: {v.shape if hasattr(v, 'shape') else 'no shape'}")
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        print("Batch moved to device")

        # Generate initial layout with RADM
        # RADM expects a list of samples, so wrap the batch in a list
        print("Running RADM inference...")
        self.logger.debug("Running RADM inference...")
        try:
            with torch.no_grad():
                radm_output_list = self.radm([batch])
            print("RADM inference completed")
        except Exception as e:
            print(f"Error in RADM inference: {e}")
            raise

        # Extract the first (and only) output from the list
        radm_output = radm_output_list[0] if isinstance(radm_output_list, list) else radm_output_list

        # Extract text features
        text_features = batch.get('text_fea', torch.randn(1, 768, device=self.device))
        image_size = (batch['image'].shape[-2], batch['image'].shape[-1])

        # Log RADM output statistics and check for valid instances
        num_instances = 0
        if isinstance(radm_output, dict) and 'instances' in radm_output:
            instances = radm_output['instances']
            num_instances = len(instances.pred_boxes)
            self.logger.debug(f"RADM detected {num_instances} layout elements")
        elif isinstance(radm_output, dict):
            self.logger.debug(f"RADM output keys: {list(radm_output.keys())}")

        # Check if we have enough instances for meaningful RL training
        if num_instances == 0:
            self.logger.warning("RADM detected no layout elements, skipping this sample")
            # Return empty storage and episode stats to skip this rollout
            empty_storage = PPOStorage(self.config)
            empty_episode_stats = {'episode_rewards': [], 'episode_lengths': [], 'layout_qualities': []}
            return empty_storage, empty_episode_stats

        # Reset environment
        print("Resetting RL environment...")
        self.logger.debug("Resetting RL environment...")
        try:
            observations = self.env.reset(radm_output, text_features, image_size)
            print(f"Environment reset with {len(observations)} agents")
            self.logger.debug(f"Environment reset with {len(observations)} agents")
        except Exception as e:
            print(f"Error in env.reset: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Initialize storage
        storage = self.ppo_trainer.storage

        episode_rewards = {agent_id: 0 for agent_id in observations.keys()}
        episode_lengths = {agent_id: 0 for agent_id in observations.keys()}
        layout_qualities = []  # Store layout quality metrics

        for step in range(self.num_steps):
            print(f"Step {step}: Sampling actions for {len(observations)} agents")
            # Sample actions
            actions, action_log_probs = self.policy(observations)
            print(f"Actions sampled: {len(actions)} actions")

            # Step environment
            next_observations, rewards, dones, infos = self.env.step(actions)

            # Get value predictions
            value_preds = self.policy.get_values(observations)

            # Store in PPO storage
            masks = {agent_id: 1.0 for agent_id in observations.keys()}
            storage.insert(observations, actions, action_log_probs, value_preds, rewards, masks)

            # Update episode statistics
            step_reward_sum = 0
            for agent_id, reward in rewards.items():
                episode_rewards[agent_id] += reward
                episode_lengths[agent_id] += 1
                step_reward_sum += reward

            # Debug logging for first few steps
            if step < 3:  # Only log first few steps per rollout
                self.logger.debug(f"Step {step}: Actions sampled for {len(actions)} agents, "
                                f"Total reward: {step_reward_sum:.4f}")

            # Update observations
            observations = next_observations

            # Check if episode is done
            if dones:
                self.logger.debug(f"Episode terminated early at step {step}")
                break

        # Compute returns
        print(f"Computing next values for final observations: {list(observations.keys())}")
        next_value = self.policy.get_values(observations)
        print(f"Got next values for agents: {list(next_value.keys())}")
        print("Computing returns...")
        storage.compute_returns(next_value)
        print("Returns computed successfully")

        # Extract layout quality metrics from the last info
        if infos and 'layout_quality' in infos:
            layout_qualities.append(infos['layout_quality'])

        # Prepare episode statistics for logging
        episode_stats = {
            'episode_rewards': list(episode_rewards.values()),
            'episode_lengths': list(episode_lengths.values()),
            'layout_qualities': layout_qualities
        }

        print(f"DEBUG: collect_rollouts returning storage with {len(storage.active_agents)} active agents")
        return storage, episode_stats

    def evaluate(self, data_loader: DataLoader, num_episodes: int = 10) -> Dict:
        """
        Evaluate current policy
        """
        self.logger.debug(f"Starting evaluation with {num_episodes} episodes")
        self.policy.eval()

        eval_rewards = []
        eval_qualities = []

        # Create a temporary iterator for evaluation (don't interfere with training iterator)
        eval_iter = iter(data_loader)

        with torch.no_grad():
            for _ in range(num_episodes):
                # Get a batch of data
                try:
                    batch_data = next(eval_iter)
                except StopIteration:
                    # If we reach the end, restart the evaluation iterator
                    eval_iter = iter(data_loader)
                    batch_data = next(eval_iter)

                # Handle detectron2 batch format (list of samples)
                if isinstance(batch_data, list):
                    # Take the first sample from the batch for evaluation
                    batch = batch_data[0]
                else:
                    batch = batch_data

                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Generate initial layout with RADM
                # RADM expects a list of samples, so wrap the batch in a list
                with torch.no_grad():
                    radm_output_list = self.radm([batch])

                # Extract the first (and only) output from the list
                radm_output = radm_output_list[0] if isinstance(radm_output_list, list) else radm_output_list

                # Extract text features
                text_features = batch.get('text_features', torch.randn(1, 768, device=self.device))
                image_size = (batch['image'].shape[-2], batch['image'].shape[-1])

                # Reset environment
                observations = self.env.reset(radm_output, text_features, image_size)

                episode_reward = 0
                step_count = 0
                max_eval_steps = 20  # Shorter evaluation

                for _ in range(max_eval_steps):
                    # Sample actions
                    actions, _ = self.policy(observations)

                    # Step environment
                    next_observations, rewards, dones, infos = self.env.step(actions)

                    # Accumulate rewards
                    episode_reward += sum(rewards.values()) / len(rewards) if rewards else 0
                    step_count += 1

                    observations = next_observations

                    if dones:
                        break

                eval_rewards.append(episode_reward)
                eval_qualities.append(infos.get('layout_quality', {}))

        # Calculate statistics (convert CUDA tensors to CPU first)
        eval_rewards_cpu = [r.cpu() if isinstance(r, torch.Tensor) else r for r in eval_rewards]
        mean_reward = np.mean(eval_rewards_cpu)
        std_reward = np.std(eval_rewards_cpu)

        # Aggregate quality metrics
        quality_stats = {}
        if eval_qualities:
            for key in eval_qualities[0].keys():
                values = [q[key] for q in eval_qualities if key in q]
                if values:
                    # Convert tensors to CPU if needed
                    values_cpu = [v.cpu() if isinstance(v, torch.Tensor) else v for v in values]
                    quality_stats[f'{key}_mean'] = np.mean(values_cpu)
                    quality_stats[f'{key}_std'] = np.std(values_cpu)

        # Log evaluation results
        self.logger.info(f"Evaluation completed: Mean Reward = {mean_reward:.4f} ± {std_reward:.4f}")
        if quality_stats:
            quality_msg = "Quality metrics: "
            for key, value in quality_stats.items():
                quality_msg += f"{key}={value:.4f} "
            self.logger.info(quality_msg)

        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'quality_stats': quality_stats
        }

    def log_training_stats(self, train_stats: Dict, total_steps: int, **episode_stats):
        """
        Log training statistics
        """
        # Calculate progress percentage
        progress_pct = (total_steps / self.num_env_steps) * 100

        log_msg = f"[{progress_pct:.1f}%] Step {total_steps}/{self.num_env_steps} | "

        # Training losses with more detail
        if 'value_loss' in train_stats:
            log_msg += f"Value Loss: {train_stats['value_loss']:.4f} | "
        if 'action_loss' in train_stats:
            log_msg += f"Action Loss: {train_stats['action_loss']:.4f} | "
        if 'dist_entropy' in train_stats:
            log_msg += f"Entropy: {train_stats['dist_entropy']:.4f} | "

        # Episode statistics
        episode_rewards = episode_stats.get('episode_rewards', [])
        episode_lengths = episode_stats.get('episode_lengths', [])
        layout_qualities = episode_stats.get('layout_qualities', [])

        # Convert tensors to CPU numpy arrays
        if episode_rewards:
            episode_rewards_cpu = [r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else r for r in episode_rewards]
            mean_reward = np.mean(episode_rewards_cpu)
            std_reward = np.std(episode_rewards_cpu)
            max_reward = np.max(episode_rewards_cpu) if episode_rewards_cpu else 0
            log_msg += f"Rewards: {mean_reward:.3f}±{std_reward:.3f} (max: {max_reward:.3f}) | "

        if episode_lengths:
            episode_lengths_cpu = [l.detach().cpu().numpy() if isinstance(l, torch.Tensor) else l for l in episode_lengths]
            mean_length = np.mean(episode_lengths_cpu)
            max_length = np.max(episode_lengths_cpu) if episode_lengths_cpu else 0
            log_msg += f"Episode Length: {mean_length:.1f} (max: {max_length}) | "

        # Layout quality metrics
        if layout_qualities:
            if layout_qualities and isinstance(layout_qualities[0], dict):
                # Multiple quality metrics
                for key in layout_qualities[0].keys():
                    values = [q[key] for q in layout_qualities if isinstance(q, dict) and key in q]
                    if values:
                        values_cpu = [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in values]
                        mean_val = np.mean(values_cpu)
                        log_msg += f"{key}: {mean_val:.4f} | "
            else:
                # Single quality metric
                layout_qualities_cpu = [q.detach().cpu().numpy() if isinstance(q, torch.Tensor) else q for q in layout_qualities]
                mean_quality = np.mean(layout_qualities_cpu)
                log_msg += f"Quality: {mean_quality:.4f}"

        # Performance metrics
        if hasattr(self, 'start_time'):
            elapsed_time = time.time() - self.start_time
            steps_per_sec = total_steps / elapsed_time if elapsed_time > 0 else 0
            eta_seconds = (self.num_env_steps - total_steps) / steps_per_sec if steps_per_sec > 0 else 0
            eta_minutes = eta_seconds / 60
            log_msg += f" | Speed: {steps_per_sec:.1f} steps/sec | ETA: {eta_minutes:.1f} min"
        else:
            self.start_time = time.time()

        print(f"[LOG] {log_msg}")  # Use print for immediate visibility
        self.logger.info(log_msg)

        # Periodic detailed logging
        if total_steps % (self.log_interval * 10) == 0:
            self.logger.info("=" * 80)
            self.logger.info("DETAILED TRAINING STATISTICS:")
            self.logger.info(f"Total Steps: {total_steps}")
            self.logger.info(f"Training Progress: {progress_pct:.2f}%")
            for key, value in train_stats.items():
                self.logger.info(f"  {key}: {value:.6f}")
            if episode_rewards:
                # Convert to numpy for statistical calculations
                episode_rewards_cpu = [r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else r for r in episode_rewards]
                self.logger.info(f"  Episode count: {len(episode_rewards_cpu)}")
                self.logger.info(f"  Reward distribution: min={np.min(episode_rewards_cpu):.3f}, "
                               f"median={np.median(episode_rewards_cpu):.3f}, max={np.max(episode_rewards_cpu):.3f}")
            self.logger.info("=" * 80)

    def save_model(self, step: int):
        """
        Save model checkpoint
        """
        checkpoint_path = os.path.join(self.model_dir, f'rl_policy_step_{step}.pth')

        torch.save({
            'step': step,
            'policy_state_dict': self.policy.state_dict(),
            'ppo_optimizer_state_dict': self.ppo_trainer.optimizer.state_dict(),
            'config': self.config
        }, checkpoint_path)

        self.logger.info(f"Saved model checkpoint to {checkpoint_path}")

    def load_model(self, checkpoint_path: str):
        """
        Load model checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.ppo_trainer.optimizer.load_state_dict(checkpoint['ppo_optimizer_state_dict'])

        self.logger.info(f"Loaded model from {checkpoint_path} at step {checkpoint['step']}")

        return checkpoint['step']

    def optimize_layout(self, radm_output: Dict, text_features: torch.Tensor,
                       image_size: Tuple[int, int], num_steps: int = 10) -> Dict:
        """
        Optimize a single layout using trained RL policy
        """
        self.policy.eval()

        # Reset environment
        observations = self.env.reset(radm_output, text_features, image_size)

        with torch.no_grad():
            for _ in range(num_steps):
                # Sample actions
                actions, _ = self.policy(observations)

                # Step environment
                next_observations, rewards, dones, infos = self.env.step(actions)

                observations = next_observations

                if dones:
                    break

        # Return optimized layout
        return {
            'boxes': self.env.get_layout_boxes(),
            'classes': self.env.get_layout_classes(),
            'quality': infos.get('layout_quality', {}),
            'final_reward': sum(rewards.values()) / len(rewards) if rewards else 0
        }


class RLLayoutInference:
    """
    Inference class for RL-optimized layout generation
    """

    def __init__(self, config, checkpoint_path: str, radm_model=None):
        self.config = config
        self.device = torch.device(config.MODEL.DEVICE)

        # Initialize components
        if radm_model is not None:
            self.radm = radm_model
        else:
            from detectron2.modeling import build_model
            from detectron2.checkpoint import DetectionCheckpointer

            self.radm = build_model(config)
            if hasattr(config.MODEL, 'WEIGHTS') and config.MODEL.WEIGHTS:
                checkpointer = DetectionCheckpointer(self.radm, save_dir=config.OUTPUT_DIR)
                checkpointer.resume_or_load(config.MODEL.WEIGHTS, resume=False)

        self.radm.eval()

        self.env = LayoutEnvironment(config)
        self.policy = MultiAgentLayoutPolicy(config)

        # Load trained RL policy
        self.load_policy(checkpoint_path)

    def load_policy(self, checkpoint_path: str):
        """Load trained policy"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.eval()
        print(f"Loaded RL policy from {checkpoint_path}")

    def __call__(self, batch: Dict) -> Dict:
        """
        Generate optimized layout for input batch
        """
        # Move to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Generate initial layout with RADM
        # RADM expects a list of samples, so wrap the batch in a list
        with torch.no_grad():
            radm_output_list = self.radm([batch])

        # Extract the first (and only) output from the list
        radm_output = radm_output_list[0] if isinstance(radm_output_list, list) else radm_output_list

        # Extract features
        text_features = batch.get('text_features', torch.randn(1, 768, device=self.device))
        image_size = (batch['image'].shape[-2], batch['image'].shape[-1])

        # Optimize with RL
        optimized_result = self.optimize_layout(radm_output, text_features, image_size)

        # Combine RADM and RL results
        result = {
            'radm_output': radm_output,
            'rl_optimized': optimized_result,
            'final_boxes': optimized_result['boxes'],
            'final_classes': optimized_result['classes']
        }

        return result

    def optimize_layout(self, radm_output: Dict, text_features: torch.Tensor,
                       image_size: Tuple[int, int], num_steps: int = 10) -> Dict:
        """
        Optimize layout using RL policy
        """
        trainer = RLLayoutTrainer(self.config)
        return trainer.optimize_layout(radm_output, text_features, image_size, num_steps)
