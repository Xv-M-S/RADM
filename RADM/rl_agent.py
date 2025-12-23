"""
Multi-Agent Reinforcement Learning Agents for Layout Optimization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np


class LayoutAgent(nn.Module):
    """
    Individual agent for layout element optimization
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device(config.MODEL.DEVICE)

        # Network dimensions
        self.hidden_dim = config.RL.HIDDEN_DIM if hasattr(config.RL, 'HIDDEN_DIM') else 256
        self.text_dim = 768  # RoBERTa text feature dimension
        self.action_dim = 4  # [dx, dy, dw, dh]

        # State encoding networks
        self.bbox_encoder = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.text_encoder = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # Neighbor communication
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # Attention for neighbor communication
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Actor-Critic networks
        self.actor = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),  # own + global + neighbors
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim * 2)  # mean and log_std
        )

        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

        # Class embedding
        self.class_embedding = nn.Embedding(
            config.MODEL.RADM.NUM_CLASSES + 1,  # +1 for background
            self.hidden_dim
        )

        self.to(self.device)

    def forward(self, observation: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for single agent
        Args:
            observation: Agent observation dict
        Returns:
            action, log_prob, value
        """
        # Encode own state
        bbox_feat = self.bbox_encoder(observation['bbox'].unsqueeze(0))
        class_feat = self.class_embedding(torch.tensor(observation['class_id'], device=self.device)).unsqueeze(0)
        own_feat = bbox_feat + class_feat

        # Encode global text feature
        global_text_feat = self.text_encoder(observation['global_text_feature'].unsqueeze(0))

        # Encode neighbor states
        if observation['neighbor_states']:
            neighbor_feats = []
            for neighbor_state in observation['neighbor_states']:
                n_bbox_feat = self.bbox_encoder(neighbor_state['bbox'].unsqueeze(0))
                n_class_feat = self.class_embedding(torch.tensor(neighbor_state['class_id'], device=self.device)).unsqueeze(0)
                n_feat = n_bbox_feat + n_class_feat
                neighbor_feats.append(n_feat)

            neighbor_feats = torch.cat(neighbor_feats, dim=0)

            # 正确调整维度以适配MultiheadAttention (batch_first=True)
            own_for_attn = own_feat.unsqueeze(1)  # [1, 1, hidden_dim] - 单个查询
            neighbor_for_attn = neighbor_feats.unsqueeze(0)  # [1, num_neighbors, hidden_dim] - 多个键值对

            attn_output, _ = self.attention(
                own_for_attn, neighbor_for_attn, neighbor_for_attn
            )  # 输出 [1, 1, hidden_dim]

            neighbor_agg = attn_output.squeeze(1)  # [1, hidden_dim] - 恢复到预期维度
            
        else:
            neighbor_agg = torch.zeros_like(own_feat)

        # Concatenate all features
        combined_feat = torch.cat([own_feat, global_text_feat, neighbor_agg], dim=-1)

        # Actor: sample action
        actor_output = self.actor(combined_feat)

        # Check for NaN/inf values
        if torch.isnan(actor_output).any() or torch.isinf(actor_output).any():
            print(f"WARNING: actor_output contains NaN/inf in forward: {actor_output}")
            actor_output = torch.where(torch.isfinite(actor_output), actor_output, torch.zeros_like(actor_output))

        mean, log_std = torch.chunk(actor_output, 2, dim=-1)

        # Clamp log_std to prevent extreme values
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        # Sample action from Gaussian
        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()
        log_prob = normal.log_prob(action).sum(dim=-1)

        # Critic: value estimation
        value = self.critic(combined_feat)

        return action.squeeze(0), log_prob.squeeze(0), value.squeeze()

    def evaluate_actions(self, observation: Dict, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO training
        """
        # Encode state (same as forward)
        bbox_feat = self.bbox_encoder(observation['bbox'].unsqueeze(0))
        class_feat = self.class_embedding(torch.tensor(observation['class_id'], device=self.device)).unsqueeze(0)
        own_feat = bbox_feat + class_feat

        global_text_feat = self.text_encoder(observation['global_text_feature'].unsqueeze(0))

        if observation['neighbor_states']:
            neighbor_feats = []
            for neighbor_state in observation['neighbor_states']:
                n_bbox_feat = self.bbox_encoder(neighbor_state['bbox'].unsqueeze(0))
                n_class_feat = self.class_embedding(torch.tensor(neighbor_state['class_id'], device=self.device)).unsqueeze(0)
                n_feat = n_bbox_feat + n_class_feat
                neighbor_feats.append(n_feat)

            neighbor_feats = torch.cat(neighbor_feats, dim=0)

            # 正确调整维度以适配MultiheadAttention (batch_first=True)
            own_for_attn = own_feat.unsqueeze(1)  # [1, 1, hidden_dim] - 单个查询
            neighbor_for_attn = neighbor_feats.unsqueeze(0)  # [1, num_neighbors, hidden_dim] - 多个键值对

            attn_output, _ = self.attention(
                own_for_attn, neighbor_for_attn, neighbor_for_attn
            )  # 输出 [1, 1, hidden_dim]

            neighbor_agg = attn_output.squeeze(1)  # [1, hidden_dim] - 恢复到预期维度
        else:
            neighbor_agg = torch.zeros_like(own_feat)

        combined_feat = torch.cat([own_feat, global_text_feat, neighbor_agg], dim=-1)

        # Actor evaluation
        actor_output = self.actor(combined_feat)

        # Debug: check for NaN/inf values
        if torch.isnan(actor_output).any() or torch.isinf(actor_output).any():
            print(f"WARNING: actor_output contains NaN/inf: {actor_output}")
            print(f"combined_feat: {combined_feat}")
            # Replace NaN/inf with zeros
            actor_output = torch.where(torch.isfinite(actor_output), actor_output, torch.zeros_like(actor_output))

        mean, log_std = torch.chunk(actor_output, 2, dim=-1)

        # Clamp log_std to prevent extreme values
        log_std = torch.clamp(log_std, -20, 2)  # exp(-20) ≈ 2e-9, exp(2) ≈ 7.4
        std = torch.exp(log_std)

        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(actions.unsqueeze(0)).sum(dim=-1)
        entropy = normal.entropy().sum(dim=-1)

        # Critic evaluation
        value = self.critic(combined_feat)

        return log_prob.squeeze(0), entropy.squeeze(0), value.squeeze()


class MultiAgentLayoutPolicy(nn.Module):
    """
    Multi-agent policy managing all layout agents
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device(config.MODEL.DEVICE)

        # Create agent pool (shared parameters)
        self.agent = LayoutAgent(config)

        # Central critic for cooperative learning
        self.central_critic = nn.Sequential(
            nn.Linear(256 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.to(self.device)

    def forward(self, observations: Dict[int, Dict]) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Forward pass for all agents
        Args:
            observations: Dict of {agent_id: observation}
        Returns:
            actions, log_probs
        """
        actions = {}
        log_probs = {}
        values = {}

        for agent_id, obs in observations.items():
            # Convert numpy arrays to tensors if needed
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    obs[key] = torch.tensor(value, device=self.device)
                elif isinstance(value, torch.Tensor):
                    obs[key] = value.to(self.device)

            action, log_prob, value = self.agent(obs)
            actions[agent_id] = action
            log_probs[agent_id] = log_prob
            values[agent_id] = value

        return actions, log_probs

    def evaluate_actions(self, observations_batch: Dict[int, List[Dict]], actions_batch: Dict[int, List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for mini-batch training (used in PPO)
        observations_batch: {agent_id: [obs1, obs2, ...]}
        actions_batch: {agent_id: [action1, action2, ...]}
        """
        all_log_probs = []
        all_entropies = []
        all_values = []

        for agent_id in observations_batch.keys():
            obs_list = observations_batch[agent_id]
            action_list = actions_batch[agent_id]

            for obs, action in zip(obs_list, action_list):
                # Convert to tensors
                for key, value in obs.items():
                    if isinstance(value, np.ndarray):
                        obs[key] = torch.tensor(value, device=self.device)
                    elif isinstance(value, torch.Tensor):
                        obs[key] = value.to(self.device)

                if isinstance(action, np.ndarray):
                    action = torch.tensor(action, device=self.device)
                elif isinstance(action, torch.Tensor):
                    action = action.to(self.device)

                log_prob, entropy, value = self.agent.evaluate_actions(obs, action)
                all_log_probs.append(log_prob)
                all_entropies.append(entropy)
                all_values.append(value)

        # Debug: print some statistics
        if all_log_probs:
            log_probs_tensor = torch.stack(all_log_probs)
            entropies_tensor = torch.stack(all_entropies)
            values_tensor = torch.stack(all_values)
            print(f"DEBUG: PPO batch - log_probs range: {log_probs_tensor.min():.4f} to {log_probs_tensor.max():.4f}, "
                  f"entropy: {entropies_tensor.mean():.4f}, values: {values_tensor.mean():.4f}")

        return torch.stack(all_log_probs), torch.stack(all_entropies), torch.stack(all_values)

    def get_values(self, observations: Dict[int, Dict]) -> Dict[int, torch.Tensor]:
        """
        Get value estimates for all agents
        """
        try:
            values = {}
            for agent_id, obs in observations.items():
                # Convert to tensors
                for key, value in obs.items():
                    if isinstance(value, np.ndarray):
                        obs[key] = torch.tensor(value, device=self.device)
                    elif isinstance(value, torch.Tensor):
                        obs[key] = value.to(self.device)

                _, _, value = self.agent(obs)
                values[agent_id] = value

        except Exception as e:
            print(f"Error in get_values: {e}")
            import traceback
            traceback.print_exc()
            raise
        return values

    def central_value_function(self, observations: Dict[int, Dict]) -> torch.Tensor:
        """
        Central value function for cooperative learning
        """
        # Aggregate all agent observations
        global_features = []
        for agent_id, obs in observations.items():
            bbox_feat = self.agent.bbox_encoder(obs['bbox'].unsqueeze(0))
            class_feat = self.agent.class_embedding(obs['class_id']).unsqueeze(0)
            own_feat = bbox_feat + class_feat
            global_features.append(own_feat.squeeze(0))

        if global_features:
            global_state = torch.stack(global_features).mean(dim=0)
        else:
            global_state = torch.zeros(self.agent.hidden_dim, device=self.device)

        # Global text feature
        text_feat = observations[list(observations.keys())[0]]['global_text_feature']

        # Combine
        combined = torch.cat([global_state, text_feat], dim=-1)
        return self.central_critic(combined.unsqueeze(0)).squeeze(0)


class PPOLayoutTrainer:
    """
    PPO trainer for layout optimization
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.MODEL.DEVICE)

        # PPO hyperparameters
        self.clip_param = config.RL.CLIP_PARAM if hasattr(config.RL, 'CLIP_PARAM') else 0.2
        self.ppo_epoch = config.RL.PPO_EPOCH if hasattr(config.RL, 'PPO_EPOCH') else 10
        self.num_mini_batch = config.RL.NUM_MINI_BATCH if hasattr(config.RL, 'NUM_MINI_BATCH') else 5
        self.value_loss_coef = config.RL.VALUE_LOSS_COEF if hasattr(config.RL, 'VALUE_LOSS_COEF') else 0.5
        self.entropy_coef = config.RL.ENTROPY_COEF if hasattr(config.RL, 'ENTROPY_COEF') else 0.01
        self.max_grad_norm = config.RL.MAX_GRAD_NORM if hasattr(config.RL, 'MAX_GRAD_NORM') else 0.5
        self.lr = config.RL.LR if hasattr(config.RL, 'LR') else 3e-4

        # Initialize policy and optimizer
        self.policy = MultiAgentLayoutPolicy(config)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        # Storage for PPO
        self.storage = PPOStorage(config)

    def update(self, rollouts):
        """
        Update policy using PPO
        """
        # PPO update called
        # Handle multi-agent rollouts
        all_advantages = []
        all_agent_indices = []
        all_step_indices = []

        for agent_id in rollouts.active_agents:
            if rollouts.returns[agent_id]:
                # Use only the valid steps (where agent exists)
                valid_steps = len(rollouts.returns[agent_id]) - 1  # -1 because we use [:-1]
                if valid_steps > 0:
                    agent_advantages = torch.stack(rollouts.returns[agent_id][:valid_steps]) - torch.stack(rollouts.value_preds[agent_id][:valid_steps])
                    all_advantages.append(agent_advantages)
                    # Track which agent and step this advantage belongs to
                    all_agent_indices.extend([agent_id] * valid_steps)
                    all_step_indices.extend(range(valid_steps))

        if all_advantages:
            advantages_flat = torch.cat(all_advantages)
            advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-5)

            # Create a 2D advantages tensor indexed by [step, agent]
            max_steps = max(len(rollouts.returns[agent_id]) - 1 for agent_id in rollouts.active_agents if rollouts.returns[agent_id])
            max_agents = len(rollouts.active_agents)
            advantages = torch.zeros(max_steps, max_agents, device=self.device)

            # Fill in the advantages
            idx = 0
            active_agent_list = sorted(rollouts.active_agents)
            for local_idx, agent_id in enumerate(active_agent_list):
                if rollouts.returns[agent_id]:
                    valid_steps = len(rollouts.returns[agent_id]) - 1
                    advantages[:valid_steps, local_idx] = advantages_flat[idx:idx + valid_steps]
                    idx += valid_steps
        else:
            advantages = torch.zeros(0, len(rollouts.returns), device=self.device)
            print("DEBUG: no advantages calculated")

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch)

            for sample in data_generator:
                observations_batch, actions_batch, value_preds_batch, \
                return_batch, old_action_log_probs_batch, adv_targ = sample

                # Evaluate actions for the mini-batch
                action_log_probs, dist_entropy, values = self.policy.evaluate_actions(
                    observations_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                # Average entropy across batch
                dist_entropy = dist_entropy.mean()

                self.optimizer.zero_grad()
                total_loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef

                # Check for NaN loss
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"WARNING: NaN/inf loss detected: {total_loss}")
                    print(f"value_loss: {value_loss}, action_loss: {action_loss}, dist_entropy: {dist_entropy}")
                    continue  # Skip this update

                total_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        return {
            'value_loss': value_loss_epoch / num_updates,
            'action_loss': action_loss_epoch / num_updates,
            'dist_entropy': dist_entropy_epoch / num_updates
        }


class PPOStorage:
    """
    Storage for PPO rollouts
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.MODEL.DEVICE)
        self.num_steps = config.RL.NUM_STEPS if hasattr(config.RL, 'NUM_STEPS') else 2048
        self.num_agents = config.RL.MAX_ELEMENTS if hasattr(config.RL, 'MAX_ELEMENTS') else 20

        # Storage buffers - use dictionaries to store only active agents
        self.observations = {}
        self.actions = {}
        self.rewards = {}
        self.value_preds = {}
        self.returns = {}
        self.action_log_probs = {}
        self.masks = {}

        self.step = 0
        self.active_agents = set()  # Track which agents are active

    def insert(self, observations, actions, action_log_probs, value_preds, rewards, masks):
        """Insert step data"""
        for agent_id in observations.keys():
            if agent_id not in self.observations:
                self.observations[agent_id] = []
                self.actions[agent_id] = []
                self.action_log_probs[agent_id] = []
                self.value_preds[agent_id] = []
                self.returns[agent_id] = []
                self.rewards[agent_id] = []
                self.masks[agent_id] = []
                self.active_agents.add(agent_id)

            self.observations[agent_id].append(observations[agent_id])
            self.actions[agent_id].append(actions[agent_id])
            self.action_log_probs[agent_id].append(action_log_probs[agent_id])
            self.value_preds[agent_id].append(value_preds[agent_id])
            self.rewards[agent_id].append(rewards[agent_id])
            self.masks[agent_id].append(masks[agent_id])

        self.step = (self.step + 1) % self.num_steps

    def compute_returns(self, next_value, gamma=0.99, tau=0.95):
        """Compute returns using GAE"""
        try:
            for agent_id in self.active_agents:
                if not self.rewards[agent_id]:
                    continue

                gae = 0
                for step in reversed(range(len(self.rewards[agent_id]))):
                    if step == len(self.rewards[agent_id]) - 1:
                        next_val = next_value[agent_id] if agent_id in next_value else 0
                    else:
                        next_val = self.value_preds[agent_id][step + 1]

                    reward = self.rewards[agent_id][step]
                    mask = self.masks[agent_id][step]
                    value_pred = self.value_preds[agent_id][step]

                    # Ensure all values are floats for calculation
                    reward_scalar = reward.item() if hasattr(reward, 'item') else float(reward)
                    next_val_scalar = next_val.item() if hasattr(next_val, 'item') else float(next_val)
                    value_pred_scalar = value_pred.item() if hasattr(value_pred, 'item') else float(value_pred)

                    delta = reward_scalar + gamma * next_val_scalar * mask - value_pred_scalar
                    gae = delta + gamma * tau * mask * gae
                    return_val = gae + value_pred_scalar
                    self.returns[agent_id].insert(0, torch.tensor(return_val, device=self.device))

            # Pad sequences to ensure all active agents have the same length
            if self.active_agents:
                max_len = max(len(self.returns[agent_id]) for agent_id in self.active_agents)
                for agent_id in self.active_agents:
                    while len(self.returns[agent_id]) < max_len:
                        # Pad with the last value
                        self.returns[agent_id].append(self.returns[agent_id][-1])
                    while len(self.value_preds[agent_id]) < max_len:
                        # Pad with zeros for value predictions
                        device = self.value_preds[agent_id][0].device if self.value_preds[agent_id] else self.device
                        self.value_preds[agent_id].append(torch.tensor(0.0, device=device))
                    while len(self.rewards[agent_id]) < max_len:
                        # Pad with zeros for rewards
                        self.rewards[agent_id].append(0.0)
                    while len(self.masks[agent_id]) < max_len:
                        # Pad with False for masks
                        self.masks[agent_id].append(False)
        except Exception as e:
            print(f"Error in compute_returns: {e}")
            import traceback
            traceback.print_exc()
            raise

    def feed_forward_generator(self, advantages, num_mini_batch):
        """Generator for mini-batch training"""
        if not self.active_agents:
            return

        # Use advantages tensor to determine the correct dimensions
        num_steps = advantages.shape[0]  # Use advantages shape instead of observations
        num_agents = advantages.shape[1]
        batch_size = num_steps * num_agents // num_mini_batch
        sampler = BatchSampler(batch_size, num_steps, num_agents)

        batch_yielded = 0
        active_agent_list = sorted(self.active_agents)
        agent_id_to_idx = {agent_id: idx for idx, agent_id in enumerate(active_agent_list)}

        for indices in sampler:
            observations_batch = {}
            actions_batch = {}
            value_preds_batch = []
            return_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for idx in indices:
                step_idx = torch.div(idx, num_agents, rounding_mode='floor')
                agent_local_idx = idx % num_agents

                # Map local index to actual agent ID
                if agent_local_idx < len(active_agent_list):
                    agent_id = active_agent_list[agent_local_idx]

                    if agent_id in self.observations and step_idx < len(self.observations[agent_id]):
                        obs = self.observations[agent_id][step_idx]
                        action = self.actions[agent_id][step_idx]

                        if agent_id not in observations_batch:
                            observations_batch[agent_id] = []
                            actions_batch[agent_id] = []

                        observations_batch[agent_id].append(obs)
                        actions_batch[agent_id].append(action)
                        value_preds_batch.append(self.value_preds[agent_id][step_idx])
                        return_batch.append(self.returns[agent_id][step_idx])
                        old_action_log_probs_batch.append(self.action_log_probs[agent_id][step_idx])
                        # Use local index for advantages tensor
                        adv_targ.append(advantages[step_idx, agent_local_idx])

            if observations_batch:
                yield observations_batch, actions_batch, \
                      torch.stack(value_preds_batch), torch.stack(return_batch), \
                      torch.stack(old_action_log_probs_batch), torch.stack(adv_targ)


class BatchSampler:
    """Batch sampler for PPO"""

    def __init__(self, batch_size, num_steps, num_agents):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_agents = num_agents

    def __iter__(self):
        indices = torch.randperm(self.num_steps * self.num_agents)
        for i in range(0, len(indices), self.batch_size):
            yield indices[i:i+self.batch_size]

    def __len__(self):
        return (self.num_steps * self.num_agents + self.batch_size - 1) // self.batch_size
