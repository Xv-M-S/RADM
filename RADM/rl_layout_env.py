"""
Multi-Agent Reinforcement Learning Environment for Layout Generation
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import copy

from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


class LayoutElement:
    """Represents a single layout element (agent)"""
    def __init__(self, element_id: int, bbox: torch.Tensor, class_id: int, text_feature: torch.Tensor):
        self.id = element_id
        self.bbox = bbox  # [x, y, w, h] in normalized coordinates
        self.class_id = class_id
        self.text_feature = text_feature
        self.neighbors = []  # List of neighboring element IDs
        self.reward_history = []

    def update_bbox(self, new_bbox: torch.Tensor):
        """Update bounding box"""
        self.bbox = torch.clamp(new_bbox, 0, 1)

    def get_state(self) -> Dict:
        """Get current state representation"""
        return {
            'bbox': self.bbox.clone(),
            'class_id': self.class_id,
            'text_feature': self.text_feature.clone(),
            'neighbors': self.neighbors.copy()
        }


class LayoutEnvironment:
    """
    Multi-Agent Environment for Layout Optimization
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.MODEL.DEVICE)
        self.max_elements = config.RL.MAX_ELEMENTS if hasattr(config.RL, 'MAX_ELEMENTS') else 20
        self.action_dim = 4  # [dx, dy, dw, dh] - relative changes
        self.action_scale = config.RL.ACTION_SCALE if hasattr(config.RL, 'ACTION_SCALE') else 0.1

        # Initialize environment state
        self.elements = {}
        self.global_text_feature = None
        self.image_size = None
        self.step_count = 0
        self.max_steps = config.RL.MAX_STEPS if hasattr(config.RL, 'MAX_STEPS') else 50

        # Reward weights
        self.reward_weights = {
            'overlap_penalty': -2.0,
            'alignment_bonus': 1.0,
            'balance_bonus': 0.5,
            'semantic_coherence': 0.8,
            'aesthetic_score': 1.2
        }

    def reset(self, radm_output, text_feature: torch.Tensor, image_size: Tuple[int, int]) -> Dict:
        """
        Initialize environment with RADM generated layout
        Args:
            radm_output: Output from RADM model (list of dicts with 'instances' key)
            text_feature: Global text feature
            image_size: (height, width) of the image
        """
        self.elements = {}
        # Handle text_feature - ensure it's a tensor
        if isinstance(text_feature, dict):
            # If text_feature is a dict, try to extract the actual feature tensor
            text_feature = text_feature.get('text_fea', torch.randn(1, 768, device=self.device))
        if isinstance(text_feature, torch.Tensor):
            self.global_text_feature = text_feature.squeeze(0).clone() if text_feature.dim() > 1 else text_feature.clone()
        else:
            self.global_text_feature = torch.randn(768, device=self.device)
        self.image_size = image_size
        self.step_count = 0

        # Handle different RADM output formats
        if isinstance(radm_output, dict) and 'pred_boxes' in radm_output:
            # Test/mock format
            pred_boxes = radm_output['pred_boxes']  # [batch, num_proposals, 4]
            pred_classes = radm_output['pred_logits']  # [batch, num_proposals, num_classes]

            # Take first batch
            boxes = pred_boxes[0]  # [num_proposals, 4]
            classes = torch.argmax(pred_classes[0], dim=-1)  # [num_proposals]

            # Filter valid elements (non-background)
            valid_mask = classes < self.config.MODEL.RADM.NUM_CLASSES
            boxes = boxes[valid_mask]
            classes = classes[valid_mask]

            if len(boxes) == 0:
                # No valid instances detected, create some default elements for RL training
                boxes = torch.tensor([[0.5, 0.5, 0.3, 0.2], [0.3, 0.3, 0.2, 0.15]], device=self.device)
                classes = torch.tensor([0, 1], device=self.device)
        elif isinstance(radm_output, dict) and 'instances' in radm_output:
            # RADM output format: {'instances': Instances(...)}
            instances = radm_output['instances']
            boxes = instances.pred_boxes.tensor  # [num_instances, 4]
            classes = instances.pred_classes  # [num_instances]

            if len(boxes) == 0:
                # No instances detected, create some default elements for RL training
                boxes = torch.tensor([[0.5, 0.5, 0.3, 0.2], [0.3, 0.3, 0.2, 0.15]], device=self.device)
                classes = torch.tensor([0, 1], device=self.device)
            else:
                # Convert to normalized coordinates
                img_h, img_w = image_size
                boxes = boxes.clone()
                boxes[:, 0] /= img_w  # x1
                boxes[:, 1] /= img_h  # y1
                boxes[:, 2] /= img_w  # x2
                boxes[:, 3] /= img_h  # y2

                # Convert to cx, cy, w, h format
                cx = (boxes[:, 0] + boxes[:, 2]) / 2
                cy = (boxes[:, 1] + boxes[:, 3]) / 2
                w = boxes[:, 2] - boxes[:, 0]
                h = boxes[:, 3] - boxes[:, 1]
                boxes = torch.stack([cx, cy, w, h], dim=1)

        elif isinstance(radm_output, list) and radm_output and 'instances' in radm_output[0]:
            # Real RADM output format (processed results)
            instances = radm_output[0]['instances']  # First image results
            boxes = instances.pred_boxes.tensor  # [num_instances, 4]
            classes = instances.pred_classes  # [num_instances]

            if len(boxes) == 0:
                # No instances detected, create some default elements for RL training
                boxes = torch.tensor([[0.5, 0.5, 0.3, 0.2], [0.3, 0.3, 0.2, 0.15]], device=self.device)
                classes = torch.tensor([0, 1], device=self.device)
            else:
                # Convert to normalized coordinates
                img_h, img_w = image_size
                boxes = boxes.clone()
                boxes[:, 0] /= img_w  # x1
                boxes[:, 1] /= img_h  # y1
                boxes[:, 2] /= img_w  # x2
                boxes[:, 3] /= img_h  # y2

                # Convert to cx, cy, w, h format
                cx = (boxes[:, 0] + boxes[:, 2]) / 2
                cy = (boxes[:, 1] + boxes[:, 3]) / 2
                w = boxes[:, 2] - boxes[:, 0]
                h = boxes[:, 3] - boxes[:, 1]
                boxes = torch.stack([cx, cy, w, h], dim=1)
        else:
            # Fallback: create some default elements
            print(f"Warning: Unknown RADM output format: {type(radm_output)}")
            boxes = torch.tensor([[0.5, 0.5, 0.3, 0.2], [0.3, 0.3, 0.2, 0.15]], device=self.device)
            classes = torch.tensor([0, 1], device=self.device)

        # Create layout elements
        for i, (bbox, class_id) in enumerate(zip(boxes, classes)):
            element = LayoutElement(
                element_id=i,
                bbox=bbox.clone(),
                class_id=class_id.item(),
                text_feature=text_feature.squeeze(0).clone()  # Remove batch dimension
            )
            self.elements[i] = element

        # Initialize neighbor relationships
        self._update_neighbor_relationships()

        # Return initial observations
        return self._get_observations()

    def step(self, actions: Dict[int, torch.Tensor]) -> Tuple[Dict, Dict, bool, Dict]:
        """
        Execute actions for all agents
        Args:
            actions: Dict of {agent_id: action_tensor}
        Returns:
            observations, rewards, done, info
        """
        self.step_count += 1

        # Store previous state for reward calculation
        prev_layout = {}
        for agent_id, element in self.elements.items():
            prev_layout[agent_id] = LayoutElement(
                element_id=element.id,
                bbox=element.bbox.clone(),
                class_id=element.class_id,
                text_feature=element.text_feature.clone()
            )
            prev_layout[agent_id].neighbors = element.neighbors.copy()
            prev_layout[agent_id].reward_history = element.reward_history.copy()

        # Execute actions
        for agent_id, action in actions.items():
            if agent_id in self.elements:
                self._execute_action(agent_id, action)

        # Update neighbor relationships
        self._update_neighbor_relationships()

        # Calculate rewards
        rewards = self._calculate_rewards(prev_layout)

        # Check termination
        done = self.step_count >= self.max_steps

        # Get observations
        observations = self._get_observations()

        # Additional info
        info = {
            'step_count': self.step_count,
            'num_elements': len(self.elements),
            'layout_quality': self._calculate_layout_quality()
        }

        return observations, rewards, done, info

    def _execute_action(self, agent_id: int, action: torch.Tensor):
        """Execute action for a single agent"""
        element = self.elements[agent_id]

        # Convert action to bbox changes
        dx, dy, dw, dh = action * self.action_scale

        # Update bbox
        new_bbox = element.bbox.clone()
        new_bbox[0] += dx  # x
        new_bbox[1] += dy  # y
        new_bbox[2] *= (1 + dw)  # w
        new_bbox[3] *= (1 + dh)  # h

        # Ensure bbox stays within bounds
        new_bbox = torch.clamp(new_bbox, 0, 1)

        element.update_bbox(new_bbox)

    def _update_neighbor_relationships(self):
        """Update neighbor relationships based on spatial proximity"""
        element_ids = list(self.elements.keys())
        if not element_ids:
            return
        boxes = torch.stack([self.elements[i].bbox for i in element_ids])

        # Calculate pairwise distances
        centers = boxes[:, :2] + boxes[:, 2:] / 2
        distances = torch.cdist(centers, centers)

        # Find k-nearest neighbors for each element
        k = min(5, len(element_ids) - 1)
        for i, elem_id in enumerate(element_ids):
            if len(element_ids) > 1:
                _, neighbor_indices = torch.topk(-distances[i], k=k+1, largest=True)
                neighbor_indices = neighbor_indices[1:]  # Exclude self
                self.elements[elem_id].neighbors = [element_ids[j] for j in neighbor_indices]

    def _calculate_rewards(self, prev_layout: Dict) -> Dict[int, float]:
        """Calculate rewards for all agents"""
        rewards = {}

        for agent_id, element in self.elements.items():
            reward = 0.0

            # Overlap penalty
            overlap_penalty = self._calculate_overlap_penalty(agent_id)
            reward += self.reward_weights['overlap_penalty'] * overlap_penalty

            # Alignment bonus
            alignment_bonus = self._calculate_alignment_bonus(agent_id)
            reward += self.reward_weights['alignment_bonus'] * alignment_bonus

            # Balance bonus
            balance_bonus = self._calculate_balance_bonus()
            reward += self.reward_weights['balance_bonus'] * balance_bonus

            # Semantic coherence reward
            semantic_reward = self._calculate_semantic_coherence(agent_id)
            reward += self.reward_weights['semantic_coherence'] * semantic_reward

            # Aesthetic score
            aesthetic_score = self._calculate_aesthetic_score(agent_id)
            reward += self.reward_weights['aesthetic_score'] * aesthetic_score

            rewards[agent_id] = reward
            element.reward_history.append(reward)

        return rewards

    def _calculate_overlap_penalty(self, agent_id: int) -> float:
        """Calculate overlap penalty for an element"""
        element = self.elements[agent_id]
        element_box = element.bbox

        overlap_area = 0.0
        for other_id, other_element in self.elements.items():
            if other_id != agent_id:
                other_box = other_element.bbox
                overlap = self._calculate_box_iou(element_box, other_box)
                overlap_area += overlap

        return overlap_area

    def _calculate_alignment_bonus(self, agent_id: int) -> float:
        """Calculate alignment bonus (edges, centers)"""
        element = self.elements[agent_id]
        bbox = element.bbox

        alignment_score = 0.0
        x, y, w, h = bbox

        # Edge alignment
        if abs(x - 0.0) < 0.05 or abs(x + w - 1.0) < 0.05:  # Left/Right edges
            alignment_score += 0.5
        if abs(y - 0.0) < 0.05 or abs(y + h - 1.0) < 0.05:  # Top/Bottom edges
            alignment_score += 0.5

        # Center alignment with neighbors
        center_x, center_y = x + w/2, y + h/2
        for neighbor_id in element.neighbors:
            neighbor = self.elements[neighbor_id]
            n_x, n_y, n_w, n_h = neighbor.bbox
            n_center_x, n_center_y = n_x + n_w/2, n_y + n_h/2

            if abs(center_x - n_center_x) < 0.1:  # Vertical alignment
                alignment_score += 0.3
            if abs(center_y - n_center_y) < 0.1:  # Horizontal alignment
                alignment_score += 0.3

        return alignment_score

    def _calculate_balance_bonus(self) -> float:
        """Calculate overall layout balance"""
        if not self.elements:
            return 0.0

        boxes = torch.stack([elem.bbox for elem in self.elements.values()])
        centers = boxes[:, :2] + boxes[:, 2:] / 2

        # Calculate center of mass
        center_of_mass = centers.mean(dim=0)
        image_center = torch.tensor([0.5, 0.5], device=self.device)

        # Distance from center of mass to image center
        balance_score = 1.0 / (1.0 + torch.norm(center_of_mass - image_center))

        return balance_score.item()

    def _calculate_semantic_coherence(self, agent_id: int) -> float:
        """Calculate semantic coherence with neighbors"""
        element = self.elements[agent_id]

        if not element.neighbors:
            return 0.0

        coherence_score = 0.0
        for neighbor_id in element.neighbors:
            neighbor = self.elements[neighbor_id]

            # Semantic similarity based on class proximity
            class_similarity = 1.0 if element.class_id == neighbor.class_id else 0.5

            # Text feature similarity
            text_sim = torch.cosine_similarity(
                element.text_feature.unsqueeze(0),
                neighbor.text_feature.unsqueeze(0),
                dim=-1
            ).squeeze()

            coherence_score += (class_similarity + text_sim) / 2.0

        return coherence_score / len(element.neighbors)

    def _calculate_aesthetic_score(self, agent_id: int) -> float:
        """Calculate aesthetic score (golden ratio, rule of thirds)"""
        element = self.elements[agent_id]
        bbox = element.bbox

        x, y, w, h = bbox
        aspect_ratio = w / h

        # Golden ratio preference
        golden_ratio = 1.618
        ratio_score = 1.0 / (1.0 + abs(aspect_ratio - golden_ratio))

        # Rule of thirds alignment
        thirds_score = 0.0
        center_x, center_y = x + w/2, y + h/2

        thirds_lines = [1/3, 2/3]
        for third in thirds_lines:
            if abs(center_x - third) < 0.1 or abs(center_y - third) < 0.1:
                thirds_score += 0.5

        return (ratio_score + thirds_score) / 2.0

    def _calculate_box_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """Calculate intersection over union of two boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Convert to xyxy format
        box1_xyxy = [x1, y1, x1+w1, y1+h1]
        box2_xyxy = [x2, y2, x2+w2, y2+h2]

        # Calculate intersection
        inter_x1 = max(box1_xyxy[0], box2_xyxy[0])
        inter_y1 = max(box1_xyxy[1], box2_xyxy[1])
        inter_x2 = min(box1_xyxy[2], box2_xyxy[2])
        inter_y2 = min(box1_xyxy[3], box2_xyxy[3])

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _get_observations(self) -> Dict[int, Dict]:
        """Get observations for all agents"""
        observations = {}

        for agent_id, element in self.elements.items():
            obs = element.get_state()

            # Add global context
            obs['global_text_feature'] = self.global_text_feature.squeeze(0).clone()  # Remove batch dimension
            obs['image_size'] = self.image_size
            obs['step_count'] = self.step_count

            # Add neighbor information
            neighbor_states = []
            for neighbor_id in element.neighbors:
                neighbor_state = self.elements[neighbor_id].get_state()
                neighbor_states.append(neighbor_state)

            obs['neighbor_states'] = neighbor_states
            observations[agent_id] = obs

        return observations

    def _calculate_layout_quality(self) -> Dict:
        """Calculate overall layout quality metrics"""
        if not self.elements:
            return {'r_shm': 0.0, 'balance': 0.0, 'alignment': 0.0}

        total_overlap = 0.0
        total_alignment = 0.0

        for agent_id in self.elements.keys():
            total_overlap += self._calculate_overlap_penalty(agent_id)
            total_alignment += self._calculate_alignment_bonus(agent_id)

        balance = self._calculate_balance_bonus()

        return {
            'r_shm': 1.0 / (1.0 + total_overlap / len(self.elements)),  # Simplified R_shm
            'balance': balance,
            'alignment': total_alignment / len(self.elements)
        }

    def get_layout_boxes(self) -> torch.Tensor:
        """Get current layout as tensor of boxes"""
        if not self.elements:
            return torch.empty(0, 4, device=self.device)

        boxes = []
        for element in self.elements.values():
            boxes.append(element.bbox)

        return torch.stack(boxes)

    def get_layout_classes(self) -> torch.Tensor:
        """Get current layout classes"""
        if not self.elements:
            return torch.empty(0, dtype=torch.long, device=self.device)

        classes = []
        for element in self.elements.values():
            classes.append(element.class_id)

        return torch.tensor(classes, device=self.device)
