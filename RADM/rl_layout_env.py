"""
RL Environment for Layout Generation and Optimization
基于RADM的强化学习环境设计，用于布局生成和优化的交互学习
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import gym
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class LayoutRLEnvironment(gym.Env):
    """
    布局生成强化学习环境

    状态空间: 当前布局配置 (boxes位置 + 文本特征 + 约束信息)
    动作空间: 布局元素的位置调整 (连续动作空间)
    奖励函数: 综合美学评分 + 约束满足度 + 可读性指标
    """

    def __init__(self,
                 max_boxes: int = 10,
                 canvas_size: Tuple[int, int] = (1, 1),
                 constraint_types: List[str] = None,
                 reward_weights: Dict[str, float] = None):
        """
        Args:
            max_boxes: 最大布局元素数量
            canvas_size: 画布尺寸 (归一化坐标 0-1)
            constraint_types: 支持的约束类型
            reward_weights: 奖励函数权重
        """
        super().__init__()

        self.max_boxes = max_boxes
        self.canvas_size = canvas_size
        self.constraint_types = constraint_types or ['left_of', 'right_of', 'above', 'below', 'full_width', 'full_height']

        # 默认奖励权重
        self.reward_weights = reward_weights or {
            'constraint': 1.0,      # 约束满足度
            'aesthetic': 0.8,       # 美学评分
            'readability': 0.6,     # 可读性
            'stability': 0.4,       # 稳定性奖励
            'progress': 0.2,        # 进度奖励
        }

        # 定义动作空间：每个box的xy位置调整 (连续动作)
        # 动作范围: [-0.1, 0.1] 表示每次最多移动10%的画布尺寸
        action_dim = max_boxes * 2  # x,y for each box
        self.action_space = spaces.Box(
            low=-0.1, high=0.1,
            shape=(action_dim,),
            dtype=np.float32
        )

        # 定义状态空间
        # 包括: boxes位置(4维), 约束状态, 历史动作等
        state_dim = self._calculate_state_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

        # 环境状态
        self.current_boxes = None
        self.target_boxes = None
        self.constraints = []
        self.text_features = None
        self.step_count = 0
        self.max_steps = 50  # 最大交互步数

        # 奖励计算器
        self.reward_calculator = LayoutRewardCalculator(self.reward_weights)

        # 初始化美学评估器
        self.aesthetic_evaluator = AestheticEvaluator()

    def _calculate_state_dim(self) -> int:
        """计算状态空间维度"""
        # boxes位置: max_boxes * 4 (x,y,w,h)
        # 约束状态: max_boxes * max_boxes (约束矩阵)
        # 历史信息: 一些统计特征
        boxes_dim = self.max_boxes * 4
        constraint_dim = self.max_boxes * self.max_boxes
        history_dim = 10  # 一些统计特征

        return boxes_dim + constraint_dim + history_dim

    def reset(self,
              initial_boxes: torch.Tensor,
              constraints: List[Dict],
              text_features: Optional[torch.Tensor] = None,
              target_boxes: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        重置环境状态

        Args:
            initial_boxes: 初始布局boxes [N, 4] (x,y,w,h)
            constraints: 布局约束列表
            text_features: 文本特征 [N, D]
            target_boxes: 目标布局 (用于监督学习)

        Returns:
            初始状态向量
        """
        self.current_boxes = initial_boxes.clone()
        self.constraints = constraints
        self.text_features = text_features
        self.target_boxes = target_boxes
        self.step_count = 0

        # 初始化历史轨迹
        self.action_history = []
        self.reward_history = []

        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步动作

        Args:
            action: 动作向量 [max_boxes * 2] 位置调整

        Returns:
            next_state, reward, done, info
        """
        self.step_count += 1

        # 解析动作：将连续动作转换为boxes位置调整
        action_tensor = torch.tensor(action, dtype=torch.float32)
        position_adjustments = action_tensor.view(self.max_boxes, 2)  # [N, 2] xy调整

        # 应用动作更新布局
        self._apply_action(position_adjustments)

        # 计算奖励
        reward, reward_info = self._calculate_reward()

        # 检查终止条件
        done = self._is_done()

        # 获取新状态
        next_state = self._get_state()

        # 记录历史
        self.action_history.append(action)
        self.reward_history.append(reward)

        info = {
            'reward_breakdown': reward_info,
            'step_count': self.step_count,
            'constraint_satisfaction': reward_info.get('constraint_score', 0),
            'aesthetic_score': reward_info.get('aesthetic_score', 0),
        }

        return next_state, reward, done, info

    def _apply_action(self, adjustments: torch.Tensor):
        """应用位置调整动作"""
        # 只调整前N个boxes的位置 (N = min(current_boxes.shape[0], max_boxes))
        n_boxes = min(self.current_boxes.shape[0], self.max_boxes)
        adjustments = adjustments[:n_boxes]

        # 更新位置，保持尺寸不变
        self.current_boxes[:n_boxes, :2] += adjustments

        # 边界裁剪
        self.current_boxes[:, :2].clamp_(0, 1)

    def _get_state(self) -> np.ndarray:
        """构建当前状态向量"""
        state_parts = []

        # 1. Boxes位置信息 (填充到max_boxes)
        boxes_state = torch.zeros(self.max_boxes, 4)
        n_boxes = min(self.current_boxes.shape[0], self.max_boxes)
        boxes_state[:n_boxes] = self.current_boxes[:n_boxes]
        state_parts.append(boxes_state.flatten())

        # 2. 约束状态矩阵
        constraint_matrix = self._build_constraint_matrix()
        state_parts.append(constraint_matrix.flatten())

        # 3. 历史统计特征
        history_features = self._extract_history_features()
        state_parts.append(history_features)

        return torch.cat(state_parts).numpy()

    def _build_constraint_matrix(self) -> torch.Tensor:
        """构建约束关系矩阵"""
        matrix = torch.zeros(self.max_boxes, self.max_boxes)

        for constraint in self.constraints:
            if 'source' in constraint and 'target' in constraint:
                source_idx = constraint['source']
                target_idx = constraint['target']
                if source_idx < self.max_boxes and target_idx < self.max_boxes:
                    # 根据约束类型设置矩阵值
                    ctype = constraint['type']
                    if ctype in ['left_of', 'right_of', 'above', 'below']:
                        matrix[source_idx, target_idx] = 1.0

        return matrix

    def _extract_history_features(self) -> torch.Tensor:
        """提取历史统计特征"""
        features = torch.zeros(10)

        if len(self.reward_history) > 0:
            features[0] = np.mean(self.reward_history[-5:])  # 最近5步平均奖励
            features[1] = np.std(self.reward_history[-5:])   # 奖励方差
            features[2] = len([r for r in self.reward_history if r > 0]) / len(self.reward_history)  # 正奖励比例

        # 布局统计特征
        if self.current_boxes is not None:
            centers = self.current_boxes[:, :2] + self.current_boxes[:, 2:] / 2
            features[3] = torch.mean(centers, dim=0)[0]  # x中心平均
            features[4] = torch.mean(centers, dim=0)[1]  # y中心平均
            features[5] = torch.std(centers, dim=0)[0]   # x分布方差
            features[6] = torch.std(centers, dim=0)[1]   # y分布方差

        features[7] = self.step_count / self.max_steps  # 进度
        features[8] = len(self.constraints)  # 约束数量
        features[9] = min(self.current_boxes.shape[0], self.max_boxes)  # 当前boxes数量

        return features

    def _calculate_reward(self) -> Tuple[float, Dict]:
        """计算综合奖励"""
        return self.reward_calculator(
            self.current_boxes,
            self.constraints,
            self.text_features,
            self.step_count,
            self.max_steps,
            self.target_boxes
        )

    def _is_done(self) -> bool:
        """检查是否终止"""
        # 达到最大步数
        if self.step_count >= self.max_steps:
            return True

        # 约束完全满足且奖励稳定 (可选)
        # if self._check_constraint_satisfaction() > 0.95 and self._check_reward_stability():
        #     return True

        return False

    def render(self, mode='human'):
        """可视化当前布局状态"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # 绘制画布
        ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.invert_yaxis()
        ax.set_aspect('equal')

        # 绘制boxes
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        if self.current_boxes is not None:
            for i, box in enumerate(self.current_boxes):
                x, y, w, h = box
                color = colors[i % len(colors)]
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor=color, alpha=0.5)
                ax.add_patch(rect)
                ax.text(x + w/2, y + h/2, f'{i}', ha='center', va='center', fontsize=12, weight='bold')

        # 绘制约束关系 (可选)
        self._draw_constraints(ax)

        plt.title(f'Layout Step {self.step_count}')
        plt.tight_layout()

        if mode == 'human':
            plt.show()
        elif mode == 'rgb_array':
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return img

        plt.close(fig)

    def _draw_constraints(self, ax):
        """绘制约束关系线"""
        if self.current_boxes is None:
            return

        for constraint in self.constraints:
            if 'source' in constraint and 'target' in constraint:
                source_idx = constraint['source']
                target_idx = constraint['target']

                if source_idx < len(self.current_boxes) and target_idx < len(self.current_boxes):
                    source_box = self.current_boxes[source_idx]
                    target_box = self.current_boxes[target_idx]

                    # 计算中心点
                    source_center = (source_box[0] + source_box[2]/2, source_box[1] + source_box[3]/2)
                    target_center = (target_box[0] + target_box[2]/2, target_box[1] + target_box[3]/2)

                    # 根据约束类型绘制不同颜色的线
                    ctype = constraint['type']
                    if ctype == 'left_of':
                        color, linestyle = 'red', '-'
                    elif ctype == 'right_of':
                        color, linestyle = 'blue', '-'
                    elif ctype == 'above':
                        color, linestyle = 'green', '--'
                    elif ctype == 'below':
                        color, linestyle = 'orange', '--'
                    else:
                        continue

                    ax.plot([source_center[0], target_center[0]],
                           [source_center[1], target_center[1]],
                           color=color, linestyle=linestyle, alpha=0.6, linewidth=2)


class LayoutRewardCalculator:
    """布局奖励计算器"""

    def __init__(self, weights: Dict[str, float]):
        self.weights = weights

    def __call__(self,
                 boxes: torch.Tensor,
                 constraints: List[Dict],
                 text_features: Optional[torch.Tensor],
                 step_count: int,
                 max_steps: int,
                 target_boxes: Optional[torch.Tensor] = None) -> Tuple[float, Dict]:

        reward_info = {}

        # 1. 约束满足度奖励
        constraint_score = self._calculate_constraint_score(boxes, constraints)
        reward_info['constraint_score'] = constraint_score

        # 2. 美学评分奖励
        aesthetic_score = self._calculate_aesthetic_score(boxes)
        reward_info['aesthetic_score'] = aesthetic_score

        # 3. 可读性奖励
        readability_score = self._calculate_readability_score(boxes, text_features)
        reward_info['readability_score'] = readability_score

        # 4. 稳定性奖励 (减少抖动)
        stability_score = self._calculate_stability_score(boxes)
        reward_info['stability_score'] = stability_score

        # 5. 进度奖励
        progress_score = step_count / max_steps
        reward_info['progress_score'] = progress_score

        # 6. 目标匹配奖励 (如果有目标布局)
        target_score = 0.0
        if target_boxes is not None:
            target_score = self._calculate_target_score(boxes, target_boxes)
        reward_info['target_score'] = target_score

        # 计算总奖励
        total_reward = (
            self.weights['constraint'] * constraint_score +
            self.weights['aesthetic'] * aesthetic_score +
            self.weights['readability'] * readability_score +
            self.weights['stability'] * stability_score +
            self.weights['progress'] * progress_score
        )

        return total_reward, reward_info

    def _calculate_constraint_score(self, boxes: torch.Tensor, constraints: List[Dict]) -> float:
        """计算约束满足度"""
        if not constraints:
            return 1.0

        total_score = 0.0
        for constraint in constraints:
            score = self._evaluate_single_constraint(boxes, constraint)
            total_score += score

        return total_score / len(constraints)

    def _evaluate_single_constraint(self, boxes: torch.Tensor, constraint: Dict) -> float:
        """评估单个约束的满足度"""
        ctype = constraint['type']

        if ctype in ['full_width', 'full_height']:
            # 绝对约束
            target_idx = constraint.get('target', 0)
            if target_idx >= len(boxes):
                return 0.0

            box = boxes[target_idx]
            if ctype == 'full_width':
                # 宽度接近1.0
                return 1.0 - abs(box[2] - 1.0)
            else:  # full_height
                return 1.0 - abs(box[3] - 1.0)

        elif 'source' in constraint and 'target' in constraint:
            # 相对约束
            source_idx = constraint['source']
            target_idx = constraint['target']

            if source_idx >= len(boxes) or target_idx >= len(boxes):
                return 0.0

            source_box = boxes[source_idx]
            target_box = boxes[target_idx]

            return self._evaluate_relative_constraint(source_box, target_box, ctype)

        return 0.0

    def _evaluate_relative_constraint(self, source_box: torch.Tensor, target_box: torch.Tensor, ctype: str) -> float:
        """评估相对约束满足度"""
        margin = 0.05  # 约束边距

        if ctype == 'left_of':
            # source在target左边
            required_gap = (target_box[0] - source_box[0] - source_box[2])
            return max(0, 1.0 - abs(required_gap - margin) / margin)
        elif ctype == 'right_of':
            # source在target右边
            required_gap = (source_box[0] - target_box[0] - target_box[2])
            return max(0, 1.0 - abs(required_gap - margin) / margin)
        elif ctype == 'above':
            # source在target上方
            required_gap = (target_box[1] - source_box[1] - source_box[3])
            return max(0, 1.0 - abs(required_gap - margin) / margin)
        elif ctype == 'below':
            # source在target下方
            required_gap = (source_box[1] - target_box[1] - target_box[3])
            return max(0, 1.0 - abs(required_gap - margin) / margin)

        return 0.0

    def _calculate_aesthetic_score(self, boxes: torch.Tensor) -> float:
        """计算美学评分"""
        if len(boxes) == 0:
            return 0.0

        scores = []

        # 1. 平衡性评分 (中心聚集度)
        centers = boxes[:, :2] + boxes[:, 2:] / 2
        center_mean = torch.mean(centers, dim=0)
        center_std = torch.std(centers, dim=0)
        balance_score = 1.0 / (1.0 + center_std.mean())  # 越集中分数越高
        scores.append(balance_score)

        # 2. 对齐评分 (边缘对齐度)
        left_align = 1.0 - torch.std(boxes[:, 0])  # 左对齐
        right_align = 1.0 - torch.std(boxes[:, 0] + boxes[:, 2])  # 右对齐
        top_align = 1.0 - torch.std(boxes[:, 1])  # 上对齐
        bottom_align = 1.0 - torch.std(boxes[:, 1] + boxes[:, 3])  # 下对齐

        alignment_score = (left_align + right_align + top_align + bottom_align) / 4.0
        scores.append(alignment_score)

        # 3. 比例和谐评分
        aspect_ratios = boxes[:, 2] / (boxes[:, 3] + 1e-6)
        golden_ratio = (1 + 5**0.5) / 2
        ratio_harmony = 1.0 - torch.mean(torch.abs(aspect_ratios - golden_ratio) / golden_ratio)
        scores.append(ratio_harmony)

        return torch.mean(torch.tensor(scores))

    def _calculate_readability_score(self, boxes: torch.Tensor, text_features: Optional[torch.Tensor]) -> float:
        """计算可读性评分"""
        if len(boxes) <= 1:
            return 1.0  # 单个元素默认可读

        # 1. 元素间距评分 (避免过于拥挤)
        centers = boxes[:, :2] + boxes[:, 2:] / 2
        min_distances = []
        for i in range(len(centers)):
            distances = torch.norm(centers[i] - centers, dim=1)
            distances = distances[distances > 0]  # 排除自己
            if len(distances) > 0:
                min_distances.append(torch.min(distances))

        if min_distances:
            avg_min_distance = torch.mean(torch.tensor(min_distances))
            spacing_score = torch.clamp(avg_min_distance * 4, 0, 1)  # 距离越大分数越高
        else:
            spacing_score = 1.0

        # 2. 大小对比评分 (避免元素过小)
        sizes = boxes[:, 2] * boxes[:, 3]
        size_score = torch.mean(torch.clamp(sizes * 10, 0, 1))  # 面积越大分数越高

        return (spacing_score + size_score) / 2.0

    def _calculate_stability_score(self, boxes: torch.Tensor) -> float:
        """计算稳定性评分 (减少布局抖动)"""
        # 这里可以基于历史轨迹计算，但简化起见返回常数
        # 在实际使用中，可以计算相邻步之间的位置变化
        return 0.5

    def _calculate_target_score(self, boxes: torch.Tensor, target_boxes: torch.Tensor) -> float:
        """计算与目标布局的匹配度"""
        if len(boxes) != len(target_boxes):
            return 0.0

        # 计算位置和尺寸的相似度
        position_diff = torch.mean(torch.abs(boxes[:, :2] - target_boxes[:, :2]))
        size_diff = torch.mean(torch.abs(boxes[:, 2:] - target_boxes[:, 2:]))

        # 转换为相似度评分
        position_score = 1.0 / (1.0 + position_diff)
        size_score = 1.0 / (1.0 + size_diff)

        return (position_score + size_score) / 2.0


class AestheticEvaluator:
    """美学评估器"""

    def __init__(self):
        # 可以集成更复杂的美学评估模型
        pass

    def evaluate(self, boxes: torch.Tensor) -> float:
        """评估布局的美学得分"""
        # 简化实现，实际可以集成专业的美学评估模型
        return 0.5
