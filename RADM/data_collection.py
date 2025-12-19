"""
数据收集和偏好学习模块
扩展RADM数据集，添加RL交互数据和人类偏好学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import json
import os
from pathlib import Path
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class LayoutPreferenceDataset:
    """
    布局偏好数据集
    收集人类或自动评估器的布局偏好数据
    """

    def __init__(self, data_dir: str = './data/preferences'):
        """
        Args:
            data_dir: 偏好数据存储目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 数据存储
        self.preferences = []  # [(layout_a, layout_b, preference)] 1表示a更好，-1表示b更好，0表示相等
        self.layouts = []      # 存储所有布局
        self.metadata = {}     # 元数据

    def add_preference(self,
                      layout_a: torch.Tensor,
                      layout_b: torch.Tensor,
                      preference: int,
                      constraints: List[Dict] = None,
                      text_desc: str = "",
                      metadata: Dict = None):
        """
        添加偏好数据

        Args:
            layout_a: 布局A [N, 4]
            layout_b: 布局B [N, 4]
            preference: 偏好值 (1: A更好, -1: B更好, 0: 相等)
            constraints: 布局约束
            text_desc: 文本描述
            metadata: 额外元数据
        """
        preference_data = {
            'layout_a_idx': len(self.layouts),
            'layout_b_idx': len(self.layouts) + 1,
            'preference': preference,
            'constraints': constraints or [],
            'text_desc': text_desc,
            'metadata': metadata or {},
            'timestamp': torch.tensor(0.0)  # 可以添加时间戳
        }

        self.preferences.append(preference_data)
        self.layouts.extend([layout_a.clone(), layout_b.clone()])

    def add_auto_preference(self,
                           layout: torch.Tensor,
                           optimized_layout: torch.Tensor,
                           constraints: List[Dict],
                           evaluator=None):
        """
        自动生成偏好数据 (通过评估器)

        Args:
            layout: 原始布局
            optimized_layout: 优化后布局
            constraints: 约束条件
            evaluator: 评估器函数
        """
        if evaluator is None:
            # 默认评估器：基于约束满足度
            score_orig = self._evaluate_layout_basic(layout, constraints)
            score_opt = self._evaluate_layout_basic(optimized_layout, constraints)

            if score_opt > score_orig + 0.1:  # 优化明显更好
                preference = 1  # 优化后更好
            elif score_orig > score_opt + 0.1:  # 原始更好 (罕见)
                preference = -1
            else:
                preference = 0  # 相等
        else:
            # 使用自定义评估器
            score_orig = evaluator(layout, constraints)
            score_opt = evaluator(optimized_layout, constraints)

            if score_opt > score_orig:
                preference = 1
            elif score_orig > score_opt:
                preference = -1
            else:
                preference = 0

        self.add_preference(
            layout, optimized_layout, preference,
            constraints, "auto_generated"
        )

    def _evaluate_layout_basic(self, layout: torch.Tensor, constraints: List[Dict]) -> float:
        """基础布局评估"""
        score = 0.0

        # 检查约束满足度
        for constraint in constraints:
            if self._check_constraint(layout, constraint):
                score += 1.0

        # 归一化
        return score / max(len(constraints), 1)

    def _check_constraint(self, layout: torch.Tensor, constraint: Dict) -> bool:
        """检查单个约束是否满足"""
        ctype = constraint['type']

        if 'source' in constraint and 'target' in constraint:
            source_idx = constraint['source']
            target_idx = constraint['target']

            if source_idx >= len(layout) or target_idx >= len(layout):
                return False

            source_box = layout[source_idx]
            target_box = layout[target_idx]

            if ctype == 'left_of':
                return source_box[0] + source_box[2] < target_box[0]
            elif ctype == 'above':
                return source_box[1] + source_box[3] < target_box[1]

        return True

    def save(self, filename: str = 'preferences.json'):
        """保存偏好数据"""
        data = {
            'preferences': self.preferences,
            'layouts': [layout.tolist() for layout in self.layouts],
            'metadata': self.metadata
        }

        with open(self.data_dir / filename, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filename: str = 'preferences.json'):
        """加载偏好数据"""
        filepath = self.data_dir / filename
        if not filepath.exists():
            return

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.preferences = data['preferences']
        self.layouts = [torch.tensor(layout) for layout in data['layouts']]
        self.metadata = data.get('metadata', {})

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        if not self.preferences:
            return {}

        preferences = np.array([p['preference'] for p in self.preferences])
        unique_prefs, counts = np.unique(preferences, return_counts=True)

        return {
            'total_preferences': len(self.preferences),
            'preference_distribution': dict(zip(unique_prefs.tolist(), counts.tolist())),
            'total_layouts': len(self.layouts),
            'avg_layouts_per_preference': len(self.layouts) / max(len(self.preferences), 1)
        }

    def sample_preference_batch(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样偏好批次数据

        Returns:
            layout_a_batch, layout_b_batch, preference_batch
        """
        if len(self.preferences) < batch_size:
            batch_size = len(self.preferences)

        indices = random.sample(range(len(self.preferences)), batch_size)

        layout_a_list = []
        layout_b_list = []
        preference_list = []

        for idx in indices:
            pref = self.preferences[idx]
            layout_a = self.layouts[pref['layout_a_idx']]
            layout_b = self.layouts[pref['layout_b_idx']]
            preference = pref['preference']

            layout_a_list.append(layout_a)
            layout_b_list.append(layout_b)
            preference_list.append(preference)

        return (
            torch.stack(layout_a_list),
            torch.stack(layout_b_list),
            torch.tensor(preference_list, dtype=torch.long)
        )


class PreferenceLearningModel(nn.Module):
    """
    偏好学习模型
    学习从布局特征预测人类偏好
    """

    def __init__(self,
                 layout_feature_dim: int = 256,
                 hidden_dim: int = 128,
                 num_heads: int = 8):
        """
        Args:
            layout_feature_dim: 布局特征维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
        """
        super().__init__()

        # 布局编码器
        self.layout_encoder = nn.Sequential(
            nn.Linear(4, layout_feature_dim // 4),  # x,y,w,h -> 特征
            nn.ReLU(),
            nn.Linear(layout_feature_dim // 4, layout_feature_dim),
            nn.ReLU()
        )

        # 交叉注意力 (比较两个布局)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=layout_feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # 偏好预测器
        self.preference_predictor = nn.Sequential(
            nn.Linear(layout_feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 3类: A更好, B更好, 相等
            nn.Softmax(dim=-1)
        )

        # 质量评估器 (回归头)
        self.quality_predictor = nn.Sequential(
            nn.Linear(layout_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 质量分数
            nn.Sigmoid()
        )

    def forward(self,
                layout_a: torch.Tensor,
                layout_b: torch.Tensor,
                constraints: Optional[List[Dict]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            layout_a: 布局A [B, N, 4]
            layout_b: 布局B [B, N, 4]
            constraints: 约束信息 (可选)

        Returns:
            preference_logits, quality_a, quality_b
        """

        # 编码布局
        features_a = self.layout_encoder(layout_a)  # [B, N, D]
        features_b = self.layout_encoder(layout_b)  # [B, N, D]

        # 全局池化得到布局级特征
        layout_feat_a = features_a.mean(dim=1)  # [B, D]
        layout_feat_b = features_b.mean(dim=1)  # [B, D]

        # 交叉注意力比较
        # 将A作为Query，B作为Key/Value
        attn_output, _ = self.cross_attention(
            layout_feat_a.unsqueeze(1),  # [B, 1, D]
            layout_feat_b.unsqueeze(1),  # [B, 1, D]
            layout_feat_b.unsqueeze(1)   # [B, 1, D]
        )
        attn_output = attn_output.squeeze(1)  # [B, D]

        # 反向注意力 (B作为Query)
        attn_output_rev, _ = self.cross_attention(
            layout_feat_b.unsqueeze(1),
            layout_feat_a.unsqueeze(1),
            layout_feat_a.unsqueeze(1)
        )
        attn_output_rev = attn_output_rev.squeeze(1)  # [B, D]

        # 拼接注意力特征
        combined_features = torch.cat([attn_output, attn_output_rev], dim=-1)  # [B, 2*D]

        # 预测偏好
        preference_logits = self.preference_predictor(combined_features)  # [B, 3]

        # 预测质量分数
        quality_a = self.quality_predictor(layout_feat_a)  # [B, 1]
        quality_b = self.quality_predictor(layout_feat_b)  # [B, 1]

        return preference_logits, quality_a, quality_b

    def predict_preference(self,
                          layout_a: torch.Tensor,
                          layout_b: torch.Tensor) -> Tuple[int, float]:
        """
        预测哪个布局更好

        Returns:
            preference (1: A更好, -1: B更好, 0: 相等), confidence
        """
        with torch.no_grad():
            logits, _, _ = self.forward(layout_a.unsqueeze(0), layout_b.unsqueeze(0))
            probs = logits.squeeze(0)
            pred_class = torch.argmax(probs).item()

            # 转换为偏好值
            if pred_class == 0:
                preference = 1   # A更好
            elif pred_class == 1:
                preference = -1  # B更好
            else:
                preference = 0   # 相等

            confidence = probs[pred_class].item()

            return preference, confidence


class InteractiveDataCollector:
    """
    交互式数据收集器
    允许用户实时提供偏好反馈
    """

    def __init__(self,
                 preference_dataset: LayoutPreferenceDataset,
                 visualization_dir: str = './visualizations'):
        """
        Args:
            preference_dataset: 偏好数据集
            visualization_dir: 可视化输出目录
        """
        self.dataset = preference_dataset
        self.vis_dir = Path(visualization_dir)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

    def collect_user_preference(self,
                               layout_a: torch.Tensor,
                               layout_b: torch.Tensor,
                               constraints: List[Dict],
                               text_desc: str = "",
                               show_visualization: bool = True) -> int:
        """
        收集用户偏好反馈

        Args:
            layout_a: 布局A
            layout_b: 布局B
            constraints: 约束条件
            text_desc: 文本描述
            show_visualization: 是否显示可视化

        Returns:
            用户偏好 (1: A更好, -1: B更好, 0: 相等)
        """

        if show_visualization:
            self._visualize_comparison(layout_a, layout_b, constraints, text_desc)

        # 简单的命令行交互 (实际应用中可以使用GUI)
        print("\n请选择哪个布局更好:")
        print("1: 左侧布局 (Layout A) 更好")
        print("2: 右侧布局 (Layout B) 更好")
        print("0: 两个布局差不多")

        while True:
            try:
                choice = int(input("请输入选择 (0/1/2): "))
                if choice in [0, 1, 2]:
                    break
            except ValueError:
                pass
            print("输入无效，请重新输入")

        # 转换为偏好值
        if choice == 1:
            preference = 1   # A更好
        elif choice == 2:
            preference = -1  # B更好
        else:
            preference = 0   # 相等

        # 添加到数据集
        self.dataset.add_preference(
            layout_a, layout_b, preference,
            constraints, text_desc
        )

        return preference

    def _visualize_comparison(self,
                             layout_a: torch.Tensor,
                             layout_b: torch.Tensor,
                             constraints: List[Dict],
                             text_desc: str):
        """可视化布局比较"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

        # 绘制布局A
        ax1.set_title(f'Layout A\n{text_desc}', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.invert_yaxis()
        ax1.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2))

        for i, box in enumerate(layout_a):
            x, y, w, h = box
            color = colors[i % len(colors)]
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor=color, alpha=0.6)
            ax1.add_patch(rect)
            ax1.text(x + w/2, y + h/2, f'{i}', ha='center', va='center', fontsize=12, weight='bold', color='white')

        # 绘制布局B
        ax2.set_title(f'Layout B\n{text_desc}', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.invert_yaxis()
        ax2.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2))

        for i, box in enumerate(layout_b):
            x, y, w, h = box
            color = colors[i % len(colors)]
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor=color, alpha=0.6)
            ax2.add_patch(rect)
            ax2.text(x + w/2, y + h/2, f'{i}', ha='center', va='center', fontsize=12, weight='bold', color='white')

        # 绘制约束关系
        self._draw_constraints(ax1, layout_a, constraints, alpha=0.3)
        self._draw_constraints(ax2, layout_b, constraints, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.vis_dir / f'comparison_{len(self.dataset.preferences)}.png', dpi=150, bbox_inches='tight')
        plt.show()

    def _draw_constraints(self, ax, layout: torch.Tensor, constraints: List[Dict], alpha: float = 0.6):
        """绘制约束关系"""
        for constraint in constraints:
            if 'source' in constraint and 'target' in constraint:
                source_idx = constraint['source']
                target_idx = constraint['target']

                if source_idx < len(layout) and target_idx < len(layout):
                    source_box = layout[source_idx]
                    target_box = layout[target_idx]

                    source_center = (source_box[0] + source_box[2]/2, source_box[1] + source_box[3]/2)
                    target_center = (target_box[0] + target_box[2]/2, target_box[1] + target_box[3]/2)

                    ctype = constraint['type']
                    if ctype == 'left_of':
                        color = 'red'
                    elif ctype == 'right_of':
                        color = 'blue'
                    elif ctype == 'above':
                        color = 'green'
                    elif ctype == 'below':
                        color = 'orange'
                    else:
                        continue

                    ax.plot([source_center[0], target_center[0]],
                           [source_center[1], target_center[1]],
                           color=color, alpha=alpha, linewidth=2, linestyle='--')


class AutomatedPreferenceGenerator:
    """
    自动偏好生成器
    使用启发式规则自动生成偏好数据
    """

    def __init__(self, preference_dataset: LayoutPreferenceDataset):
        self.dataset = preference_dataset
        self.rules = [
            self.rule_constraint_satisfaction,
            self.rule_aesthetic_balance,
            self.rule_alignment_quality,
            self.rule_readability_score
        ]

    def generate_batch_preferences(self,
                                 layout_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
                                 constraints_list: List[List[Dict]],
                                 batch_size: int = 100):
        """
        批量生成偏好数据

        Args:
            layout_pairs: 布局对列表 [(layout_a, layout_b), ...]
            constraints_list: 对应的约束列表
            batch_size: 生成批次大小
        """
        generated_count = 0

        for layout_a, layout_b in layout_pairs:
            if generated_count >= batch_size:
                break

            # 为每个布局对尝试不同规则
            for rule_func in self.rules:
                preference, confidence = rule_func(layout_a, layout_b, constraints_list[generated_count % len(constraints_list)])

                if abs(preference) > 0 and confidence > 0.7:  # 只保留高置信度的偏好
                    self.dataset.add_preference(
                        layout_a, layout_b, preference,
                        constraints_list[generated_count % len(constraints_list)],
                        f"auto_rule_{rule_func.__name__}"
                    )
                    generated_count += 1
                    break

    def rule_constraint_satisfaction(self,
                                   layout_a: torch.Tensor,
                                   layout_b: torch.Tensor,
                                   constraints: List[Dict]) -> Tuple[int, float]:
        """基于约束满足度的规则"""
        score_a = self._calculate_constraint_score(layout_a, constraints)
        score_b = self._calculate_constraint_score(layout_b, constraints)

        if score_a > score_b + 0.2:
            return 1, min(1.0, (score_a - score_b) / 0.5)  # A更好
        elif score_b > score_a + 0.2:
            return -1, min(1.0, (score_b - score_a) / 0.5)  # B更好
        else:
            return 0, 0.5  # 相等

    def rule_aesthetic_balance(self,
                              layout_a: torch.Tensor,
                              layout_b: torch.Tensor,
                              constraints: List[Dict]) -> Tuple[int, float]:
        """基于美学平衡的规则"""
        balance_a = self._calculate_balance_score(layout_a)
        balance_b = self._calculate_balance_score(layout_b)

        if balance_a > balance_b + 0.1:
            return 1, min(1.0, (balance_a - balance_b) / 0.3)
        elif balance_b > balance_a + 0.1:
            return -1, min(1.0, (balance_b - balance_a) / 0.3)
        else:
            return 0, 0.5

    def rule_alignment_quality(self,
                             layout_a: torch.Tensor,
                             layout_b: torch.Tensor,
                             constraints: List[Dict]) -> Tuple[int, float]:
        """基于对齐质量的规则"""
        align_a = self._calculate_alignment_score(layout_a)
        align_b = self._calculate_alignment_score(layout_b)

        if align_a > align_b + 0.15:
            return 1, min(1.0, (align_a - align_b) / 0.4)
        elif align_b > align_a + 0.15:
            return -1, min(1.0, (align_b - align_a) / 0.4)
        else:
            return 0, 0.5

    def rule_readability_score(self,
                             layout_a: torch.Tensor,
                             layout_b: torch.Tensor,
                             constraints: List[Dict]) -> Tuple[int, float]:
        """基于可读性得分的规则"""
        read_a = self._calculate_readability_score(layout_a)
        read_b = self._calculate_readability_score(layout_b)

        if read_a > read_b + 0.1:
            return 1, min(1.0, (read_a - read_b) / 0.3)
        elif read_b > read_a + 0.1:
            return -1, min(1.0, (read_b - read_a) / 0.3)
        else:
            return 0, 0.5

    # 辅助计算函数
    def _calculate_constraint_score(self, layout: torch.Tensor, constraints: List[Dict]) -> float:
        """计算约束满足度"""
        score = 0.0
        for constraint in constraints:
            if self._check_constraint(layout, constraint):
                score += 1.0
        return score / max(len(constraints), 1)

    def _calculate_balance_score(self, layout: torch.Tensor) -> float:
        """计算平衡得分"""
        if len(layout) == 0:
            return 0.0
        centers = layout[:, :2] + layout[:, 2:] / 2
        center_std = torch.std(centers, dim=0).mean()
        return 1.0 / (1.0 + center_std)

    def _calculate_alignment_score(self, layout: torch.Tensor) -> float:
        """计算对齐得分"""
        if len(layout) <= 1:
            return 1.0

        left_align = 1.0 - torch.std(layout[:, 0])
        right_align = 1.0 - torch.std(layout[:, 0] + layout[:, 2])
        top_align = 1.0 - torch.std(layout[:, 1])
        bottom_align = 1.0 - torch.std(layout[:, 1] + layout[:, 3])

        return (left_align + right_align + top_align + bottom_align) / 4.0

    def _calculate_readability_score(self, layout: torch.Tensor) -> float:
        """计算可读性得分"""
        if len(layout) <= 1:
            return 1.0

        centers = layout[:, :2] + layout[:, 2:] / 2
        sizes = layout[:, 2] * layout[:, 3]

        # 计算最小距离
        min_distances = []
        for i in range(len(centers)):
            distances = torch.norm(centers[i] - centers, dim=1)
            distances = distances[distances > 0]
            if len(distances) > 0:
                min_distances.append(torch.min(distances))

        if min_distances:
            spacing_score = torch.mean(torch.tensor(min_distances)) * 2.0
            spacing_score = torch.clamp(spacing_score, 0, 1)
        else:
            spacing_score = 1.0

        size_score = torch.mean(torch.clamp(sizes * 5, 0, 1))

        return (spacing_score + size_score) / 2.0

    def _check_constraint(self, layout: torch.Tensor, constraint: Dict) -> bool:
        """检查约束是否满足"""
        ctype = constraint['type']
        if 'source' in constraint and 'target' in constraint:
            source_idx = constraint['source']
            target_idx = constraint['target']

            if source_idx >= len(layout) or target_idx >= len(layout):
                return False

            source_box = layout[source_idx]
            target_box = layout[target_idx]

            if ctype == 'left_of':
                return source_box[0] + source_box[2] < target_box[0]
            elif ctype == 'above':
                return source_box[1] + source_box[3] < target_box[1]

        return True
