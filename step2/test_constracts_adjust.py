"""
README:
This script is used to test the optimization of the layout of the boxes with constraints.
The boxes are initialized with random positions and random sizes.
The optimization is performed by minimizing the loss function.
The loss function is the sum of the following three terms:
1. The alignment loss of the bottom of the boxes.
2. The overlap loss of the boxes.
3. The boundary loss of the boxes.

Advantages:
1. we can control the relationship between the boxes.
2. we can control the fidelity of the layout.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class AdjustableLayoutOptimizer:
    def __init__(self, initial_boxes, constraints, alpha=0.0):
        """
        alpha: 调控因子 (0.0 ~ 100.0)
               0.0  = 自由优化 (Freedom)
               high = 强力锚定 (Rigidity)
        """
        # 1. 深度拷贝初始状态，作为“锚点”，不可训练
        self.initial_state = initial_boxes.clone().detach()
        
        # 2. 待优化参数
        self.boxes = nn.Parameter(initial_boxes.clone())
        
        self.constraints = constraints
        self.alpha = alpha # 用户指定的调控因子
        self.margin = 0.05

    def get_corners(self, xywh):
        x, y, w, h = xywh
        return x, y, x+w, y+h

    def calc_constraint_loss(self):
        """计算原本的约束 Loss (相对位置 + 绝对位置 + 物理约束)"""
        loss = torch.tensor(0.0)
        
        # --- A. 简化版的约束逻辑 (同前) ---
        for c in self.constraints:
            ctype = c['type']
            if 'source' in c:
                # 相对约束
                i, j = c['source'], c['target']
                xi_min, yi_min, xi_max, yi_max = self.get_corners(self.boxes[i])
                xj_min, yj_min, xj_max, yj_max = self.get_corners(self.boxes[j])
                
                if ctype == 'left_of': loss += torch.relu((xi_max + self.margin) - xj_min)
                elif ctype == 'right_of': loss += torch.relu((xj_max + self.margin) - xi_min)
                elif ctype == 'below': loss += torch.relu((yj_max + self.margin) - yi_min)
                # ... 其他约束省略 ...
            
            else:
                # 绝对/尺寸约束
                pass # 为演示简洁略过，逻辑同前文

        # 基础物理约束 (防重叠 + 边界)
        # 简单防重叠示例
        for i in range(len(self.boxes)):
            for j in range(i + 1, len(self.boxes)):
                xi1, yi1, xi2, yi2 = self.get_corners(self.boxes[i])
                xj1, yj1, xj2, yj2 = self.get_corners(self.boxes[j])
                
                inter_w = torch.relu(torch.min(xi2, xj2) - torch.max(xi1, xj1))
                inter_h = torch.relu(torch.min(yi2, yj2) - torch.max(yi1, yj1))
                loss += (inter_w * inter_h) * 10.0 # 重叠惩罚

        return loss

    def calc_fidelity_loss(self):
        """
        核心改动：计算保真度 Loss
        惩罚当前 boxes 与 initial_state 的距离
        """
        # 1. 位置差异 (x, y)
        pos_diff = (self.boxes[:, :2] - self.initial_state[:, :2]) ** 2
        # 2. 尺寸差异 (w, h)
        size_diff = (self.boxes[:, 2:] - self.initial_state[:, 2:]) ** 2
        
        # 求和
        return torch.sum(pos_diff + size_diff)

    def optimize(self, steps=500, lr=0.01):
        optimizer = optim.Adam([self.boxes], lr=lr)
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # 1. 计算必须满足的约束 Loss
            l_const = self.calc_constraint_loss()
            
            # 2. 计算保持原样的保真度 Loss
            l_fidelity = self.calc_fidelity_loss()
            
            # 3. 核心公式：总 Loss = 约束 + alpha * 锚定
            # 注意：如果 alpha 很大，优化器会尽量不动，除非 l_const 极大
            total_loss = l_const + self.alpha * l_fidelity
            
            total_loss.backward()
            optimizer.step()
            
        return self.boxes.detach()

# ==========================================
# 4. 可视化对比：不同 Alpha 的影响
# ==========================================

# 场景：Box 0 (左边) 和 Box 1 (右边) 发生了重叠
# 用户约束：Box 0 必须在 Box 1 左边 (left_of)
# 初始状态：Box 0 在 0.5, Box 1 在 0.4 (严重违规，且重叠)

init_boxes = torch.tensor([
    [0.5, 0.4, 0.2, 0.2], # Box 0 (原本在右边)
    [0.4, 0.4, 0.2, 0.2], # Box 1 (原本在左边)
])

constraints = [{'type': 'left_of', 'source': 0, 'target': 1}]

# 实验不同的 alpha
alphas = [0.0, 1.0, 10.0]
results = []

for a in alphas:
    opt = AdjustableLayoutOptimizer(init_boxes, constraints, alpha=a)
    res = opt.optimize()
    results.append((a, res))

# 绘图逻辑 (略微简化)
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# 1. 初始状态
axes[0].set_title("Initial State\n(Violates '0 left of 1')")
axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1); axes[0].invert_yaxis()
for i, b in enumerate(init_boxes):
    axes[0].add_patch(patches.Rectangle((b[0], b[1]), b[2], b[3], ec='black', fc=['red','blue'][i], alpha=0.5))
    axes[0].text(b[0], b[1], f"{i}", fontweight='bold')

# 2. 不同 Alpha 的结果
for idx, (alpha, res_boxes) in enumerate(results):
    ax = axes[idx+1]
    ax.set_title(f"Alpha = {alpha}\n(Fidelity: {'None' if alpha==0 else 'High'})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.invert_yaxis()
    
    # 画初始位置的虚线框作为参考
    for b in init_boxes:
        ax.add_patch(patches.Rectangle((b[0], b[1]), b[2], b[3], ec='gray', fill=False, ls='--', alpha=0.3))

    # 画优化后的位置
    for i, b in enumerate(res_boxes):
        ax.add_patch(patches.Rectangle((b[0], b[1]), b[2], b[3], ec='black', fc=['red','blue'][i], alpha=0.6))
        ax.text(b[0], b[1], f"{i}", fontweight='bold')

plt.tight_layout()
plt.savefig('test_constracts_adjust.png')
# plt.show()