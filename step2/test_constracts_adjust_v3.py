"""
README:
This script is used to test the optimization of the layout of the boxes with constraints.
The boxes are initialized with random positions and fixed sizes.
The optimization is performed by minimizing the loss function.
The loss function is the sum of the following three terms:
1. The alignment loss of the bottom of the boxes.
2. The overlap loss of the boxes.
3. The boundary loss of the boxes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class DynamicBarrierFixedSizeOptimizer:
    def __init__(self, initial_boxes, constraints, fidelity_alpha=1.0, threshold=1e-4):
        """
        Modified Version: Fixed Size (w, h), Variable Position (x, y)
        """
        # 1. 记录初始位置 (仅记录 xy 用于保真度计算)
        self.initial_xy = initial_boxes[:, :2].clone().detach()
        
        # 2. 拆分参数：
        # self.xy -> 是参数，需要梯度，可以被修改
        # self.wh -> 是常量，不需要梯度，固定不变
        self.xy = nn.Parameter(initial_boxes[:, :2].clone())
        self.wh = initial_boxes[:, 2:].clone().detach()
        
        self.constraints = constraints
        self.alpha = fidelity_alpha
        self.threshold = threshold
        self.margin = 0.05

        # 记录历史
        self.history = {'boxes': [], 'loss_c': [], 'loss_f': []}

    @property
    def boxes(self):
        """
        动态属性：每次调用时，将当前优化的 xy 和固定的 wh 拼接起来。
        这使得外部调用看起来 self.boxes 依然是一个完整的 Tensor。
        """
        return torch.cat([self.xy, self.wh], dim=1)

    def get_corners(self, xywh):
        x, y, w, h = xywh
        return x, y, x+w, y+h

    def calc_constraint_loss(self):
        loss = torch.tensor(0.0)
        
        # 获取当前的完整 boxes (包含最新的 x,y 和固定的 w,h)
        current_boxes = self.boxes

        # --- A. 逻辑约束 ---
        for c in self.constraints:
            ctype = c['type']
            if 'source' in c:
                i, j = c['source'], c['target']
                xi_min, yi_min, xi_max, yi_max = self.get_corners(current_boxes[i])
                xj_min, yj_min, xj_max, yj_max = self.get_corners(current_boxes[j])
                
                if ctype == 'left_of':   val = (xi_max + self.margin) - xj_min
                elif ctype == 'right_of': val = (xj_max + self.margin) - xi_min
                elif ctype == 'above':    val = (yi_max + self.margin) - yj_min
                elif ctype == 'below':    val = (yj_max + self.margin) - yi_min
                else: val = torch.tensor(0.0)
                
                loss += torch.relu(val)

        # --- B. 物理约束 (防重叠) ---
        n = len(current_boxes)
        for i in range(n):
            for j in range(i + 1, n):
                xi_min, yi_min, xi_max, yi_max = self.get_corners(current_boxes[i])
                xj_min, yj_min, xj_max, yj_max = self.get_corners(current_boxes[j])
                
                # 计算重叠面积
                inter_w = torch.relu(torch.min(xi_max, xj_max) - torch.max(xi_min, xj_min))
                inter_h = torch.relu(torch.min(yi_max, yj_max) - torch.max(yi_min, yj_min))
                loss += inter_w * inter_h * 10.0

        # --- C. 边界约束 ---
        # 确保 x, y 不跑出画布 (0,0) -> (1,1)
        # 注意：这里我们不需要约束 w, h > 0，因为 w,h 是固定的且初始值应该就是合法的
        l_bound = (torch.relu(-current_boxes[:,0]) + torch.relu(current_boxes[:,0]+current_boxes[:,2]-1) + 
                   torch.relu(-current_boxes[:,1]) + torch.relu(current_boxes[:,1]+current_boxes[:,3]-1)).sum()
        
        return loss + l_bound * 10.0

    def calc_fidelity_loss(self):
        """
        计算保真度 Loss
        修改点：只计算 xy 的偏移量 (MSE)，忽略 wh (因为 wh 根本不会变)
        """
        return torch.mean((self.xy - self.initial_xy) ** 2)

    def optimize(self, steps=500, lr=0.01):
        # 修改点：优化器只接收 self.xy，不接收 self.wh
        optimizer = optim.Adam([self.xy], lr=lr)
        
        print(f"开始优化 (Fixed Size): Alpha = {self.alpha}")
        
        for step in range(steps):
            optimizer.zero_grad()
            
            l_const = self.calc_constraint_loss()
            l_fid = self.calc_fidelity_loss()
            
            # 动态 Barrier 逻辑保持不变
            if l_const > self.threshold:
                w_c = 1000.0
                w_f = 0.0
                state_msg = "VIOLATION"
            else:
                w_c = 100.0
                w_f = self.alpha
                state_msg = "FEASIBLE"
                
            total_loss = w_c * l_const + w_f * l_fid
            
            total_loss.backward()
            optimizer.step()
            
            # 记录数据 (需要 detach 并转 numpy)
            current_snapshot = self.boxes.detach().numpy().copy()
            self.history['boxes'].append(current_snapshot)
            self.history['loss_c'].append(l_const.item())
            self.history['loss_f'].append(l_fid.item())
            
            if step % 50 == 0:
                print(f"Step {step:03d} [{state_msg}] | Loss_C: {l_const.item():.6f} | Loss_F: {l_fid.item():.6f}")

        return self.boxes.detach()

# ==========================================
# 测试与可视化
# ==========================================

# 场景：两个矩形互换位置，但这次它们的形状(宽高)必须保持绝对不变
# A (Red):  瘦高 (w=0.1, h=0.3)
# B (Blue): 矮胖 (w=0.3, h=0.1)
init_boxes = torch.tensor([
    [0.2, 0.35, 0.1, 0.3], # Box A
    [0.8, 0.45, 0.3, 0.1], # Box B
])

# 强约束：A 必须跑到 B 的右边
constraints = [{'type': 'right_of', 'source': 0, 'target': 1}]

# 实例化新的固定尺寸优化器
opt = DynamicBarrierFixedSizeOptimizer(init_boxes, constraints, fidelity_alpha=5.0)
final_boxes = opt.optimize(steps=1000)

# 复用之前的可视化代码
def visualize_trajectory(optimizer, title):
    history = np.array(optimizer.history['boxes'])
    loss_c = optimizer.history['loss_c']
    loss_f = optimizer.history['loss_f']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 轨迹图
    ax1.set_title(title)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.invert_yaxis()
    ax1.add_patch(patches.Rectangle((0,0),1,1, fill=False, ec='black'))
    
    colors = ['red', 'blue']
    labels = ['A (Tall)', 'B (Wide)'] # 标注特征
    
    # 起点
    initial = history[0]
    for i in range(len(initial)):
        b = initial[i]
        ax1.add_patch(patches.Rectangle((b[0], b[1]), b[2], b[3], 
                                        ec=colors[i], fill=False, ls='--', alpha=0.5, label=f'{labels[i]} Start'))
    # 终点
    final = history[-1]
    for i in range(len(final)):
        b = final[i]
        ax1.add_patch(patches.Rectangle((b[0], b[1]), b[2], b[3], ec='black', fc=colors[i], alpha=0.6, label=f'{labels[i]} End'))
        ax1.text(b[0]+b[2]/2, b[1]+b[3]/2, labels[i], color='white', ha='center', va='center', fontweight='bold', fontsize=8)

    # 轨迹
    for i in range(len(initial)):
        cx = history[:, i, 0] + history[:, i, 2]/2
        cy = history[:, i, 1] + history[:, i, 3]/2
        ax1.plot(cx, cy, color=colors[i], alpha=0.3, linewidth=2)
        ax1.scatter(cx[0], cy[0], c=colors[i], marker='o', s=30)
        ax1.scatter(cx[-1], cy[-1], c=colors[i], marker='x', s=50)

    ax1.legend(loc='upper right')

    # Loss图
    ax2.set_title("Dynamics: Position Only Optimization")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Loss")
    ax2_r = ax2.twinx()
    l1 = ax2.plot(loss_c, color='orange', label='Constraint Loss', linewidth=2)
    l2 = ax2_r.plot(loss_f, color='green', label='Fidelity Loss (XY only)', linewidth=2, linestyle='--')
    ax2.set_yscale('log'); ax2.set_ylim(bottom=1e-5)
    ax2.legend(l1+l2, [l.get_label() for l in l1+l2])
    
    plt.tight_layout()
    plt.savefig('test_constracts_adjust_v3.png')
    # plt.show()

visualize_trajectory(opt, "Fixed Size Optimization\n(Moving Blocks while Preserving Aspect Ratio)")