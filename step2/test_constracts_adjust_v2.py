"""
README:
This script is used to test the optimization of the layout of the boxes with constraints.
The boxes are initialized with random positions and random sizes.
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

class DynamicBarrierOptimizer:
    def __init__(self, initial_boxes, constraints, fidelity_alpha=1.0, threshold=1e-4):
        """
        fidelity_alpha: 也就是用户指定的调控因子，用于在满足约束后，控制“回弹”的力度。
        threshold: 判定约束是否满足的阈值 (epsilon)
        """
        # 1. 冻结初始状态 (Anchor)
        self.initial_state = initial_boxes.clone().detach()
        
        # 2. 待优化参数
        self.boxes = nn.Parameter(initial_boxes.clone())
        
        self.constraints = constraints
        self.alpha = fidelity_alpha
        self.threshold = threshold
        self.margin = 0.05

        # 记录历史用于可视化
        self.history = {'boxes': [], 'loss_c': [], 'loss_f': []}

    def get_corners(self, xywh):
        x, y, w, h = xywh
        return x, y, x+w, y+h

    def calc_constraint_loss(self):
        """只计算硬约束违规量"""
        loss = torch.tensor(0.0)
        
        # --- A. 逻辑约束 ---
        for c in self.constraints:
            ctype = c['type']
            if 'source' in c:
                i, j = c['source'], c['target']
                xi_min, yi_min, xi_max, yi_max = self.get_corners(self.boxes[i])
                xj_min, yj_min, xj_max, yj_max = self.get_corners(self.boxes[j])
                
                # 使用 ReLU 确保满足条件时 Loss 为 0 (梯度消失，从而允许 Fidelity 接管)
                if ctype == 'left_of':  val = (xi_max + self.margin) - xj_min
                elif ctype == 'right_of': val = (xj_max + self.margin) - xi_min
                elif ctype == 'above':    val = (yi_max + self.margin) - yj_min
                elif ctype == 'below':    val = (yj_max + self.margin) - yi_min
                else: val = torch.tensor(0.0)
                
                loss += torch.relu(val)

        # --- B. 物理约束 (防重叠) ---
        # 简单两两重叠检测
        n = len(self.boxes)
        for i in range(n):
            for j in range(i + 1, n):
                xi_min, yi_min, xi_max, yi_max = self.get_corners(self.boxes[i])
                xj_min, yj_min, xj_max, yj_max = self.get_corners(self.boxes[j])
                
                inter_w = torch.relu(torch.min(xi_max, xj_max) - torch.max(xi_min, xj_min))
                inter_h = torch.relu(torch.min(yi_max, yj_max) - torch.max(yi_min, yj_min))
                loss += inter_w * inter_h * 10.0 # 重叠是硬伤，加权

        # --- C. 边界约束 ---
        l_bound = (torch.relu(-self.boxes[:,0]) + torch.relu(self.boxes[:,0]+self.boxes[:,2]-1) + 
                   torch.relu(-self.boxes[:,1]) + torch.relu(self.boxes[:,1]+self.boxes[:,3]-1)).sum()
        
        return loss + l_bound * 10.0

    def calc_fidelity_loss(self):
        """计算与初始状态的偏差 (MSE)"""
        # 包含位置和尺寸的偏差
        return torch.mean((self.boxes - self.initial_state) ** 2)

    def optimize(self, steps=500, lr=0.01):
        optimizer = optim.Adam([self.boxes], lr=lr)
        
        print(f"开始优化: Alpha (Fidelity Strength) = {self.alpha}")
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # 1. 分别计算 Loss
            l_const = self.calc_constraint_loss()
            l_fid = self.calc_fidelity_loss()
            
            # ==================================================
            # 核心：动态 Barrier 逻辑
            # ==================================================
            
            if l_const > self.threshold:
                # 状态 1: 严重违规 (Violation Mode)
                # 策略: 忽略 Fidelity，全力修复约束
                # 权重: Constraint = 极高, Fidelity = 0
                w_c = 1000.0
                w_f = 0.0
                state_msg = "VIOLATION"
            else:
                # 状态 2: 可行域内 (Feasible Mode)
                # 策略: 开启 Fidelity，但保留 Constraint 作为“墙”
                # 注意：Constraint 权重依然要保留，否则 Fidelity 会把盒子拉回违规区
                w_c = 100.0  # 这里的权重充当 Barrier 的硬度
                w_f = self.alpha # 用户指定的力度
                state_msg = "FEASIBLE"
                
            total_loss = w_c * l_const + w_f * l_fid
            
            total_loss.backward()
            optimizer.step()
            
            # 记录数据
            self.history['boxes'].append(self.boxes.detach().numpy().copy())
            self.history['loss_c'].append(l_const.item())
            self.history['loss_f'].append(l_fid.item())
            
            if step % 50 == 0:
                print(f"Step {step:03d} [{state_msg}] | Loss_C: {l_const.item():.6f} | Loss_F: {l_fid.item():.6f}")

        return self.boxes.detach()

# ==========================================
# 测试场景设计
# ==========================================
# 场景："乾坤大挪移" (The Great Swap)
# 初始：Box A 在左 (0.2), Box B 在右 (0.8)
# 约束：强制要求 "Box A 在 Box B 右边" (right_of)
# 预期：它们必须互换位置（严重违反 Fidelity），一旦互换成功，它们应该尽量靠近初始位置（挤在中间）

init_boxes = torch.tensor([
    [0.2, 0.4, 0.1, 0.2], # Box 0 (Initially Left)
    [0.8, 0.4, 0.1, 0.2], # Box 1 (Initially Right)
])

# 这是一个非常强的冲突约束
constraints = [{'type': 'right_of', 'source': 0, 'target': 1}]

# 用户设置一个中等的保真度意愿
opt = DynamicBarrierOptimizer(init_boxes, constraints, fidelity_alpha=5.0)
final_boxes = opt.optimize(steps=300)

# ==========================================
# 可视化：轨迹追踪
# ==========================================
def visualize_trajectory(optimizer, title):
    history = np.array(optimizer.history['boxes']) # Shape: [steps, N, 4]
    loss_c = optimizer.history['loss_c']
    loss_f = optimizer.history['loss_f']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- 图 1: 运动轨迹 ---
    ax1.set_title(title)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.invert_yaxis()
    ax1.add_patch(patches.Rectangle((0,0),1,1, fill=False, ec='black'))
    
    colors = ['red', 'blue']
    labels = ['A', 'B']
    
    # 画初始位置 (虚线)
    initial = history[0]
    for i in range(len(initial)):
        b = initial[i]
        ax1.add_patch(patches.Rectangle((b[0], b[1]), b[2], b[3], 
                                        ec=colors[i], fill=False, ls='--', alpha=0.5, label=f'{labels[i]} Start'))
    
    # 画最终位置 (实线)
    final = history[-1]
    for i in range(len(final)):
        b = final[i]
        rect = patches.Rectangle((b[0], b[1]), b[2], b[3], ec='black', fc=colors[i], alpha=0.6, label=f'{labels[i]} End')
        ax1.add_patch(rect)
        ax1.text(b[0]+b[2]/2, b[1]+b[3]/2, labels[i], color='white', ha='center', va='center', fontweight='bold')

    # 画轨迹线 (Trajectory)
    for i in range(len(initial)):
        # 取中心点轨迹
        cx = history[:, i, 0] + history[:, i, 2]/2
        cy = history[:, i, 1] + history[:, i, 3]/2
        ax1.plot(cx, cy, color=colors[i], alpha=0.3, linewidth=2)
        # 标出起点和终点
        ax1.scatter(cx[0], cy[0], c=colors[i], marker='o', s=30)
        ax1.scatter(cx[-1], cy[-1], c=colors[i], marker='x', s=50)

    ax1.legend()

    # --- 图 2: Loss 变化曲线 ---
    ax2.set_title("Optimization Dynamics: Barrier Method")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Loss Value")
    
    # 双轴绘制
    ax2_r = ax2.twinx()
    l1 = ax2.plot(loss_c, color='orange', label='Constraint Loss (Log Scale)', linewidth=2)
    l2 = ax2_r.plot(loss_f, color='green', label='Fidelity Loss', linewidth=2, linestyle='--')
    
    ax2.set_yscale('log') # 约束 Loss 通常变化巨大，用对数坐标
    ax2.set_ylim(bottom=1e-5)
    
    # 添加图例
    lines = l1 + l2
    labs = [l.get_label() for l in lines]
    ax2.legend(lines, labs, loc='center right')
    
    plt.tight_layout()
    plt.savefig('test_constracts_adjust_v2.png')
    # plt.show()

visualize_trajectory(opt, "Dynamic Barrier Optimization\n(Swapping Positions while maintaining Fidelity)")