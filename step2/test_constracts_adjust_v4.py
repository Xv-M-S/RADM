"""
README:
This script is used to test the optimization of the layout of the boxes with constraints.
The boxes are initialized with random positions and fixed sizes.
The optimization is performed by minimizing the loss function.
The loss function is the sum of the following three terms:
1. The alignment loss of the bottom of the boxes.
2. The overlap loss of the boxes.
3. The boundary loss of the boxes.
4. The centering loss of the boxes.
"""



import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class DynamicBarrierFixedSizeOptimizer:
    def __init__(self, initial_boxes, constraints, fidelity_alpha=1.0, centering_alpha=0.5, threshold=1e-4):
        """
        Modified Version: Fixed Size (w, h) + Centering Potential
        """
        # 1. 记录初始位置 (用于保真度计算)
        self.initial_xy = initial_boxes[:, :2].clone().detach()
        
        # 2. 拆分参数：xy 可动，wh 固定
        self.xy = nn.Parameter(initial_boxes[:, :2].clone())
        self.wh = initial_boxes[:, 2:].clone().detach()
        
        self.constraints = constraints
        self.fid_alpha = fidelity_alpha   # 保真度权重 (拉回原位)
        self.cent_alpha = centering_alpha # 居中权重 (拉向中心)
        self.threshold = threshold
        self.margin = 0.05

        # 记录历史 (新增 loss_center)
        self.history = {'boxes': [], 'loss_c': [], 'loss_f': [], 'loss_center': []}

    @property
    def boxes(self):
        return torch.cat([self.xy, self.wh], dim=1)

    def get_corners(self, xywh):
        x, y, w, h = xywh
        return x, y, x+w, y+h

    def calc_constraint_loss(self):
        # ... (保持原有的约束逻辑不变) ...
        loss = torch.tensor(0.0)
        current_boxes = self.boxes

        # A. 逻辑约束
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

        # B. 物理互斥 (防重叠)
        n = len(current_boxes)
        for i in range(n):
            for j in range(i + 1, n):
                xi_min, yi_min, xi_max, yi_max = self.get_corners(current_boxes[i])
                xj_min, yj_min, xj_max, yj_max = self.get_corners(current_boxes[j])
                
                inter_w = torch.relu(torch.min(xi_max, xj_max) - torch.max(xi_min, xj_min))
                inter_h = torch.relu(torch.min(yi_max, yj_max) - torch.max(yi_min, yj_min))
                loss += inter_w * inter_h * 10.0

        # C. 边界约束
        l_bound = (torch.relu(-current_boxes[:,0]) + torch.relu(current_boxes[:,0]+current_boxes[:,2]-1) + 
                   torch.relu(-current_boxes[:,1]) + torch.relu(current_boxes[:,1]+current_boxes[:,3]-1)).sum()
        
        return loss + l_bound * 10.0

    def calc_fidelity_loss(self):
        # 计算相对于初始位置的位移 (MSE)
        return torch.mean((self.xy - self.initial_xy) ** 2)

    def calc_centering_loss(self):
        """
        [新增] 计算居中势能
        目标：让每个 Box 的中心点 (cx, cy) 尽可能靠近画布中心 (0.5, 0.5)
        """
        # 计算当前中心点
        current_boxes = self.boxes
        cx = current_boxes[:, 0] + current_boxes[:, 2] / 2
        cy = current_boxes[:, 1] + current_boxes[:, 3] / 2
        
        # 计算到 (0.5, 0.5) 的欧式距离平方和
        # 使用 sum 而不是 mean，以便组件越多，向心力越强（可选）
        dist = (cx - 0.5)**2 + (cy - 0.5)**2
        return torch.sum(dist)

    def optimize(self, steps=500, lr=0.01):
        optimizer = optim.Adam([self.xy], lr=lr)
        
        print(f"开始优化 (Centered): Fid_Alpha={self.fid_alpha}, Cent_Alpha={self.cent_alpha}")
        
        for step in range(steps):
            optimizer.zero_grad()
            
            l_const = self.calc_constraint_loss()
            l_fid = self.calc_fidelity_loss()
            l_center = self.calc_centering_loss() # 计算居中 Loss
            
            # --- 动态权重调度 ---
            if l_const > self.threshold:
                # [VIOLATION 模式]
                # 优先级：全力解决约束冲突
                # 策略：关闭居中力，防止它干扰拓扑解结
                w_c = 1000.0
                w_f = 0.0       # 暂时忽略保真度
                w_center = 0.0  # 暂时忽略居中
                state_msg = "VIOLATION"
            else:
                # [FEASIBLE 模式]
                # 优先级：在可行域内寻找更优解
                # 策略：开启保真度和居中力
                w_c = 100.0     # 保持约束壁垒
                w_f = self.fid_alpha 
                w_center = self.cent_alpha # 引入全局先验
                state_msg = "FEASIBLE"
            
            total_loss = w_c * l_const + w_f * l_fid + w_center * l_center
            
            total_loss.backward()
            optimizer.step()
            
            # 记录历史
            current_snapshot = self.boxes.detach().numpy().copy()
            self.history['boxes'].append(current_snapshot)
            self.history['loss_c'].append(l_const.item())
            self.history['loss_f'].append(l_fid.item())
            self.history['loss_center'].append(l_center.item()) # 记录居中Loss
            
            if step % 50 == 0:
                print(f"Step {step:03d} [{state_msg}] | Const: {l_const.item():.4f} | Fid: {l_fid.item():.4f} | Center: {l_center.item():.4f}")

        return self.boxes.detach()

# ==========================================
# 测试与可视化 (更新版)
# ==========================================

# 场景设置：两个矩形互换位置
init_boxes = torch.tensor([
    [0.2, 0.35, 0.1, 0.3], # Box A
    [0.8, 0.45, 0.3, 0.1], # Box B
])

# 约束：A 必须在 B 右边 (强制互换)
constraints = [{'type': 'right_of', 'source': 0, 'target': 1}]

# 实例化优化器：
# fidelity_alpha = 1.0 (即使满足约束，也不要跑太远)
# centering_alpha = 2.0 (稍微强一点的向心力，让它们排好后往中间靠)
opt = DynamicBarrierFixedSizeOptimizer(init_boxes, constraints, fidelity_alpha=1.0, centering_alpha=2.0)
final_boxes = opt.optimize(steps=1000)

def visualize_trajectory_with_center(optimizer, title):
    history = np.array(optimizer.history['boxes'])
    loss_c = optimizer.history['loss_c']
    loss_f = optimizer.history['loss_f']
    loss_center = optimizer.history['loss_center'] # 获取居中Loss数据
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- 左图：轨迹 ---
    ax1.set_title(title)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.invert_yaxis()
    ax1.add_patch(patches.Rectangle((0,0),1,1, fill=False, ec='black', lw=2))
    # 画出画布中心点
    ax1.scatter(0.5, 0.5, c='black', marker='+', s=100, label='Canvas Center')
    
    colors = ['red', 'blue']
    labels = ['A', 'B']
    
    # 绘制起点和终点
    initial = history[0]
    final = history[-1]
    
    for i in range(len(initial)):
        # Start
        b_s = initial[i]
        ax1.add_patch(patches.Rectangle((b_s[0], b_s[1]), b_s[2], b_s[3], 
                                      ec=colors[i], fill=False, ls='--', alpha=0.4))
        # End
        b_e = final[i]
        ax1.add_patch(patches.Rectangle((b_e[0], b_e[1]), b_e[2], b_e[3], 
                                      ec='black', fc=colors[i], alpha=0.6, label=f'{labels[i]} Final'))
        
        # 轨迹线
        cx = history[:, i, 0] + history[:, i, 2]/2
        cy = history[:, i, 1] + history[:, i, 3]/2
        ax1.plot(cx, cy, color=colors[i], alpha=0.3, linewidth=2)
    
    ax1.legend()

    # --- 右图：Loss 曲线 ---
    ax2.set_title("Optimization Dynamics (Log Scale)")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Loss Magnitude")
    
    # 使用双轴 (Constraints 通常很大，Fidelity/Center 通常很小)
    l1 = ax2.plot(loss_c, color='orange', label='Constraint (Barrier)', linewidth=2)
    ax2.set_yscale('log')
    ax2.set_ylim(bottom=1e-5)
    
    ax2_r = ax2.twinx()
    l2 = ax2_r.plot(loss_f, color='green', label='Fidelity (Stay)', linewidth=1.5, linestyle='--')
    l3 = ax2_r.plot(loss_center, color='purple', label='Centering (Gravity)', linewidth=2, linestyle='-')
    
    # 合并图例
    lines = l1 + l2 + l3
    labs = [l.get_label() for l in lines]
    ax2.legend(lines, labs, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('test_constracts_adjust_v4.png')
    # plt.show()

visualize_trajectory_with_center(opt, "Fixed Size + Centering Prior\n(Blocks swap & move to center)")