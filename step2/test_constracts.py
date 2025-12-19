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

# ==========================================
# 1. 核心优化器类 (集成相对与绝对约束逻辑)
# ==========================================
class LayoutOptimizer:
    def __init__(self, initial_boxes, constraints, canvas_size=(1.0, 1.0)):
        # 允许优化 x, y, w, h
        self.boxes = nn.Parameter(initial_boxes.clone())
        self.constraints = constraints
        self.W, self.H = canvas_size
        self.margin = 0.05 # 物体间的最小间隙

    def get_corners(self, idx):
        x, y, w, h = self.boxes[idx]
        return x, y, x+w, y+h, w, h

    def get_center(self, idx):
        x, y, w, h = self.boxes[idx]
        return x + w/2, y + h/2

    def calc_loss(self):
        loss = torch.tensor(0.0)
        
        # --- A. 相对位置约束 ---
        for c in self.constraints:
            ctype = c['type']
            if 'source' not in c: continue # 跳过绝对约束
            
            i, j = c['source'], c['target']
            xi_min, yi_min, xi_max, yi_max, _, _ = self.get_corners(i)
            xj_min, yj_min, xj_max, yj_max, _, _ = self.get_corners(j)

            if ctype == 'left_of':
                # i 的右边界 > j 的左边界 -> 产生 Loss
                loss += torch.relu((xi_max + self.margin) - xj_min)
            elif ctype == 'right_of':
                loss += torch.relu((xj_max + self.margin) - xi_min)
            elif ctype == 'above':
                loss += torch.relu((yi_max + self.margin) - yj_min)
            elif ctype == 'below':
                loss += torch.relu((yj_max + self.margin) - yi_min)

        # --- B. 绝对位置约束 ---
        anchors = {
            'top': (0.5, 0.1), 'bottom': (0.5, 0.9),
            'left': (0.1, 0.5), 'right': (0.9, 0.5)
        }
        
        for c in self.constraints:
            ctype = c['type']
            if 'source' in c: continue # 跳过相对约束
            
            i = c['target']
            x, y, x_max, y_max, w, h = self.get_corners(i)
            cx, cy = self.get_center(i)

            # 锚点吸附
            if ctype in anchors:
                tx, ty = anchors[ctype]
                # 这里的逻辑稍微简化：如果是 top，只约束 y；如果是 left，只约束 x
                if ctype == 'top': loss += (y)**2 # y 趋近 0
                if ctype == 'bottom': loss += (y_max - 1.0)**2 # y_max 趋近 1
                if ctype == 'left': loss += (x)**2
                if ctype == 'right': loss += (x_max - 1.0)**2
            
            # 尺寸强约束
            elif ctype == 'full_width':
                loss += (x)**2 + (w - 1.0)**2 # x=0, w=1
            elif ctype == 'full_height':
                loss += (y)**2 + (h - 1.0)**2

        # --- C. 基础物理约束 ---
        # 1. 边界限制 (0-1之间)
        l_bound = (torch.relu(-self.boxes[:,0]) + torch.relu(self.boxes[:,0]+self.boxes[:,2]-1.0) + 
                   torch.relu(-self.boxes[:,1]) + torch.relu(self.boxes[:,1]+self.boxes[:,3]-1.0)).sum()
        
        # 2. 尺寸合理性 (w, h 必须 > 0.05，防止缩成一个点)
        l_size = (torch.relu(0.05 - self.boxes[:,2]) + torch.relu(0.05 - self.boxes[:,3])).sum()

        return loss + 10.0 * l_bound + 10.0 * l_size

    def optimize(self, steps=500, lr=0.01):
        optimizer = optim.Adam([self.boxes], lr=lr)
        history = []
        
        for step in range(steps):
            optimizer.zero_grad()
            loss = self.calc_loss()
            loss.backward()
            optimizer.step()
            # 记录用于后续分析（可选）
            if step % 100 == 0:
                pass # print(f"Step {step}, Loss: {loss.item()}")
        
        return self.boxes.detach()

# ==========================================
# 2. 场景定义：网页布局
# ==========================================

# 定义 4 个 Box
# 0: Header, 1: Sidebar, 2: Content, 3: Footer
labels = ["Header", "Sidebar", "Content", "Footer"]
colors = ['#FF9999', '#99FF99', '#9999FF', '#CCCCCC'] # 红绿蓝灰

# 初始状态：全部挤在中间，尺寸很小，完全不符合要求
init_boxes = torch.tensor([
    [0.4, 0.4, 0.2, 0.1], # 0
    [0.4, 0.5, 0.1, 0.2], # 1
    [0.5, 0.5, 0.2, 0.2], # 2
    [0.4, 0.7, 0.2, 0.1], # 3
])

# 约束定义（自然语言翻译过来的逻辑）
constraints = [
    # --- Header ---
    {'type': 'full_width', 'target': 0}, # 撑满宽度
    {'type': 'top', 'target': 0},        # 靠顶
    
    # --- Sidebar ---
    {'type': 'left', 'target': 1},       # 靠左
    {'type': 'below', 'source': 1, 'target': 0}, # 在 Header 下面
    
    # --- Content ---
    {'type': 'right_of', 'source': 2, 'target': 1}, # 在 Sidebar 右边
    {'type': 'below', 'source': 2, 'target': 0},    # 在 Header 下面
    {'type': 'above', 'source': 2, 'target': 3},    # 在 Footer 上面
    
    # --- Footer ---
    {'type': 'full_width', 'target': 3}, # 撑满宽度
    {'type': 'bottom', 'target': 3},     # 靠底
    {'type': 'below', 'source': 3, 'target': 1},    # 在 Sidebar 下面 (防止Sidebar穿过Footer)
]

# ==========================================
# 3. 运行与可视化
# ==========================================

opt = LayoutOptimizer(init_boxes, constraints)
final_boxes = opt.optimize(steps=1000, lr=0.01)

def draw_layout(ax, boxes, title):
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis() # 坐标原点在左上角
    
    # 画边框
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2))
    
    boxes_np = boxes.numpy()
    
    for i, box in enumerate(boxes_np):
        x, y, w, h = box
        # 简单防呆，防止画图报错
        w, h = max(0.01, w), max(0.01, h)
        
        # 矩形
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='black', facecolor=colors[i], alpha=0.8)
        ax.add_patch(rect)
        
        # 文字
        ax.text(x + w/2, y + h/2, labels[i], 
                ha='center', va='center', fontsize=10, fontweight='bold', color='black')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

draw_layout(ax1, init_boxes.detach(), "Initial State\n(Random & Overlapping)")
draw_layout(ax2, final_boxes, "Optimized State\n(Constrained Layout)")

plt.tight_layout()
plt.savefig('test_constracts.png')
# plt.show()