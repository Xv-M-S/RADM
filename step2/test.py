"""
README:

This script is used to test the optimization of the layout of the boxes.
The boxes are initialized with random positions and fixed sizes.
The optimization is performed by minimizing the loss function.
The loss function is the sum of the following three terms:
1. The alignment loss of the bottom of the boxes.
2. The overlap loss of the boxes.
3. The boundary loss of the boxes.


Disadvantages:
1. we cannot control the relationship between the boxes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# 1. 定义核心优化逻辑
# ==========================================

def get_bboxes(xy, wh):
    """辅助函数：将中心/角点转换为 x_min, y_min, x_max, y_max"""
    # 假设 xy 是左上角坐标
    x, y = xy[:, 0], xy[:, 1]
    w, h = wh[:, 0], wh[:, 1]
    return x, y, x + w, y + h

def calculate_loss(xy, wh):
    """计算总能量（Loss）"""
    x_min, y_min, x_max, y_max = get_bboxes(xy, wh)
    
    # --- A. 底部对齐 Loss (Bottom Alignment) ---
    # 我们希望两个 Box 的底部 (y_max) 在同一水平线
    # 假设坐标系原点在左上角，y 向下增加，则 y_max 代表底部
    l_align = (y_max[0] - y_max[1]) ** 2
    
    # --- B. 防重叠 Loss (Overlap) ---
    # 计算交集面积
    inter_x1 = torch.max(x_min[0], x_min[1])
    inter_y1 = torch.max(y_min[0], y_min[1])
    inter_x2 = torch.min(x_max[0], x_max[1])
    inter_y2 = torch.min(y_max[0], y_max[1])
    
    inter_w = torch.relu(inter_x2 - inter_x1)
    inter_h = torch.relu(inter_y2 - inter_y1)
    l_overlap = inter_w * inter_h
    
    # --- C. 边界 Loss (Boundary) ---
    # 限制在 [0, 1] 范围内
    l_bound = (torch.relu(-x_min).sum() + torch.relu(x_max - 1).sum() +
               torch.relu(-y_min).sum() + torch.relu(y_max - 1).sum())

    # 总 Loss 加权
    total_loss = 20.0 * l_align + 50.0 * l_overlap + 10.0 * l_bound
    return total_loss

# ==========================================
# 2. 初始化数据与优化器
# ==========================================

# 场景：
# Box A (蓝色): 宽大的背景物体 (e.g. 房子)
# Box B (橙色): 细高的前景物体 (e.g. 树/人)
# 初始状态：两者重叠，且底部没有对齐
initial_wh = torch.tensor([
    [0.4, 0.3],  # Box A w, h
    [0.15, 0.5]  # Box B w, h
])
initial_xy = torch.tensor([
    [0.4, 0.4],  # Box A x, y (故意放在中间)
    [0.45, 0.5]  # Box B x, y (故意重叠)
])

# 记录初始状态用于绘图
start_xy = initial_xy.clone()

# 设置可优化参数
boxes_xy = nn.Parameter(initial_xy.clone())
optimizer = optim.Adam([boxes_xy], lr=0.02)

# ==========================================
# 3. 执行推理时优化
# ==========================================

print("开始优化布局...")
steps = 200
for i in range(steps):
    optimizer.zero_grad()
    loss = calculate_loss(boxes_xy, initial_wh)
    loss.backward()
    optimizer.step()
    
    # 可选：每一步都 Clamp 一下防止跑太远，或者依靠 Boundary Loss 拉回来
    # with torch.no_grad():
    #     boxes_xy.clamp_(0, 1)

final_xy = boxes_xy.detach()
print("优化完成。")

# ==========================================
# 4. 可视化绘图
# ==========================================

def draw_boxes(ax, xy, wh, title, colors):
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis() # 图像坐标系：原点在左上角
    ax.set_aspect('equal')
    
    # 画画布边框
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2))

    # 获取数据
    x = xy[:, 0].numpy()
    y = xy[:, 1].numpy()
    w = wh[:, 0].numpy()
    h = wh[:, 1].numpy()

    labels = ['A (Building)', 'B (Person)']
    
    # 计算底部位置用于画辅助线
    bottoms = y + h
    
    for i in range(len(x)):
        # 画矩形
        rect = patches.Rectangle((x[i], y[i]), w[i], h[i], 
                                 linewidth=2, edgecolor=colors[i], facecolor=colors[i], alpha=0.5)
        ax.add_patch(rect)
        # 标字
        ax.text(x[i], y[i]-0.02, labels[i], color=colors[i], fontsize=10, weight='bold')
        # 画底部点
        ax.scatter(x[i] + w[i]/2, y[i] + h[i], color='red', zorder=5)

    # 画一条虚线表示 Box A 的底部，看 B 是否对齐
    ax.axhline(bottoms[0], color='gray', linestyle='--', alpha=0.5, label='Alignment Line')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 绘制初始状态
draw_boxes(ax1, start_xy, initial_wh, "Before Optimization\n(Overlapping & Misaligned)", ['blue', 'orange'])

# 绘制优化后状态
draw_boxes(ax2, final_xy, initial_wh, "After Optimization\n(Separated & Bottom Aligned)", ['blue', 'orange'])

plt.tight_layout()
plt.savefig('test.png')
# plt.show()