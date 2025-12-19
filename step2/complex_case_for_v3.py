import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from test_constracts_adjust_v3 import DynamicBarrierFixedSizeOptimizer

# (此处假定 DynamicBarrierFixedSizeOptimizer 类已定义，直接调用)

# ==========================================
# 场景 1: 移动端个人主页 (The Mobile Profile)
# ==========================================
# 结构设计：
# 0. Navbar (顶部导航): Full Width, Top
# 1. Avatar (头像): 左上，Navbar 下方
# 2. Name (用户名): Avatar 右侧
# 3. Bio (简介): Name 下方，但不能超过 Avatar 的底边太多(对齐)
# 4. Stats (数据栏): Avatar/Bio 下方，Full Width
# 5. Gallery 1 (图片): Stats 下方，左侧
# 6. Gallery 2 (图片): Gallery 1 右侧
# 7. Button (底部按钮): 底部，Full Width

def run_complex_scenario_1():
    print(">>> 运行场景 1: Mobile Profile UI")
    
    # 初始状态：全部挤在中间的一坨乱麻
    # 格式: x, y, w, h
    init_boxes = torch.tensor([
        [0.4, 0.4, 0.2, 0.1], # 0 Navbar
        [0.4, 0.4, 0.1, 0.1], # 1 Avatar
        [0.4, 0.5, 0.2, 0.1], # 2 Name
        [0.5, 0.5, 0.2, 0.1], # 3 Bio
        [0.4, 0.6, 0.2, 0.1], # 4 Stats
        [0.4, 0.7, 0.1, 0.1], # 5 Gal 1
        [0.5, 0.7, 0.1, 0.1], # 6 Gal 2
        [0.4, 0.8, 0.2, 0.1], # 7 Button
    ])

    labels = ["Navbar", "Avatar", "Name", "Bio", "Stats", "Gal-1", "Gal-2", "Button"]
    colors = ['#333333', '#FF5733', '#33FF57', '#3357FF', '#F0F0F0', '#FF33FF', '#33FFFF', '#FFD700']

    constraints = [
        # --- Navbar ---
        {'type': 'top', 'target': 0},
        {'type': 'full_width', 'target': 0},
        
        # --- Header Section ---
        {'type': 'below', 'source': 1, 'target': 0}, # Avatar below Navbar
        {'type': 'left', 'target': 1},               # Avatar aligns left
        {'type': 'right_of', 'source': 2, 'target': 1}, # Name right of Avatar
        {'type': 'below', 'source': 2, 'target': 0},    # Name below Navbar
        {'type': 'below', 'source': 3, 'target': 2},    # Bio below Name
        {'type': 'right_of', 'source': 3, 'target': 1}, # Bio right of Avatar
        
        # --- Stats Section ---
        {'type': 'below', 'source': 4, 'target': 1}, # Stats below Avatar
        {'type': 'below', 'source': 4, 'target': 3}, # Stats below Bio (防止Bio太长穿模)
        {'type': 'full_width', 'target': 4},
        
        # --- Gallery Grid ---
        {'type': 'below', 'source': 5, 'target': 4}, # Gal-1 below Stats
        {'type': 'left', 'target': 5},               # Gal-1 align left
        {'type': 'right_of', 'source': 6, 'target': 5}, # Gal-2 right of Gal-1
        {'type': 'below', 'source': 6, 'target': 4},    # Gal-2 below Stats
        
        # --- Footer ---
        {'type': 'below', 'source': 7, 'target': 5}, # Button below Gal-1
        {'type': 'below', 'source': 7, 'target': 6}, # Button below Gal-2
        {'type': 'bottom', 'target': 7},             # Button stick to bottom
        {'type': 'full_width', 'target': 7}
    ]

    # 运行优化 (设置较高的 Fidelity，假设初始尺寸是设计大致想要的)
    opt = DynamicBarrierFixedSizeOptimizer(init_boxes, constraints, fidelity_alpha=2.0)
    final_boxes = opt.optimize(steps=400)
    
    return init_boxes, final_boxes, labels, colors


# ==========================================
# 场景 2: 仪表盘布局 (Bento Dashboard)
# ==========================================
# 结构设计：
# 0. Main Chart (主图表): 占据左上大片区域
# 1. KPI 1 (指标卡): 右侧顶部
# 2. KPI 2 (指标卡): 右侧中部
# 3. KPI 3 (指标卡): 右侧底部
# 4. Log Console (底部日志): 贯穿底部的长条

def run_complex_scenario_2():
    print("\n>>> 运行场景 2: Bento Dashboard")
    
    # 初始状态：模拟用户大概拖拽了一下，但没对齐
    init_boxes = torch.tensor([
        [0.05, 0.05, 0.4, 0.4], # 0 Main Chart (位置大概对，但和其他重叠)
        [0.6, 0.1, 0.2, 0.1],   # 1 KPI 1
        [0.6, 0.3, 0.2, 0.1],   # 2 KPI 2
        [0.6, 0.5, 0.2, 0.1],   # 3 KPI 3
        [0.1, 0.8, 0.8, 0.1],   # 4 Console
    ])
    
    labels = ["Main Chart", "KPI 1", "KPI 2", "KPI 3", "Console"]
    colors = ['#1f77b4', '#ff7f0e', '#ff7f0e', '#ff7f0e', '#2ca02c']

    constraints = [
        # --- Main Chart ---
        {'type': 'top', 'target': 0},
        {'type': 'left', 'target': 0},
        
        # --- Right Column Stack ---
        {'type': 'right_of', 'source': 1, 'target': 0}, # KPI 1 在主图右边
        {'type': 'top', 'target': 1},                   # KPI 1 顶对齐
        
        {'type': 'below', 'source': 2, 'target': 1},    # KPI 2 在 KPI 1 下面
        {'type': 'right_of', 'source': 2, 'target': 0}, # KPI 2 在主图右边
        
        {'type': 'below', 'source': 3, 'target': 2},    # KPI 3 在 KPI 2 下面
        {'type': 'right_of', 'source': 3, 'target': 0}, # KPI 3 在主图右边
        
        # --- Bottom Console ---
        {'type': 'below', 'source': 4, 'target': 0},    # 在主图下面
        {'type': 'below', 'source': 4, 'target': 3},    # 也在 KPI 3 下面 (关键！防止跟右侧列重叠)
        {'type': 'bottom', 'target': 4},                # 沉底
        {'type': 'full_width', 'target': 4}
    ]

    # 这里我们设置 alpha 较小，允许布局为了对齐做较大的形变
    opt = DynamicBarrierFixedSizeOptimizer(init_boxes, constraints, fidelity_alpha=0.5)
    final_boxes = opt.optimize(steps=400)
    
    return init_boxes, final_boxes, labels, colors

# ==========================================
# 通用绘图函数
# ==========================================
def plot_results(init, final, labels, colors, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    def draw(ax, boxes, t):
        ax.set_title(t)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.invert_yaxis()
        ax.add_patch(patches.Rectangle((0,0),1,1, fill=False, ec='black'))
        
        for i, b in enumerate(boxes):
            x, y, w, h = b
            # 绘制
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='black', facecolor=colors[i], alpha=0.7)
            ax.add_patch(rect)
            # 文本
            ax.text(x+w/2, y+h/2, labels[i], ha='center', va='center', fontsize=9, color='white', weight='bold')

    draw(ax1, init, "Initial State (Messy)")
    draw(ax2, final, "Optimized State (Structured)")
    plt.tight_layout()
    plt.savefig(f'{title}.png')
    # plt.show()

# ==========================================
# 执行
# ==========================================

# 运行场景 1
i1, f1, l1, c1 = run_complex_scenario_1()
plot_results(i1, f1, l1, c1, "Scenario 1: Mobile App UI")

# 运行场景 2
i2, f2, l2, c2 = run_complex_scenario_2()
plot_results(i2, f2, l2, c2, "Scenario 2: Bento Dashboard")