# RADM 代码修改说明

> 基于论文需求文档 `/home/sxm/data02Space/GraduationProject/textToLayoutBaseRADM/README.md` 对 RADM 仓库的代码改写。

---

## 一、需求概述

论文提出了一个**融合视觉先验与拓扑约束的布局生成框架**，核心包括以下模块：

1. **基于异构图神经网络的约束图编码** — 将离散语义与空间关系统一映射为结构化节点表征
2. **辅助监督损失** — 关系重构损失 L_rel 与网格位置分类损失 L_pos
3. **背景视觉先验编码** — DINO-ViT + Linear Adapter + RoIAlign 提取背景视觉特征
4. **几何关系感知的空间上下文建模** — 基于几何注意力加权聚合视觉特征
5. **布局表示与多模态解码** — CLIP文本编码 + Fourier特征编码 + MM-DiT风格联合注意力解码

---

## 二、新建文件

### 2.1 `RADM/rgcn.py` — 关系感知图卷积网络

**对应论文章节**: §约束图表示与空间语义编码

| 类名 | 功能 |
|------|------|
| `RGCNLayer` | 单层RGCN：对每种关系类型独立的线性变换 + 归一化邻接矩阵消息传递 + 基分解参数效率优化 |
| `RGCN` | 多层RGCN堆叠：`in_dim → hidden_dim → ... → out_dim`，每层带残差连接 + LayerNorm。输入初始节点特征 `h_i^(0)` 和邻接矩阵 `adj_mats[3, N, N]`，输出 `H_topo[N, out_dim]` |
| `RelationClassifier` | 关系重构分类头：对节点对特征 `[h_i, h_j, h_i*h_j]` 预测相对位置类别 |
| `GridPositionPredictor` | 网格位置预测头：对节点特征预测 S×S 网格索引 |

**关键参数**:
- `NUM_RELATIONS=3` (BB背景-背景, BF背景-前景, FF前景-前景)
- `NUM_BASES=4` (基分解减少参数量)
- `NUM_LAYERS=2` (消息传递层数)

### 2.2 `RADM/constraint_graph.py` — 约束图构建器

**对应论文章节**: §约束图表示与空间语义编码

| 类名 | 功能 |
|------|------|
| `PositionalEncoding1D` | 对bbox坐标 `(cx,cy,w,h)` 进行正弦位置编码 |
| `ConstraintGraphBuilder` | 构建多模态约束图 `G=(V,E)` |

**节点特征初始化**: `h_i^(0) = [e_i^text ‖ e_i^pos ‖ e_i^cls]`

- `e_i^text`: 预训练文本编码器(768维) → Linear投影 → hidden_dim
- `e_i^pos`: 绝对位置嵌入 (PositionalEncoding1D)
- `e_i^cls`: 可学习类别嵌入 (Embedding)

**边构建**: 根据节点类型(背景/前景)自动分配关系类型

**关系标签**: 8方向离散相对位置标签用于辅助监督

### 2.3 `RADM/visual_encoder.py` — 背景视觉先验编码器

**对应论文章节**: §背景视觉先验编码

| 类名 | 功能 |
|------|------|
| `DinoViTEncoder` | DINOv2 ViT主干网络：提取空间patch tokens `Z_spatial` |
| `PlaceholderViT` | 占位ViT编码器(当DINOv2不可用时) |
| `LinearAdapter` | 轻量线性适配器：`F_vis = W_proj·F + b_proj`，将特征维度映射到目标通道数 |
| `RoIExtractor` | RoIAlign操作：`V = RoIAlign(F_vis, x)` 提取候选布局框的局部视觉特征 |
| `VisualPriorEncoder` | 完整流水线封装：DINO-ViT → Reshape → LinearAdapter → RoIAlign |

**处理流程**:
```
I_bg (H×W×3) → DINO-ViT → Z_spatial [(N_p+1)×D]
→ 去掉CLS token → Reshape → F [D×H_f×W_f]
→ LinearAdapter → F_vis [C×H_f×W_f]
→ RoIAlign(boxes) → V [C×H_r×W_r]
```

### 2.4 `RADM/geometry_relation.py` — 几何关系感知模块

**对应论文章节**: §几何关系感知的空间上下文建模

| 类名 | 功能 |
|------|------|
| `GeometryRelationModule` | 单batch几何关系建模 |
| `GeometryRelationModuleBatch` | 处理变长元素的batch版本 |

**几何关系向量**:
```
g_ij = [log(|x_i-x_j|/w_j), log(|y_i-y_j|/h_j), log(w_i/w_j), log(h_i/h_j)]
```

**计算流程**:
1. 构建尺度不变相对几何向量 `g_ij`
2. Sin-Cos位置编码: `R_ij^p = PE(g_ij)` (高频空间结构信息)
3. 几何注意力: `α_ij = Softmax(MLP(R_ij^p))`
4. 加权聚合: `h_i^geo = Σ_j α_ij · P(v_j)` (融合空间上下文)

### 2.5 `RADM/layout_decoder.py` — 布局编码与多模态解码器

**对应论文章节**: §布局表示与多模态解码

| 类名 | 功能 |
|------|------|
| `FourierFeatureEncoding` | 傅里叶特征映射: `Fourier(b_i) = [sin(2π·B·b_i), cos(2π·B·b_i)]`，B为固定高斯随机矩阵 |
| `LayoutEncoder` | 布局Token编码: `h_i^l = MLP([t_i ‖ Fourier(b_i)])`，CLIP文本 + Fourier坐标 |
| `JointCrossAttention` | MM-DiT风格联合注意力: `[H_l', H_c'] = Attention([Q_l,Q_c], [K_l,K_c], [V_l,V_c])` |
| `MultiModalInteractionBranch` | 单模态交互分支(CrossAttention + FFN) |
| `MultiModalDecoder` | 三支路并行解码器(视觉/拓扑/几何) + 融合 |
| `LayoutGenerationHead` | 完整布局生成头(Encoder + Decoder) |

**三支路交互**: 视觉特征 `F_vis`、拓扑特征 `H_topo`、几何特征 `H_geo` 分别通过独立CrossAttention与布局Token交互。

**最终融合**: `H_final = Concat(H_l^vis, H_l^topo, H_l^geo) · W_fuse` → MLP预测头 → 边界框坐标

---

## 三、修改文件

### 3.1 `RADM/config.py`

新增配置组:

```yaml
# RGCN
MODEL.RGCN.ENABLED / NUM_LAYERS / HIDDEN_DIM / NUM_RELATIONS / NUM_BASES / DROPOUT

# 辅助损失
MODEL.AUX_LOSS.RELATION_RECONSTRUCT / GRID_POSITION / LAMBDA_REL / LAMBDA_POS / GRID_SIZE

# 视觉编码器
MODEL.VISUAL_ENCODER.ENABLED / MODEL_NAME / PATCH_SIZE / FEATURE_DIM / OUT_CHANNELS / FREEZE_BACKBONE / ROI_OUTPUT_SIZE

# 几何关系模块
MODEL.GEO_RELATION.ENABLED / EMBED_DIM / WAVE_LENGTH / FC_OUT_CHANNELS / OUT_DIM

# 布局解码器
MODEL.LAYOUT_DECODER.ENABLED / TEXT_ENCODER / TEXT_DIM / FOURIER_SCALE / FOURIER_DIM / MM_NUM_HEADS / MM_DROPOUT / FUSION_MODE

# RADM标志
MODEL.RADM.withRGCN / WITH_ENHANCED_GEO
```

### 3.2 `RADM/detector.py`

**RADM类初始化** (新增约80行):
- 条件初始化 `RGCN`、`ConstraintGraphBuilder`、`RelationClassifier`、`GridPositionPredictor`
- 条件初始化 `GeometryRelationModule`、`LayoutGenerationHead`、`VisualPriorEncoder`
- 条件初始化 `GraphEncodingLoss`

**新增方法**:
| 方法 | 功能 |
|------|------|
| `prepare_graph_targets()` | 从GT实例提取图构建所需数据 |
| `compute_graph_encoding()` | 构建约束图并执行RGCN编码 |
| `compute_graph_aux_losses()` | 计算关系重构损失和网格位置损失 |

**修改方法**:
| 方法 | 变更 |
|------|------|
| `model_predictions()` | 新增 `H_topo_list`、`H_geo_list` 参数，传入head |
| `ddim_sample()` | 推理时动态构建约束图，计算拓扑/几何条件特征 |
| `forward()` (train) | 新增图编码辅助损失计算并合并到loss_dict |

### 3.3 `RADM/head.py`

**RCNNHead类**:
- 新增 `self.withRGCN`、`self.withEnhancedGeo` 属性
- `d_fused` 维度动态扩展：VTRAM(+d_model), GRAM(+topo_out_dim), RGCN(+d_model), EnhancedGeo(+d_model)
- `forward()` 新增 `H_topo_list`、`H_geo_list` 参数
- fc_feature拼接: 依次拼接VTRAM特征、GRAM特征、H_topo特征、H_geo特征

**DynamicHead类**:
- `forward()` 新增 `H_topo_list`、`H_geo_list` 参数，透传到RCNNHead

### 3.4 `RADM/loss.py`

新增类:

| 类名 | 功能 |
|------|------|
| `GraphEncodingLoss` | 综合图编码辅助损失：`L_enc = λ_rel·L_rel + λ_pos·L_pos` |
| `CombinedCriterion` | 组合检测损失和图编码损失的联合评价准则 |
| `build_graph_encoding_loss()` | 从config构建GraphEncodingLoss的工厂函数 |

**损失公式实现**:

**关系重构损失** L_rel:
```
L_rel = -1/|E| Σ_{(i,j)∈E} Σ_c y_{ij,c}·log(ŷ_{ij,c})
```
→ 只对存在边的节点对计算CrossEntropyLoss

**网格位置分类损失** L_pos:
```
L_pos = -1/|V| Σ_i Σ_k I(t_i=k)·log(p̂_{i,k})
```
→ 将画布离散为 S×S 网格，预测每个节点所属网格

### 3.5 `RADM/__init__.py`

新增导出:
```python
from .rgcn import RGCN, RGCNLayer, RelationClassifier, GridPositionPredictor
from .constraint_graph import ConstraintGraphBuilder
from .geometry_relation import GeometryRelationModule
from .layout_decoder import LayoutEncoder, MultiModalDecoder, LayoutGenerationHead, FourierFeatureEncoding
from .loss import GraphEncodingLoss, CombinedCriterion, build_graph_encoding_loss
```

### 3.6 `configs/radm.yaml`

新增配置项:
```yaml
RADM:
  withRGCN: True
  WITH_ENHANCED_GEO: True
```

---

## 四、配置开关说明

| 开关 | 默认值 | 说明 |
|------|--------|------|
| `MODEL.RADM.withRGCN` | True | 启用RGCN拓扑推理特征 |
| `MODEL.RADM.WITH_ENHANCED_GEO` | True | 启用增强几何关系特征 |
| `MODEL.RGCN.ENABLED` | True | RGCN模块开关 |
| `MODEL.AUX_LOSS.RELATION_RECONSTRUCT` | True | 启用关系重构辅助损失 |
| `MODEL.AUX_LOSS.GRID_POSITION` | True | 启用网格位置辅助损失 |
| `MODEL.VISUAL_ENCODER.ENABLED` | True | 启用DINO-ViT视觉编码器(需安装dinov2) |
| `MODEL.GEO_RELATION.ENABLED` | True | 启用几何关系模块 |
| `MODEL.LAYOUT_DECODER.ENABLED` | True | 启用多模态布局解码器 |

---

## 五、数据流

### 训练阶段

```
输入: image + text_features + gt_instances
│
├─ Backbone → features [P2,P3,P4,P5]
│
├─ prepare_targets() → diffused_boxes, noises, t
│
├─ ConstraintGraphBuilder(text, gt_boxes, gt_classes)
│  └─ node_features + adj_mats + edge_labels
│
├─ RGCN(node_features, adj_mats) → H_topo
│
├─ DynamicHead(features, boxes, H_topo, H_geo) → class_logits, pred_boxes
│
├─ SetCriterionDynamicK(pred, targets) → detection_losses  }
│                                                          } → total_loss
└─ GraphEncodingLoss(H_topo, edge_labels, adj_mats, boxes) }
   ├─ RelationClassifier → L_rel
   └─ GridPositionPredictor → L_pos
```

### 推理阶段

```
输入: image + text_features
│
├─ Backbone → features
│
├─ ddim_sample() loop:
│  ├─ img (current noisy boxes)
│  ├─ ConstraintGraphBuilder(text, img, class_ids) → node_feat, adj_mats
│  ├─ RGCN(node_feat, adj_mats) → H_topo
│  ├─ GeometryRelationModule(H_topo, img) → H_geo
│  ├─ model_predictions(..., H_topo, H_geo) → pred_noise, x_start
│  └─ DDIM update step
│
└─ inference() → bbox + scores + classes
```

---

## 六、依赖要求

| 依赖 | 用途 | 必需 |
|------|------|------|
| `torch >= 1.10` | 基础框架 | ✓ |
| `detectron2` | 检测框架基础 | ✓ |
| `fvcore` | 损失函数 | ✓ |
| `torchvision` | RoIAlign | ✓ |
| `dinov2` (可选) | DINO-ViT视觉编码器 | 仅视觉编码器 |

---

## 七、文件清单

```
RADM/
├── __init__.py              # ✏️ 修改 - 新增模块导出
├── config.py                # ✏️ 修改 - 新增配置组
├── detector.py              # ✏️ 修改 - 集成新模块
├── head.py                  # ✏️ 修改 - 接收拓扑/几何特征
├── loss.py                  # ✏️ 修改 - 新增辅助损失
├── rgcn.py                  # 🆕 新建 - RGCN图神经网络
├── constraint_graph.py      # 🆕 新建 - 约束图构建器
├── visual_encoder.py        # 🆕 新建 - DINO-ViT视觉编码器
├── geometry_relation.py     # 🆕 新建 - 几何关系感知模块
├── layout_decoder.py        # 🆕 新建 - 布局编码与多模态解码器
├── dataset_mapper.py        # 未修改
├── evaluator.py             # 未修改
├── test_time_augmentation.py # 未修改
└── util/                    # 未修改

configs/
└── radm.yaml                # ✏️ 修改 - 新增配置项
```

---

## 八、测试验证

所有新增模块已通过以下验证:

- ✅ Python语法检查 (`py_compile`) — 8/8文件通过
- ✅ 核心模块运行时Smoke Test:
  - `RGCN`: 前向传播形状正确 `(10, 256)`
  - `ConstraintGraphBuilder`: 图构建输出正确 `nodes(10,256), adj(3,10,10), edges(10,10)`
  - `GeometryRelationModule`: 几何特征输出正确 `(10, 256)`
  - `LayoutEncoder`: 布局Token输出正确 `(10, 256)`
  - `LayoutGenerationHead`: bbox预测 `(2,10,4)` + 融合特征 `(2,10,256)`
  - `GraphEncodingLoss`: 损失计算正常
