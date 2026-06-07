# RADM 项目运行说明

> 基于论文 _Relation-Aware Diffusion Model for Controllable Poster Layout Generation_ (CIKM 2023)
> 代码改写后版本，新增 RGCN 约束图编码、视觉先验编码、几何关系感知、多模态布局解码等模块。

---

## 一、环境要求

| 依赖 | 版本要求 | 说明 |
|------|---------|------|
| Python | 3.7+ | 推荐 3.8 |
| PyTorch | 1.8.0+ | CUDA 11.1 |
| detectron2 | — | [安装指南](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) |
| fvcore | — | detectron2 依赖 |
| torchvision | — | RoIAlign 依赖 |
| numpy, opencv, Pillow | — | 数据处理 |
| pycocotools | — | COCO 格式评估 |
| dinov2 | 可选 | DINO-ViT 视觉编码器 (`pip install dinov2`) |

---

## 二、安装步骤

### 1. 创建并激活 conda 环境

```bash
conda activate radm
```

### 2. 安装 detectron2

根据 CUDA 和 PyTorch 版本选择对应命令，参考 [detectron2 安装文档](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)：

```bash
# CUDA 11.1 + PyTorch 1.8 示例
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
```

### 3. 安装其他依赖

```bash
cd /home/sxm/data02Space/GraduationProject/textToLayoutBaseRADM/RADM
pip install -r requirements.txt
```

### 4. (可选) 安装 DINOv2 以启用视觉先验编码模块

```bash
pip install dinov2
```

---

## 三、数据准备

### 数据集目录结构

```
DatasetRoot/
├── annotations/
│   ├── train.json              # COCO 格式训练标注
│   └── test.json               # COCO 格式测试标注
├── images/
│   ├── train/                  # 训练图像（经 LaMa 擦除文字后的干净背景图）
│   └── test/                   # 测试图像
├── text_content/
│   ├── train.txt               # 训练集文本内容
│   └── test.txt                # 测试集文本内容
└── text_features/
    ├── train/                  # 预提取的训练文本特征 (*_feats.pth)
    └── test/                   # 预提取的测试文本特征 (*_feats.pth)
```

### 数据获取

- **训练数据**: 从 [CGL-dataset](https://tianchi.aliyun.com/dataset/142692) 下载，使用 [LaMa](https://github.com/advimman/lama) 擦除图片中的文字得到干净背景图
- **测试数据 + 文本特征**: 从 [此链接](https://3.cn/10-dQKDKG) 下载并解压到 `DatasetRoot`

### 文本特征格式

每个 `*_feats.pth` 文件包含一个字典：

```python
{
    'feats': [tensor(1, 768), tensor(1, 768), ...],  # 每条文本的 CLIP 特征
}
```

---

## 四、配置文件修改

编辑 `configs/radm.yaml`，将 `${yourPath}` 替换为实际路径：

```yaml
_BASE_: "Base-RADM.yaml"
MODEL:
  RESNETS:
    DEPTH: 50
    STRIDE_IN_1X1: False
  RADM:
    withVTRAM: True          # 视觉-文本关系感知模块
    withGRAM: True           # 几何关系感知模块 (原始)
    withRGCN: True           # 🆕 RGCN 拓扑推理
    WITH_ENHANCED_GEO: True  # 🆕 增强几何关系
    NUM_PROPOSALS: 100
    NUM_CLASSES: 4
    NMS_THRESH: 0.15
    CLASS_THRESH: 0.25

DATASETS:
  TEXT_FEATURE_PATH: '/data/DatasetRoot/text_features'   # ← 修改为实际路径
  DATASET_PATH: '/data/DatasetRoot'                       # ← 修改为实际路径
  TRAIN: ("layout_train",)
  TEST:  ("layout_val",)

SOLVER:
  STEPS: (150000, 220000)
  MAX_ITER: 250000

OUTPUT_DIR: "./output"    # 模型权重和日志输出目录
```

`Base-RADM.yaml` 中定义了基础训练超参数（无需修改）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `SOLVER.IMS_PER_BATCH` | 16 | 每 batch 图片数 |
| `SOLVER.BASE_LR` | 2.5e-5 | 基础学习率 |
| `SOLVER.MAX_ITER` | 270000 | 最大迭代数 |
| `SOLVER.OPTIMIZER` | ADAMW | 优化器 |
| `SOLVER.CLIP_GRADIENTS.ENABLED` | True | 梯度裁剪 |
| `TEST.EVAL_PERIOD` | 5000 | 每 N iter 评估一次 |

---

## 五、训练

### 单卡训练

```bash
cd /home/sxm/data02Space/GraduationProject/textToLayoutBaseRADM/RADM

python3 train_net.py --num-gpus 1 \
    --config-file configs/radm.yaml
```

### 多卡训练

```bash
# 4 卡训练
python3 train_net.py --num-gpus 4 \
    --config-file configs/radm.yaml
```

### 从 checkpoint 恢复训练

```bash
python3 train_net.py --num-gpus 4 \
    --config-file configs/radm.yaml \
    --resume
```

### 训练过程说明

- 使用 detectron2 的 `launch()` 启动分布式训练
- 每 `EVAL_PERIOD` (默认 5000) 次迭代在验证集上评估
- checkpoint 保存在 `OUTPUT_DIR` 中
- 日志包含 loss 值（含新增的 `loss_relation_reconstruct`、`loss_grid_position` 等辅助损失）

---

## 六、推理与评估

### 模型推理

```bash
python3 train_net.py --num-gpus 1 \
    --config-file configs/radm.yaml \
    --eval-only --resume
```

推理结果输出到 `OUTPUT_DIR/inference/coco_instances_results.json`。

### 计算布局评估指标

先修改 `metrics.py` 中的路径（第 220–222 行）：

```python
test_imgdir = '/data/DatasetRoot/images/test'
test_label = '/data/DatasetRoot/annotations/test.json'
test_annotation = './output/inference/coco_instances_results.json'
vis_example = './vis_example/'
```

然后运行：

```bash
python3 metrics.py
```

输出指标：

| 指标 | 含义 |
|------|------|
| **R_occ** | 遮挡率 — 预测到的元素比例 |
| **R_com** | 复杂度 — 文本区域背景的 Sobel 梯度均值 |
| **R_ove** | 重叠率 — 非衬底/非装饰元素间的平均重叠比 |
| **R_und** | 衬底覆盖率 — 衬底元素与其他元素的最大重叠度 |
| **R_ali** | 对齐度 — 元素间间距的负对数度量 |

可视化结果保存在 `vis_example/` 目录。

---

## 七、新增模块配置开关

本次改写新增的模块均可独立开关，支持通过命令行覆盖配置：

```bash
# 关闭 RGCN 拓扑推理
python3 train_net.py ... MODEL.RGCN.ENABLED False MODEL.RADM.withRGCN False

# 关闭增强几何关系模块
python3 train_net.py ... MODEL.GEO_RELATION.ENABLED False MODEL.RADM.WITH_ENHANCED_GEO False

# 关闭视觉先验编码器
python3 train_net.py ... MODEL.VISUAL_ENCODER.ENABLED False

# 关闭辅助监督损失
python3 train_net.py ... MODEL.AUX_LOSS.RELATION_RECONSTRUCT False MODEL.AUX_LOSS.GRID_POSITION False

# 调整辅助损失权重
python3 train_net.py ... MODEL.AUX_LOSS.LAMBDA_REL 0.5 MODEL.AUX_LOSS.LAMBDA_POS 0.5

# 调整 RGCN 层数
python3 train_net.py ... MODEL.RGCN.NUM_LAYERS 3

# 调整网格大小
python3 train_net.py ... MODEL.AUX_LOSS.GRID_SIZE 16
```

### 完整配置项列表

| 配置路径 | 默认值 | 说明 |
|---------|--------|------|
| `MODEL.RADM.withRGCN` | True | 启用 RGCN 拓扑特征注入 head |
| `MODEL.RADM.WITH_ENHANCED_GEO` | True | 启用增强几何特征注入 head |
| `MODEL.RGCN.ENABLED` | True | RGCN 模块开关 |
| `MODEL.RGCN.NUM_LAYERS` | 2 | RGCN 消息传递层数 |
| `MODEL.RGCN.HIDDEN_DIM` | 256 | RGCN 隐层维度 |
| `MODEL.RGCN.NUM_RELATIONS` | 3 | 关系类型数 (BB/BF/FF) |
| `MODEL.RGCN.NUM_BASES` | 4 | 基分解数 |
| `MODEL.RGCN.DROPOUT` | 0.1 | RGCN dropout |
| `MODEL.AUX_LOSS.RELATION_RECONSTRUCT` | True | 关系重构辅助损失 |
| `MODEL.AUX_LOSS.GRID_POSITION` | True | 网格位置辅助损失 |
| `MODEL.AUX_LOSS.LAMBDA_REL` | 1.0 | 关系损失权重 |
| `MODEL.AUX_LOSS.LAMBDA_POS` | 1.0 | 位置损失权重 |
| `MODEL.AUX_LOSS.GRID_SIZE` | 8 | 画布网格 S×S |
| `MODEL.VISUAL_ENCODER.ENABLED` | True | DINO-ViT 视觉编码器 |
| `MODEL.VISUAL_ENCODER.MODEL_NAME` | dinov2_vits14 | DINOv2 模型变体 |
| `MODEL.VISUAL_ENCODER.FREEZE_BACKBONE` | True | 冻结 ViT 权重 |
| `MODEL.GEO_RELATION.ENABLED` | True | 增强几何关系模块 |
| `MODEL.GEO_RELATION.EMBED_DIM` | 64 | PE 嵌入维度 |
| `MODEL.GEO_RELATION.OUT_DIM` | 256 | 几何特征输出维度 |
| `MODEL.LAYOUT_DECODER.ENABLED` | True | 多模态布局解码器 |
| `MODEL.LAYOUT_DECODER.TEXT_DIM` | 512 | CLIP 文本特征维度 |
| `MODEL.LAYOUT_DECODER.FOURIER_DIM` | 128 | Fourier 特征维度 |
| `MODEL.LAYOUT_DECODER.MM_NUM_HEADS` | 8 | 多模态注意力头数 |

---

## 八、常见问题

### Q1: 报错 `ModuleNotFoundError: No module named 'detectron2'`

需要在 `radm` conda 环境中安装 detectron2，参考[官方安装指南](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)。

### Q2: 报错 `ModuleNotFoundError: No module named 'dinov2'`

DINOv2 是可选的视觉编码器依赖。可以安装 `pip install dinov2`，或在配置中关闭：`MODEL.VISUAL_ENCODER.ENABLED False`。关闭后模型仍可正常运行（仅缺少视觉先验条件）。

### Q3: 数据集路径找不到

确保 `configs/radm.yaml` 中的 `DATASETS.DATASET_PATH` 和 `DATASETS.TEXT_FEATURE_PATH` 已修改为实际路径。

### Q4: 文本特征文件缺失

文本特征文件需要预先用 CLIP 文本编码器提取。每个 `.pth` 文件对应一张图片，包含该图中所有文本元素的特征。如果缺少，`DatasetMapper` 会用零向量填充。

### Q5: 如何验证新模块是否正常工作

```bash
conda activate radm
cd /home/sxm/data02Space/GraduationProject/textToLayoutBaseRADM/RADM

python3 -c "
import torch
from RADM.rgcn import RGCN
from RADM.constraint_graph import ConstraintGraphBuilder
from RADM.geometry_relation import GeometryRelationModule
from RADM.layout_decoder import LayoutGenerationHead
from RADM.loss import GraphEncodingLoss

# RGCN 测试
rgcn = RGCN(in_dim=256, hidden_dim=256, out_dim=256)
out = rgcn(torch.randn(10, 256), torch.randint(0, 2, (3, 10, 10)).float())
print(f'RGCN: {out.shape}')  # → torch.Size([10, 256])

# 约束图构建测试
builder = ConstraintGraphBuilder(text_dim=768, hidden_dim=256)
nf, am, el = builder(torch.randn(10, 768), torch.rand(10, 4), torch.randint(0, 4, (10,)))
print(f'Graph: nodes={nf.shape}, adj={am.shape}, edges={el.shape}')

# 几何关系测试
geo = GeometryRelationModule(in_channels=256, embed_dim=64, out_dim=256)
print(f'Geo: {geo(torch.randn(10, 256), torch.rand(10, 4)).shape}')

# 布局生成头测试
head = LayoutGenerationHead(d_model=256, text_dim=512)
bp, hf = head(torch.randn(2, 10, 512), torch.rand(2, 10, 4),
              torch.randn(2, 10, 256), torch.randn(2, 10, 256))
print(f'LayoutHead: bbox={bp.shape}, final={hf.shape}')

# 辅助损失测试
gl = GraphEncodingLoss()
adj = torch.zeros(3, 10, 10); adj[0, 0, 1] = adj[1, 1, 0] = 1.0
loss, d = gl(torch.randn(10, 256), torch.randint(0, 8, (10, 10)), adj, torch.rand(10, 4))
print(f'GraphLoss: {loss.item():.4f}, keys={list(d.keys())}')

print('All new modules OK!')
```
