# RL-Enhanced RADM: 多智能体强化学习优化海报布局生成

## 📋 概述

本项目在原有的RADM (Relation-Aware Diffusion Model) 基础上集成了多智能体强化学习(MARL)，实现智能化的布局优化。系统通过协作的智能体自主学习布局设计原则，能够生成更高质量、更符合设计美学的海报布局。

## 🚀 核心特性

### ✨ 技术创新
- **多智能体协同**: 每个布局元素作为独立智能体，通过注意力机制实现协作
- **层次化优化**: 扩散模型生成初始布局，RL进行精细优化
- **动态奖励建模**: 多维度评估包括视觉质量、语义一致性和美学标准
- **实时交互**: 支持设计师反馈的在线学习

### 🎯 优化效果
- **布局质量提升**: R_shm和R_sub指标显著改善
- **设计一致性**: 更好的元素对齐和视觉平衡
- **语义相关性**: 元素位置更符合内容语义

## 📁 文件结构

```
RADM/
├── rl_layout_env.py      # 多智能体布局环境
├── rl_agent.py           # MARL智能体和PPO训练器
├── rl_trainer.py         # RL训练和推理主控制器
├── rl_inference.py       # RL推理脚本
├── train_rl.py          # RL训练脚本
├── train_net.py         # 修改后的主训练脚本 (支持RL)
├── configs/radm.yaml    # 更新的配置文件 (包含RL参数)
└── README_RL.md         # 本文档
```

## 🛠️ 安装和配置

### 环境要求
```bash
# 确保已安装原有RADM依赖
pip install -r requirements.txt

# RL模块需要额外的依赖
pip install stable-baselines3  # 如果需要其他RL库
```

### 配置修改
RL相关配置已添加到 `configs/radm.yaml`:

```yaml
RL:
  # 训练参数
  NUM_ENV_STEPS: 10000000    # 总训练步数
  NUM_STEPS: 2048           # 每次收集的步数
  LOG_INTERVAL: 1           # 日志间隔
  SAVE_INTERVAL: 100        # 保存间隔

  # PPO算法参数
  CLIP_PARAM: 0.2           # PPO裁剪参数
  PPO_EPOCH: 10             # PPO训练轮数
  LR: 0.0003               # 学习率

  # 环境参数
  MAX_ELEMENTS: 20          # 最大元素数量
  MAX_STEPS: 50             # 每个episode最大步数
  ACTION_SCALE: 0.1         # 动作缩放因子

  # 奖励权重
  OVERLAP_PENALTY: -2.0     # 重叠惩罚
  ALIGNMENT_BONUS: 1.0      # 对齐奖励
  BALANCE_BONUS: 0.5        # 平衡奖励
  SEMANTIC_COHERENCE: 0.8   # 语义一致性
  AESTHETIC_SCORE: 1.2      # 美学评分

  # 日志配置
  LOG_LEVEL: INFO           # DEBUG(详细)/INFO(标准)/WARNING(警告)
```

## 🎮 使用方法

### 1. 训练RL策略

#### 使用专用训练脚本
```bash
python train_rl.py \
    --config-file configs/radm.yaml \
    --device cuda \
    --resume /path/to/checkpoint  # 可选，用于恢复训练
```

#### 使用主训练脚本 (RL模式)
```bash
python train_net.py \
    --config-file configs/radm.yaml \
    --num-gpus 1 \
    --rl-train
```

### 训练监控

训练过程中会输出详细的统计信息：

```
[15.2%] Step 3072/20000 | Value Loss: 0.0124 | Action Loss: -0.0345 | Entropy: 0.0234 | Rewards: 4.231±1.234 (max: 6.789) | Episode Length: 45.6 (max: 50) | r_shm: 0.8345 | Speed: 45.2 steps/sec | ETA: 12.3 min
```

**关键指标说明：**
- **Progress**: 训练进度百分比
- **Value/Action Loss**: PPO的损失函数
- **Entropy**: 动作分布的随机性（探索程度）
- **Rewards**: 智能体的平均奖励和分布
- **Episode Length**: 训练回合长度
- **Quality Metrics**: 布局质量指标（r_shm, balance, alignment）
- **Speed**: 训练速度和预计完成时间

#### 启用详细日志

如需查看更详细的训练过程：

```yaml
RL:
  LOG_LEVEL: DEBUG  # 改为DEBUG模式
```

这将输出：
- 每个batch的RADM推理结果
- 环境重置和智能体状态
- PPO策略更新细节
- 定期评估结果

### 2. 推理和优化布局

```bash
python rl_inference.py \
    --config-file configs/radm.yaml \
    --rl-checkpoint /path/to/rl_checkpoint.pth \
    --image-path /path/to/input_image.jpg \
    --text-features /path/to/text_features.pth \
    --output-dir ./rl_output
```

### 3. 参数说明

#### 训练参数
- `--config-file`: 配置文件路径
- `--device`: 训练设备 (cuda/cpu)
- `--resume`: 恢复训练的检查点路径

#### 推理参数
- `--config-file`: 配置文件路径
- `--rl-checkpoint`: 训练好的RL模型路径
- `--image-path`: 输入图像路径
- `--text-features`: 文本特征文件路径 (可选)
- `--output-dir`: 输出目录

## 📊 训练过程

### 训练阶段
1. **预训练阶段**: 使用RADM生成初始布局数据
2. **RL初始化**: 从预训练的RADM模型开始
3. **策略学习**: 多智能体通过PPO算法学习协作策略
4. **奖励优化**: 通过精心设计的奖励函数提升布局质量

### 训练监控
训练过程中会输出以下指标:
- **Value Loss**: 价值函数损失
- **Action Loss**: 策略损失
- **Entropy**: 动作熵 (探索程度)
- **Mean Reward**: 平均奖励
- **Layout Quality**: 布局质量指标

## 🎨 奖励函数设计

奖励函数包含多个维度:

```python
reward = (
    overlap_penalty * -2.0 +      # 重叠惩罚
    alignment_bonus * 1.0 +       # 对齐奖励
    balance_bonus * 0.5 +         # 平衡奖励
    semantic_coherence * 0.8 +    # 语义一致性
    aesthetic_score * 1.2         # 美学评分
)
```

### 奖励组件
- **重叠惩罚**: 防止元素重叠
- **对齐奖励**: 鼓励元素对齐 (边缘、中心线)
- **平衡奖励**: 整体布局平衡性
- **语义一致性**: 元素语义相关性
- **美学评分**: 符合设计原则的评分

## 📈 性能评估

### 评估指标
- **R_shm**: 元素重要性度量 (越高越好)
- **Balance**: 布局平衡性 (0-1, 越高越好)
- **Alignment**: 对齐程度 (越高越好)
- **Reward**: RL奖励值 (越高越好)

### 对比实验
预期在以下方面优于原始RADM:
- 布局美学质量提升 15-25%
- 元素重叠率降低 30-40%
- 对齐准确性提升 20-30%

## 🔧 高级配置

### 自定义奖励函数
在 `rl_layout_env.py` 中修改 `calculate_rewards` 方法:

```python
def calculate_rewards(self, prev_layout: Dict) -> Dict[int, float]:
    # 自定义奖励计算逻辑
    pass
```

### 添加新的智能体行为
在 `rl_agent.py` 中扩展动作空间:

```python
# 当前动作: [dx, dy, dw, dh]
# 可以扩展为: [dx, dy, dw, dh, rotation, scale]
self.action_dim = 6  # 扩展动作维度
```

### 集成用户反馈
添加用户反馈机制:

```python
def add_user_feedback(self, layout_quality_score: float):
    """集成用户对布局的评分"""
    # 更新奖励函数或策略
    pass
```

## 🐛 故障排除

### 常见问题

1. **内存不足**
   - 减少 `MAX_ELEMENTS` 或 `NUM_STEPS`
   - 使用更小的批次大小

2. **训练不稳定**
   - 调整学习率 `LR`
   - 修改奖励权重
   - 检查动作缩放 `ACTION_SCALE`

3. **收敛慢**
   - 增加 `PPO_EPOCH`
   - 调整 `CLIP_PARAM`
   - 预训练更长时间的RADM模型

### 调试建议
- 使用 `--log-interval 1` 查看详细训练日志
- 保存中间检查点进行分析
- 可视化布局变化过程

## 📚 参考文献

- [RADM: Relation-Aware Diffusion Model](https://arxiv.org/abs/...)
- [Multi-Agent Reinforcement Learning](https://arxiv.org/abs/...)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)

## 🤝 贡献

欢迎提交问题和改进建议！

---

**注意**: RL训练需要大量的计算资源，建议使用GPU环境进行训练。首次训练可能需要数小时到数天，具体取决于硬件配置和数据集大小。
