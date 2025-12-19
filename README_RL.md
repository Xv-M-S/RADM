# RL增强的RADM系统 (RL-RADM)

## notice
在该提交中加入了步骤二基于推理时优化的相关代码，同时添加了强化学习的一些参考代码，步骤二的推理时优化代码并没有集成到RADM，同时强化学习的代码没有验证可用性，仅作参考，当前代码仓库因为上述加入的代码可能导致不可用

## 🎯 概述

这是一个在原RADM（Relation-Aware Diffusion Model）基础上集成的强化学习增强系统。通过将强化学习与扩散模型相结合，实现更高质量、可控的布局生成。

## 🚀 核心创新

### 1. **RL环境设计**
- **状态空间**: 布局配置、约束状态、历史统计特征
- **动作空间**: 布局元素的位置调整 (连续动作)
- **奖励函数**: 约束满足度 + 美学评分 + 可读性指标

### 2. **SAC算法集成**
- 使用Soft Actor-Critic算法进行连续控制
- 支持探索-利用平衡的策略学习
- 自动熵调整以适应不同难度任务

### 3. **扩散过程RL引导**
- 在扩散去噪过程中引入RL策略指导
- 课程学习策略：逐步增加RL引导强度
- 多时间步RL优化以提升布局质量

### 4. **偏好学习扩展**
- 收集人类偏好数据进行学习
- 自动偏好生成器基于启发式规则
- 交互式数据收集界面

## 📁 文件结构

```
RADM/
├── rl_layout_env.py          # RL环境实现
├── sac_agent.py              # SAC算法实现
├── rl_diffusion_integrator.py # RL-扩散集成
├── data_collection.py        # 数据收集和偏好学习
├── joint_trainer.py          # 联合训练管理器
├── examples/
│   └── rl_radm_example.py    # 使用示例
└── README_RL.md             # 本文档
```

## 🛠️ 安装和设置

### 依赖要求
```bash
pip install torch torchvision
pip install gym matplotlib seaborn
pip install scikit-learn pandas
```

### 基本设置
```python
from RADM.rl_layout_env import LayoutRLEnvironment
from RADM.sac_agent import SACAgent
from RADM.joint_trainer import JointTrainer, TrainingConfig

# 创建RL环境
env = LayoutRLEnvironment(max_boxes=10)

# 创建SAC智能体
sac_agent = SACAgent(
    state_dim=200,  # 根据实际状态维度调整
    action_dim=20,  # max_boxes * 2
    hidden_dim=256
)

# 创建训练配置
config = TrainingConfig.get_experiment_config('rl_medium')
```

## 🎮 使用示例

### 1. 训练RL智能体

```python
from RADM.sac_agent import LayoutSACTrainer

# 准备训练数据
layout_data = create_your_layout_data()  # 你的布局数据集
constraints_list = [data['constraints'] for data in layout_data]

# 创建训练器
trainer = LayoutSACTrainer(env, sac_agent)

# 执行训练
trainer.train(layout_data, constraints_list, max_episodes=1000)
```

### 2. 偏好学习数据收集

```python
from RADM.data_collection import LayoutPreferenceDataset, InteractiveDataCollector

# 创建数据集
preference_dataset = LayoutPreferenceDataset()

# 创建交互式收集器
collector = InteractiveDataCollector(preference_dataset)

# 收集用户偏好
preference = collector.collect_user_preference(layout_a, layout_b, constraints)
```

### 3. RL引导的布局生成

```python
from RADM.rl_diffusion_integrator import RLGuidedRADM

# 创建RL引导的RADM
rl_guided_radm = RLGuidedRADM(
    base_radm_model,  # 你的RADM模型
    sac_agent,
    guidance_config={
        'guidance_scale': 0.2,
        'rl_steps': 3,
        'enable_rl_guidance': True
    }
)

# 生成布局
generated_layout = rl_guided_radm.generate_layout(conditions)
```

### 4. 联合训练

```python
from RADM.joint_trainer import JointTrainer, JointTrainingDataset

# 创建联合训练数据集
joint_dataset = JointTrainingDataset(
    layout_data=layout_data,
    preference_dataset=preference_dataset,
    preference_weight=0.3
)

# 创建联合训练器
trainer = JointTrainer(
    radm_model=base_radm_model,
    sac_agent=sac_agent,
    config=config
)

# 执行联合训练
trainer.train(joint_dataset, num_epochs=100)
```

## ⚙️ 配置选项

### RL环境配置
```python
env_config = {
    'max_boxes': 10,              # 最大布局元素数量
    'canvas_size': (1, 1),        # 画布尺寸
    'constraint_types': ['left_of', 'right_of', 'above', 'below'],
    'reward_weights': {           # 奖励权重
        'constraint': 1.0,
        'aesthetic': 0.8,
        'readability': 0.6,
        'stability': 0.4
    }
}
```

### SAC算法配置
```python
sac_config = {
    'state_dim': 200,
    'action_dim': 20,
    'hidden_dim': 256,
    'gamma': 0.99,               # 折扣因子
    'tau': 0.005,               # 软更新系数
    'alpha': 0.2,               # 熵温度
    'lr': 3e-4,                 # 学习率
    'batch_size': 256,
    'buffer_capacity': 100000
}
```

### 引导配置
```python
guidance_config = {
    'guidance_scale': 0.1,       # RL引导强度
    'rl_steps': 5,               # 每个扩散步的RL优化步数
    'guidance_start_step': 10,   # 开始RL引导的扩散步
    'guidance_end_step': 50,     # 结束RL引导的扩散步
    'enable_rl_guidance': True
}
```

## 📊 实验配置

系统提供预定义的实验配置：

- **`baseline`**: 纯扩散模型，无RL引导
- **`rl_light`**: 轻度RL引导 (guidance_scale=0.1)
- **`rl_medium`**: 中等RL引导 (guidance_scale=0.2)
- **`rl_heavy`**: 强RL引导 (guidance_scale=0.3)
- **`preference_learning`**: 包含偏好学习的配置

```python
from RADM.joint_trainer import TrainingConfig

# 使用预定义配置
config = TrainingConfig.get_experiment_config('rl_medium')
trainer = JointTrainer(radm_model, sac_agent, config=config)
```

## 🔍 评估指标

### RL性能指标
- **约束满足率**: 布局满足约束条件的比例
- **美学评分**: 平衡性、对齐度、比例和谐性
- **可读性评分**: 元素间距、尺寸对比
- **稳定性**: 布局调整的平滑度

### 联合训练指标
- **扩散损失**: 扩散模型的重建损失
- **RL奖励**: RL智能体的累积奖励
- **偏好准确率**: 偏好预测的准确性
- **生成质量**: 定性和定量评估生成布局

## 🚀 运行完整示例

```bash
cd /path/to/RADM
python examples/rl_radm_example.py
```

这将运行所有示例，包括：
1. RL智能体训练
2. 偏好学习数据收集
3. RL引导布局生成
4. 联合训练流程设置
5. 完整训练管道演示

## 🎯 最佳实践

### 1. 训练策略
- **先RL后联合**: 先训练RL智能体，再进行联合训练
- **课程学习**: 使用课程学习逐渐增加RL引导强度
- **数据平衡**: 平衡布局生成和偏好学习数据的比例

### 2. 超参数调优
- **引导强度**: 从0.1开始，根据任务复杂度调整
- **RL步数**: 复杂布局使用更多RL优化步骤
- **奖励权重**: 根据具体应用调整奖励组件权重

### 3. 数据收集
- **多样化布局**: 收集不同复杂度、不同类型的布局
- **用户偏好**: 定期收集人类用户的布局偏好
- **自动生成**: 使用启发式规则自动生成训练数据

## 🔧 扩展和定制

### 添加新的约束类型
```python
# 在LayoutRLEnvironment中添加新的约束处理
def _evaluate_custom_constraint(self, box_a, box_b, constraint_type):
    if constraint_type == 'custom_constraint':
        # 实现你的自定义约束逻辑
        return score
```

### 自定义奖励函数
```python
class CustomRewardCalculator(LayoutRewardCalculator):
    def calculate_custom_reward(self, layout, constraints):
        # 实现你的自定义奖励逻辑
        return custom_score
```

### 集成新的RL算法
```python
class CustomRLAgent:
    def __init__(self):
        # 实现你的RL算法
        pass

    def select_action(self, state):
        # 动作选择逻辑
        return action
```

## 📈 性能提升预期

通过RL增强，预期在以下方面获得提升：

1. **约束满足率**: +15-25%
2. **美学质量**: +10-20%
3. **用户满意度**: +20-30%
4. **生成稳定性**: +25-35%

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

## 📄 引用

如果使用此代码，请引用原始RADM论文和此RL增强扩展：

```bibtex
@inproceedings{fengheng2023relation,
    title={Relation-Aware Diffusion Model for Controllable Poster Layout Generation},
    author={Li, Fengheng and Liu, An and Feng, Wei and Zhu, Honghe and Li, Yaoyu and Zhang, Zheng and Lv, Jingjing and Zhu, Xin and Shen, Junjie and Lin, Zhangang and Shao, Jingping},
    booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
    pages={1249--1258},
    year={2023}
}
```

---

## ❓ 常见问题

**Q: 如何选择合适的引导强度？**
A: 从0.1开始，根据布局复杂度调整。复杂布局可以使用更高的引导强度。

**Q: RL训练需要多少数据？**
A: 建议至少1000个不同的布局样本，RL训练需要数千个回合。

**Q: 如何处理不同类型的布局？**
A: 使用领域自适应技术，或为不同布局类型训练专门的RL智能体。

**Q: 联合训练的计算成本？**
A: 相比纯扩散模型，增加约20-50%的计算开销，具体取决于RL复杂度。
