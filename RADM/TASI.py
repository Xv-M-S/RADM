import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --------------------------------------------------------
# 辅助函数：正弦位置编码 (Sinusoidal Time Embedding)
# RADM 原代码里应该已经有了类似实现，如果有可以直接复用
# --------------------------------------------------------
def get_timestep_embedding(timesteps, embedding_dim):
    """
    输入: timesteps (Tensor [Batch]), embedding_dim (int)
    输出: (Tensor [Batch, embedding_dim])
    """
    assert len(timesteps.shape) == 1
    
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb

# --------------------------------------------------------
# 核心模块：TASI (Time-Adaptive Semantic Injection)
# --------------------------------------------------------
class TimeAdaptiveVTRAM(nn.Module):
    def __init__(self, 
                 visual_dim=256,   # 视觉特征维度 (FPN输出通道数)
                 text_dim=768,     # 文本特征维度 (RoBERTa embedding维度)
                 time_emb_dim=256, # 时间嵌入维度
                 num_heads=8,      # 注意力头数
                 dropout=0.1):
        super().__init__()
        
        self.visual_dim = visual_dim
        
        # 1. 基础 Cross-Attention 层
        # Visual = Query, Text = Key/Value
        # 注意: nn.MultiheadAttention 默认输入是 (Seq_Len, Batch, Dim)
        # 如果你的输入是 (Batch, Seq_Len, Dim)，需要设置 batch_first=True
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=visual_dim, 
            num_heads=num_heads, 
            kdim=text_dim, 
            vdim=text_dim, 
            dropout=dropout,
            batch_first=True 
        )
        
        # 归一化层 (Pre-Norm 或 Post-Norm 均可，这里用 Post-Norm)
        self.norm1 = nn.LayerNorm(visual_dim)
        self.norm2 = nn.LayerNorm(visual_dim)
        
        # 前馈网络 (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(visual_dim, visual_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(visual_dim * 4, visual_dim),
            nn.Dropout(dropout)
        )

        # 2. [核心创新] 时间感知门控网络 (Time-Gating MLP)
        # 这个网络负责根据 t 生成 (scale, shift)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(), # SiLU (Swish) 在 Diffusion 模型中效果很好
            nn.Linear(time_emb_dim, visual_dim * 2) # 输出 2 倍维度: 对应 scale 和 shift
        )
        
        # 初始化 Time MLP 的最后一层为 0
        # 这样初始状态下 scale=0, shift=0，不影响原有特征流
        # 有助于模型训练初期的稳定性
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)

    def forward(self, visual_feats, text_feats, timesteps):
        """
        Args:
            visual_feats: [Batch, Num_Pixels, Visual_Dim] (图像特征)
            text_feats:   [Batch, Seq_Len, Text_Dim]      (文本特征)
            timesteps:    [Batch]                         (当前时间步 t)
        """
        # --- Step 1: 准备时间嵌入 ---
        # 如果传入的已经是 embedding 就不需要这一步
        if len(timesteps.shape) == 1:
            time_emb = get_timestep_embedding(timesteps, self.visual_dim) # 这里假设 time_dim = visual_dim
        else:
            time_emb = timesteps
            
        # --- Step 2: 计算时间调制参数 (scale, shift) ---
        # style: [Batch, Visual_Dim * 2]
        style = self.time_mlp(time_emb)
        
        # 将其拆分为 scale 和 shift
        # scale, shift: [Batch, Visual_Dim]
        scale, shift = style.chunk(2, dim=1)
        
        # 维度对齐以便广播: [Batch, 1, Visual_Dim]
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        
        # --- Step 3: 执行 Cross-Attention ---
        # 残差连接 1
        residual = visual_feats
        
        # Attention 计算
        # Query=Visual, Key=Text, Value=Text
        attn_out, _ = self.cross_attn(query=visual_feats, 
                                      key=text_feats, 
                                      value=text_feats)
        
        # --- Step 4: [关键] 应用时间自适应调制 (FiLM) ---
        # 逻辑: modulated = attn_out * (1 + scale) + shift
        # 这意味着模型可以根据时间 t，增强或抑制 Attention 的结果
        modulated_attn = attn_out * (1 + scale) + shift
        
        # 残差连接 + 归一化
        x = self.norm1(residual + modulated_attn)
        
        # --- Step 5: FFN ---
        # 残差连接 2
        x = self.norm2(x + self.ffn(x))
        
        return x