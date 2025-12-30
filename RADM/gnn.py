import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseEdgeEvolvingLayer(nn.Module):
    """
    单层 CA-EEGN：
    1. 根据两端节点特征 + 旧边特征 -> 更新边特征
    2. 根据新边特征引导 Attention -> 更新节点特征
    """
    def __init__(self, node_dim, edge_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        
        # --- A. Edge Update Components ---
        # Input: [Node_i || Node_j || Edge_ij] -> Output: New_Edge_ij
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, edge_dim * 2),
            nn.LayerNorm(edge_dim * 2),
            nn.GELU(),
            nn.Linear(edge_dim * 2, edge_dim),
            nn.Dropout(dropout)
        )
        self.edge_norm = nn.LayerNorm(edge_dim)

        # --- B. Node Update Components (Attention) ---
        self.query = nn.Linear(node_dim, node_dim, bias=False)
        self.key = nn.Linear(node_dim, node_dim, bias=False)
        self.value = nn.Linear(node_dim, node_dim, bias=False)
        
        # 将边特征投影到 Attention Head 的维度，作为 Bias
        self.edge_to_attn_bias = nn.Linear(edge_dim, num_heads, bias=False)

        self.out_proj = nn.Linear(node_dim, node_dim)
        self.node_norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_feat, mask=None):
        """
        Args:
            h: [B, N, D_n] - 节点特征
            edge_feat: [B, N, N, D_e] - 边特征
            mask: [B, N] - Padding Mask (True表示由内容, False表示Padding)
        Returns:
            new_h, new_edge_feat
        """
        B, N, _ = h.shape
        
        # ---------------------------------------------------------
        # Step 1: 显式边更新 (Explicit Edge Update)
        # ---------------------------------------------------------
        # 构建节点对上下文: Pair(i, j) = [h_i || h_j]
        # h_i_expand: [B, N, 1, D] -> [B, N, N, D] (Rows)
        # h_j_expand: [B, 1, N, D] -> [B, N, N, D] (Cols)
        h_i_expand = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j_expand = h.unsqueeze(1).expand(-1, N, -1, -1)
        
        # input: [B, N, N, 2*D_n + D_e]
        edge_input = torch.cat([h_i_expand, h_j_expand, edge_feat], dim=-1)
        
        # 更新边特征 (Residual + Norm)
        edge_update = self.edge_mlp(edge_input)
        new_edge_feat = self.edge_norm(edge_feat + edge_update) # 残差连接

        # ---------------------------------------------------------
        # Step 2: 关系引导的节点更新 (Relation-Guided Node Update)
        # ---------------------------------------------------------
        # Q, K, V: [B, N, D] -> [B, N, Heads, D/Heads]
        head_dim = self.node_dim // self.num_heads
        
        q = self.query(h).view(B, N, self.num_heads, head_dim).transpose(1, 2) # [B, H, N, d]
        k = self.key(h).view(B, N, self.num_heads, head_dim).transpose(1, 2)
        v = self.value(h).view(B, N, self.num_heads, head_dim).transpose(1, 2)
        
        # 原始 Attention Score: (Q @ K^T) / sqrt(d) -> [B, H, N, N]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        
        # 关键点：将更新后的边特征注入 Attention
        # new_edge_feat: [B, N, N, D_e] -> [B, N, N, Heads] -> [B, Heads, N, N]
        edge_bias = self.edge_to_attn_bias(new_edge_feat).permute(0, 3, 1, 2)
        
        # 融合几何/逻辑 Bias
        attn_scores = attn_scores + edge_bias

        # 处理 Mask (屏蔽 Padding 节点)
        if mask is not None:
            # mask: [B, N] -> [B, 1, 1, N]
            attn_mask = mask.view(B, 1, 1, N).expand(-1, self.num_heads, N, -1)
            # 将 mask 为 0 (False) 的位置设为负无穷
            attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))

        # Softmax & Aggregation
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # [B, H, N, N] @ [B, H, N, d] -> [B, H, N, d]
        context = torch.matmul(attn_probs, v)
        
        # 还原形状 -> [B, N, D]
        context = context.transpose(1, 2).contiguous().view(B, N, self.node_dim)
        
        # Output Projection + Residual + Norm
        new_h = self.node_norm(h + self.dropout(self.out_proj(context)))
        
        return new_h, new_edge_feat


class ContextAwareGNN(nn.Module):
    """
    CA-EEGN 主模型
    """
    def __init__(self, num_node_types, num_edge_types, hidden_dim=256, num_layers=3, heads=4):
        super().__init__()
        
        # Embeddings
        self.node_emb = nn.Embedding(num_node_types, hidden_dim)
        
        # 边特征Embedding：这里假设输入是边的类别索引
        # 如果某些位置没有边，建议用一种特殊的类别索引（如0）表示 "No Edge"
        self.edge_emb = nn.Embedding(num_edge_types, hidden_dim)
        
        self.layers = nn.ModuleList([
            DenseEdgeEvolvingLayer(hidden_dim, hidden_dim, num_heads=heads)
            for _ in range(num_layers)
        ])
        
    def forward(self, node_inputs, edge_inputs, mask=None):
        """
        Args:
            node_inputs: [B, N] (LongTensor) - 节点类别ID
            edge_inputs: [B, N, N] (LongTensor) - 边类别ID
            mask: [B, N] (BoolTensor) - 有效节点Mask
        Returns:
            final_h: [B, N, Hidden_Dim]
            final_e: [B, N, N, Hidden_Dim] -> 这里的 E 就是要喂给 GTFM 的
        """
        # 1. Init Features
        h = self.node_emb(node_inputs)        # [B, N, D]
        e = self.edge_emb(edge_inputs)        # [B, N, N, D]
        
        # 2. Iterate Layers
        for layer in self.layers:
            h, e = layer(h, e, mask)
            
        return h, e
    





def test_ca_eegn():
    print("=== Testing Dense Context-Aware Edge-Evolving GNN ===")
    
    # 1. 超参数设置
    B = 4          # Batch Size
    N = 10         # Number of Nodes (e.g., max layout elements)
    Node_Types = 20 # 比如 Logo, Text, Button...
    Edge_Types = 5  # 比如 Left, Top, Inside, No-Relation...
    Hidden_Dim = 64
    
    # 2. 模拟输入数据
    # 随机生成节点类别
    node_inputs = torch.randint(0, Node_Types, (B, N))
    
    # 随机生成边类别 (B, N, N)
    edge_inputs = torch.randint(0, Edge_Types, (B, N, N))
    
    # 生成 Mask (假设每个样本的最后几个节点是 Padding)
    mask = torch.ones((B, N), dtype=torch.bool)
    mask[:, -2:] = False # 每个样本最后2个节点无效
    
    # 3. 实例化模型
    model = ContextAwareGNN(
        num_node_types=Node_Types,
        num_edge_types=Edge_Types,
        hidden_dim=Hidden_Dim,
        num_layers=2, # 堆叠2层
        heads=4
    )
    
    print(f"Model created. Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 4. 前向传播
    h_out, e_out = model(node_inputs, edge_inputs, mask)
    
    # 5. 验证输出形状
    print("\n--- Output Shapes ---")
    print(f"Node Features: {h_out.shape} (Expected: [{B}, {N}, {Hidden_Dim}])")
    print(f"Edge Features: {e_out.shape} (Expected: [{B}, {N}, {N}, {Hidden_Dim}])")
    
    assert h_out.shape == (B, N, Hidden_Dim)
    assert e_out.shape == (B, N, N, Hidden_Dim)
    
    # 6. 验证反向传播 (确保梯度流是通的)
    print("\n--- Backward Pass Check ---")
    try:
        dummy_loss = h_out.sum() + e_out.sum()
        dummy_loss.backward()
        print("Backward pass successful! Gradients computed.")
    except Exception as e:
        print(f"Backward pass failed: {e}")

    # 7. 模拟接入 GTFM 的数据流
    print("\n--- Integration Check (GTFM Input) ---")
    print("The 'e_out' tensor is ready to be passed to GTFM as 'gnn_edge_feats'.")
    print(f"GTFM Input Shape: {e_out.shape}")

if __name__ == "__main__":
    test_ca_eegn()