import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================
# Added StructureEvolvingSERM Class
# ========================================
class StructureEvolvingSERM(nn.Module):
    def __init__(self, d_model, nhead=8, k_neighbors=5, dropout=0.1):
        """
        Structure-Evolving Relation Module (SERM)
        """
        super().__init__()
        self.d_model = d_model
        self.k = k_neighbors
        self.nhead = nhead

        # 使用 MultiheadAttention，通过 mask 实现 Graph Attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # FFN
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    
    def build_dynamic_knn_mask(self, rois):
        B, N, _ = rois.shape
        centers = rois[:, :, :2]
        dist_matrix = torch.cdist(centers, centers, p=2)
        k_val = min(self.k + 1, N)
        _, knn_indices = dist_matrix.topk(k_val, dim=-1, largest=False)

        # 创建 float mask: -inf 表示屏蔽，0.0 表示保留
        mask = torch.full((B, N, N), float('-inf'), device=rois.device)
        
        batch_idx = torch.arange(B, device=rois.device).view(B, 1, 1).expand(B, N, k_val)
        row_idx = torch.arange(N, device=rois.device).view(1, N, 1).expand(B, N, k_val)
        mask[batch_idx, row_idx, knn_indices] = 0.0

        # Visual Hierarchy
        mask[:, :, 0] = 0.0
        mask[:, 0, :] = 0.0

        # 扩展到 head 维度
        mask = mask.repeat_interleave(self.nhead, dim=0)  # (B * nhead, N, N)

        return mask  # 返回 float tensor，不是 bool！


    def forward(self, rois, features):
        # features: [B, L, E] --> 转为 [L, B, E]
        features_T = features.transpose(0, 1)  # (L, B, E)

        # 构建 mask —— 注意：现在我们需要 (B * nhead, L, L)
        attn_mask = self.build_dynamic_knn_mask(rois)  # 保持返回 (B * nhead, L, L)

        # 调用 attention
        src2_T, attn_weights = self.self_attn(
            features_T, features_T, features_T,
            attn_mask=attn_mask,
            need_weights=True
        )

        # 转回 [B, L, E]
        src2 = src2_T.transpose(0, 1)

        # 后续计算用 [B, L, E]
        src = features + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, attn_weights
    




class DualStreamSERM(nn.Module):
    def __init__(self, d_model, nhead=8, k_neighbors=5, dropout=0.1, alpha=0.7):
        """
        Structure-Evolving Relation Module (SERM) with Dual-Stream Interaction
        Args:
            d_model: 特征维度
            nhead: 注意力头数
            k_neighbors: K 近邻数量
            alpha: 几何距离的权重 (0.0 - 1.0)。 alpha 越大，越看重物理距离；alpha 越小，越看重语义相似度。
        """
        super().__init__()
        self.d_model = d_model
        self.k = k_neighbors
        self.nhead = nhead
        self.alpha = alpha  # Hyperparameter to balance Geometry and Semantics

        # Multihead Attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # FFN
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def compute_dual_stream_distance(self, rois, features):
        """
        计算几何与语义的融合距离矩阵
        Args:
            rois: [B, N, 4] -> (cx, cy, w, h)
            features: [B, N, C] -> 语义特征
        Returns:
            fused_dist: [B, N, N] -> 融合后的距离矩阵，值越小代表越相关
        """
        # --- Stream 1: Geometric Distance ---
        centers = rois[:, :, :2]
        # 计算欧氏距离 [B, N, N]
        geo_dist = torch.cdist(centers, centers, p=2)
        # 归一化几何距离到 0-1 之间 (为了能和语义距离相加)
        # 简单的 Min-Max 归一化 (加上 1e-6 防止除零)
        geo_min = geo_dist.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        geo_max = geo_dist.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        geo_dist_norm = (geo_dist - geo_min) / (geo_max - geo_min + 1e-6)

        # --- Stream 2: Semantic Similarity ---
        # 归一化特征向量，方便计算 Cosine Distance
        feats_norm = F.normalize(features, p=2, dim=-1)
        # 计算余弦相似度矩阵 [B, N, N] (范围 -1 到 1)
        # bmm: (B, N, C) x (B, C, N) -> (B, N, N)
        cos_sim = torch.bmm(feats_norm, feats_norm.transpose(1, 2))
        
        # 将相似度转换为“距离” (1 - sim)
        # Cosine Distance范围变为 0 (最相似) 到 2 (最不相似)
        sem_dist = 1.0 - cos_sim
        # 归一化语义距离到 0-1
        sem_dist_norm = sem_dist / 2.0 

        # --- Gated Fusion ---
        # 融合公式：Weighted Sum
        # 背景抑制原理：背景框的特征杂乱，与前景框的 sem_dist 会很大 (接近 1)。
        # 导致 fused_dist 变大，从而不会被选入 Top-K。
        fused_dist = self.alpha * geo_dist_norm + (1 - self.alpha) * sem_dist_norm
        
        return fused_dist

    def build_dual_stream_mask(self, rois, features):
        """
        基于融合距离构建 Mask
        """
        B, N, _ = rois.shape
        
        # 1. 计算双流融合距离
        # 如果是 Inference 阶段且 N 很大 (Flatten过)，可能需要先 view 回去，
        # 但这里的 features 已经是 [B, N, C] 了 (因为 head.py 里处理过)，所以直接用。
        dist_matrix = self.compute_dual_stream_distance(rois, features)
        
        # 2. 找到最近的 K 个邻居 (基于融合距离)
        k_val = min(self.k + 1, N)
        _, knn_indices = dist_matrix.topk(k_val, dim=-1, largest=False)
        
        # 3. 构建 Float Mask (-inf / 0.0)
        mask = torch.full((B, N, N), float('-inf'), device=rois.device)
        
        batch_indices = torch.arange(B, device=rois.device).view(B, 1, 1).expand(B, N, k_val)
        row_indices = torch.arange(N, device=rois.device).view(1, N, 1).expand(B, N, k_val)
        
        mask[batch_indices, row_indices, knn_indices] = 0.0
        
        # [Global Connection]
        mask[:, :, 0] = 0.0
        mask[:, 0, :] = 0.0
        
        # 4. 扩展维度
        mask = mask.repeat_interleave(self.nhead, dim=0)
        
        return mask

    def forward(self, rois, features):
        # features: [B, N, C] → [N, B, C]
        features_T = features.transpose(0, 1)

        # 构建 mask: 仍返回 (B * nhead, N, N) —— 这是 batch_first=False 下的标准格式
        attn_mask = self.build_dual_stream_mask(rois, features)  # (B * nhead, N, N)

        # 调用 attention
        src2_T, attn_weights = self.self_attn(
            features_T, features_T, features_T,
            attn_mask=attn_mask,
            need_weights=True
        )

        # 转回 [B, N, C]
        src2 = src2_T.transpose(0, 1)

        # 后续残差连接...
        src = features + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, attn_weights
    





class DualGeometryRelationAwareModule(nn.Module):
    def __init__(self,
                 topo_in_dim,
                 topo_out_dim=256,
                 drop_rate=0.,
                 embd_dim=64,
                 wave_length=1000,
                 fc_out_channels=1,
                 sem_alpha=0.5): # [New] 新增超参数，控制语义门控的强度
        super(DualGeometryRelationAwareModule, self).__init__()
        # used in calculate "weight_geo"
        self.linear = nn.Linear(embd_dim, fc_out_channels)
        self.relu = nn.ReLU(inplace=True)

        # W^v
        self.linear2 = nn.Linear(topo_in_dim, topo_out_dim)

        self.out_dim = embd_dim
        self.wave_length = wave_length
        self.topo_out_dim = topo_out_dim
        self.drop_rate = drop_rate
        self.sem_alpha = sem_alpha # 保存 alpha

        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.linear.weight, 0, 0.01)
        nn.init.constant_(self.linear.bias, 0)

        nn.init.normal_(self.linear2.weight, 0, 0.01)
        nn.init.constant_(self.linear2.bias, 0)

    def build_relative_geo(self, rois, gt):
        assert rois.shape[1] == gt.shape[1]

        rois_repeat = rois[..., None].repeat(1, 1, gt.shape[0])  # broadcast
        gt_x = gt[:, 0]
        gt_y = gt[:, 1]
        gt_w = gt[:, 2]
        gt_h = gt[:, 3]
        gt_w = gt_w.maximum(torch.tensor(1e-3).to(gt.device))
        gt_h = gt_h.maximum(torch.tensor(1e-3).to(gt.device))

        # x
        rois_x = rois_repeat[:, 0, :]  # [512, gt.shape[0]]  ,broadcast
        rel_x = torch.abs(gt_x - rois_x) / gt_w
        rel_x = rel_x.maximum(torch.tensor(1e-3).to(rois.device))

        # y
        rois_y = rois_repeat[:, 1, :]
        rel_y = torch.abs(gt_y - rois_y) / gt_h
        rel_y = rel_y.maximum(torch.tensor(1e-3).to(rois.device))

        # w
        rois_w = rois_repeat[:, 2, :]
        rel_w = rois_w / gt_w
        rel_w = rel_w.maximum(torch.tensor(1e-3).to(rois.device))

        # h
        rois_h = rois_repeat[:, 3, :]
        rel_h = rois_h / gt_h
        rel_h = rel_h.maximum(torch.tensor(1e-3).to(rois.device))

        relative_geo = torch.stack([rel_x, rel_y, rel_w, rel_h], dim=-1).float()

        return torch.log(relative_geo)

    def extract_position_embedding(self, relative_geo, feat_dim=64, wave_length=1000):
        '''
        relative_geo: [num_rois, num_gt_rois, 4]
        '''
        feat_range = torch.arange(0, feat_dim / 8)
        dim_mat = torch.pow(torch.full((1,), wave_length), (8. / feat_dim) * feat_range).to(relative_geo.device)  # shape [1,8]
        dim_mat = dim_mat.reshape((1, 1, 1, -1))  # shape [1,1,1,8]

        relative_geo = torch.unsqueeze(100.0 * relative_geo, dim=-1)  # [num_rois, num_gt_rois, 4, 1]
        div_mat = relative_geo / dim_mat
        sin_mat = div_mat.sin()
        cos_mat = div_mat.cos()
        embedding = torch.stack([sin_mat, cos_mat], dim=-1)  # [num_rois, num_gt_rois, 4, feat_dim/4]
        embedding = embedding.flatten(2)  # [num_rois, num_gt_rois, feat_dim]

        return embedding
    
    def forward(self, rois, gt_rois, gt_bbox_feats, batch_num):
        """ 
        extract topology features for text tracking
        :param rois: shape (n, 5), [batch_ind, x1, y1, x2, y2]
        :param gt_rois: ground truth (m, 5)   
        :param gt_bbox_feats: [m, channel, width, height] 
        :param batch_num: batch size
        :return: topology_feats [n, self.out_dim]
        """
        
        n = rois.shape[1]
        rois = rois.reshape(-1,4) #[n,4]
        rois = torch.cat((torch.tensor([[i for j in range(n)]for i in range(batch_num)]).reshape(-1, 1).to(rois.device), rois), dim=1) #[n,5]
        gt_rois = rois # 这里逻辑看似 rois 和 gt_rois 是一样的 (self-attention)
        
        rois_xywh = torch.stack((rois[:, 0], (rois[:, 1] + rois[:, 3]) / 2, (rois[:, 2] + rois[:, 4]) / 2,
                                 rois[:, 3] - rois[:, 1], rois[:, 4] - rois[:, 2]), 1)
        
        gt_xywh = torch.stack((gt_rois[:, 0], (gt_rois[:, 1] + gt_rois[:, 3]) / 2, (gt_rois[:, 2] + gt_rois[:, 4]) / 2,
                               gt_rois[:, 3] - gt_rois[:, 1], gt_rois[:, 4] - gt_rois[:, 2]), 1)
        
        # [Semantic Stream Step 1] 投影特征
        gt_bbox_feats_trans = self.linear2(gt_bbox_feats.view(gt_bbox_feats.size(0), -1))  # [M, topo_out_dim]
        
        topology_feats = torch.zeros((rois.shape[0], self.topo_out_dim)).to(rois.device) 
        
        for i in range(batch_num):
            # 1. 准备数据
            rois_i = rois_xywh[rois[:, 0] == i, 1::]  # [n_i, 4]
            gt_i = gt_xywh[gt_rois[:, 0] == i, 1::]   # [m_i, 4]
            
            # [Semantic Stream Step 2] 提取当前 batch 的特征
            feats_i = gt_bbox_feats_trans[gt_rois[:, 0] == i] # [n_i, C]
            
            # 2. Geometry Stream Calculation
            relative_geo = self.build_relative_geo(rois_i, gt_i)
            geo_embedding = self.extract_position_embedding(relative_geo, self.out_dim, self.wave_length)
            
            # [Modified] 获取原始的几何 Logits (Unnormalized Score)
            # 原始代码: weight_geo = self.relu(self.linear(weight_geo)).squeeze(-1)
            # 我们保留这个作为 "Geometry Energy"
            geo_energy = self.relu(self.linear(geo_embedding)).squeeze(-1) # [n_i, m_i] (>=0 due to ReLU)

            # [Semantic Stream Step 3] 计算语义相似度矩阵
            # Normalize features for Cosine Similarity
            feats_norm = F.normalize(feats_i, p=2, dim=1)
            # Matmul: (n, C) x (C, n) -> (n, n)
            sim_matrix = torch.mm(feats_norm, feats_norm.t()) 
            
            # [Dual-Stream Interaction / Gating]
            # 逻辑：Semantics as a Gate.
            # 将相似度 (-1, 1) 映射到 (0, 1) 作为门控系数
            # sim_gate 越接近 1，表示语义越相关，保留几何连接；越接近 0，抑制几何连接
            sim_gate = torch.sigmoid(sim_matrix * 5) # *5 是为了增加梯度的陡峭度，让区分更明显
            
            # 融合公式：Fused Energy = Geometry Energy * Semantic Gate
            # 如果两个框是背景和前景，sim_gate 很小，geo_energy 被抑制
            fused_energy = geo_energy * (self.sem_alpha * sim_gate + (1 - self.sem_alpha))
            
            # 3. Attention Calculation
            weight_final = F.softmax(fused_energy, dim=-1)
            weight_final = F.dropout(weight_final, p=self.drop_rate, training=self.training)
        
            # 4. Aggregation
            topology_feats[rois[:, 0] == i] = torch.mm(weight_final, feats_i)

        return topology_feats