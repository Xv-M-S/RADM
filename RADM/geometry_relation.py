# ========================================
# Geometry Relation-Aware Spatial Context Modeling
#
# Enhanced geometry relation module with:
# - Relative geometric vector g_ij (scale-invariant)
# - Sin-Cos position encoding PE(g_ij)
# - Geometric attention α_ij via MLP + Softmax
# - Weighted feature aggregation h_i^geo = Σ_j α_ij * P(v_j)
# ========================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GeometryRelationModule(nn.Module):
    """
    Geometry Relation-Aware Module for spatial context modeling.

    For any two layout elements i and j with bounding boxes
    (x_i, y_i, w_i, h_i) and (x_j, y_j, w_j, h_j) in normalized coordinates,
    builds scale-invariant geometric relation vector and computes
    geometry-aware attention weights for feature aggregation.
    """

    def __init__(self, in_channels=256, embed_dim=64, fc_out_channels=1,
                 wave_length=1000, dropout=0.1, out_dim=256):
        """
        Args:
            in_channels: dimension of input RoI visual features v_j
            embed_dim: dimension of PE embedding for geometric features
            fc_out_channels: output channels of MLP for attention score
            wave_length: wavelength for sin-cos PE
            dropout: dropout rate
            out_dim: output feature dimension
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.wave_length = wave_length
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.dropout_rate = dropout

        # MLP for computing attention scores from geometric embeddings
        # α_ij = Softmax(MLP(R_ij^p))
        self.geo_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, fc_out_channels),
        )

        # Linear mapping for visual features: P(v_j)
        self.visual_proj = nn.Sequential(
            nn.Linear(in_channels, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

        # Output layer normalization
        self.output_norm = nn.LayerNorm(out_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def build_relative_geometry(self, boxes_i, boxes_j):
        """
        Build scale-invariant relative geometric relation vector.

        g_ij = [log(|x_i-x_j|/w_j), log(|y_i-y_j|/h_j), log(w_i/w_j), log(h_i/h_j)]

        Args:
            boxes_i: (N, 4) or (N, M, 4) center-format boxes (cx, cy, w, h)
            boxes_j: (M, 4) center-format boxes (cx, cy, w, h)

        Returns:
            g_ij: (N, M, 4) relative geometric vectors
        """
        if boxes_i.dim() == 2 and boxes_j.dim() == 2:
            # (N, 4) vs (M, 4) -> (N, M, 4)
            N, M = boxes_i.size(0), boxes_j.size(0)
            eps = 1e-6

            # Extract coordinates
            x_i, y_i, w_i, h_i = boxes_i[:, 0], boxes_i[:, 1], boxes_i[:, 2], boxes_i[:, 3]
            x_j, y_j, w_j, h_j = boxes_j[:, 0], boxes_j[:, 1], boxes_j[:, 2], boxes_j[:, 3]

            # Broadcast
            x_i = x_i.unsqueeze(1).expand(N, M)
            y_i = y_i.unsqueeze(1).expand(N, M)
            w_i = w_i.unsqueeze(1).expand(N, M)
            h_i = h_i.unsqueeze(1).expand(N, M)

            x_j = x_j.unsqueeze(0).expand(N, M)
            y_j = y_j.unsqueeze(0).expand(N, M)
            w_j = w_j.unsqueeze(0).expand(N, M)
            h_j = h_j.unsqueeze(0).expand(N, M)

            h_j = h_j.clamp(min=eps)

            # Compute relative features
            rel_dx = torch.log(torch.abs(x_i - x_j) / w_j + eps)
            rel_dy = torch.log(torch.abs(y_i - y_j) / h_j + eps)
            rel_dw = torch.log(w_i / w_j.clamp(min=eps) + eps)
            rel_dh = torch.log(h_i / h_j.clamp(min=eps) + eps)

            g_ij = torch.stack([rel_dx, rel_dy, rel_dw, rel_dh], dim=-1)
        else:
            raise ValueError(f"Unexpected box shapes: {boxes_i.shape}, {boxes_j.shape}")

        return g_ij

    def extract_position_embedding(self, g_ij):
        """
        Map low-dimensional geometric vector to high-dimensional embedding
        using sinusoidal-cosine position encoding.

        R_ij^p = PE(g_ij) where PE uses sine and cosine at different frequencies.

        Args:
            g_ij: (N, M, 4) relative geometric vectors

        Returns:
            R_ij^p: (N, M, embed_dim) PE embeddings
        """
        N, M, _ = g_ij.shape
        device = g_ij.device

        # feat_dim per coordinate
        feat_dim_per_coord = self.embed_dim // 4

        feat_range = torch.arange(0, feat_dim_per_coord // 2, device=device).float()
        dim_mat = torch.pow(
            torch.full((1,), self.wave_length, device=device),
            (4. / feat_dim_per_coord) * feat_range
        )  # (feat_dim_per_coord // 2,)
        dim_mat = dim_mat.view(1, 1, 1, 1, -1)  # (1, 1, 1, 1, K)

        # Scale geometric features and compute sin/cos
        g_ij_expanded = g_ij.unsqueeze(-1)  # (N, M, 4, 1)
        g_ij_scaled = 100.0 * g_ij_expanded  # scaling factor

        div_mat = g_ij_scaled / dim_mat  # (N, M, 4, K)
        sin_mat = div_mat.sin()  # (N, M, 4, K)
        cos_mat = div_mat.cos()  # (N, M, 4, K)

        embedding = torch.stack([sin_mat, cos_mat], dim=-1)  # (N, M, 4, K, 2)
        embedding = embedding.flatten(2)  # (N, M, 4 * K * 2)

        # Pad or trim to match embed_dim
        if embedding.size(-1) < self.embed_dim:
            pad = torch.zeros(N, M, self.embed_dim - embedding.size(-1), device=device)
            embedding = torch.cat([embedding, pad], dim=-1)
        elif embedding.size(-1) > self.embed_dim:
            embedding = embedding[..., :self.embed_dim]

        return embedding

    def forward(self, roi_features, boxes, return_attention=False):
        """
        Compute geometry-enhanced features via geometric attention.

        Args:
            roi_features: (N, in_channels) RoI visual features v_i for each element
            boxes: (N, 4) bounding boxes in center format (cx, cy, w, h), normalized
            return_attention: whether to return attention weights

        Returns:
            h_geo: (N, out_dim) geometry-enhanced features
            attn_weights: (N, N) attention weights (if return_attention=True)
        """
        N = boxes.size(0)
        device = boxes.device

        # Step 1: Build relative geometric vectors g_ij
        g_ij = self.build_relative_geometry(boxes, boxes)  # (N, N, 4)

        # Step 2: PE encoding R_ij^p
        R_ij_p = self.extract_position_embedding(g_ij)  # (N, N, embed_dim)

        # Step 3: Compute geometric attention weights
        # α_ij = Softmax(MLP(R_ij^p))
        geo_scores = self.geo_mlp(R_ij_p).squeeze(-1)  # (N, N)
        # Mask self-attention
        self_mask = torch.eye(N, device=device).bool()
        geo_scores = geo_scores.masked_fill(self_mask, float('-inf'))
        attn_weights = F.softmax(geo_scores, dim=-1)  # (N, N)
        attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)

        # Step 4: Weighted aggregation
        # P(v_j): project visual features
        v_proj = self.visual_proj(roi_features)  # (N, out_dim)

        # h_i^geo = Σ_j α_ij * P(v_j)
        h_geo = attn_weights @ v_proj  # (N, out_dim)
        h_geo = self.output_norm(h_geo + v_proj)  # residual + norm

        if return_attention:
            return h_geo, attn_weights

        return h_geo


class GeometryRelationModuleBatch(nn.Module):
    """
    Batch version of GeometryRelationModule that handles
    variable-sized elements per batch item.
    """

    def __init__(self, in_channels=256, embed_dim=64, fc_out_channels=1,
                 wave_length=1000, dropout=0.1, out_dim=256):
        super().__init__()
        self.core = GeometryRelationModule(
            in_channels=in_channels,
            embed_dim=embed_dim,
            fc_out_channels=fc_out_channels,
            wave_length=wave_length,
            dropout=dropout,
            out_dim=out_dim,
        )

    def forward(self, roi_features_list, boxes_list):
        """
        Args:
            roi_features_list: list of (N_i, in_channels) tensors
            boxes_list: list of (N_i, 4) tensors in center format

        Returns:
            h_geo_list: list of (N_i, out_dim) tensors
        """
        outputs = []
        for roi_feat, boxes in zip(roi_features_list, boxes_list):
            if roi_feat is None or roi_feat.numel() == 0:
                outputs.append(None)
            else:
                h_geo = self.core(roi_feat, boxes)
                outputs.append(h_geo)

        return outputs
