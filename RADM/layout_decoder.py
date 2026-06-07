# ========================================
# Layout Encoder & Multi-Modal Decoder
#
# Layout Encoder:
#   - CLIP text encoder + Fourier feature encoding + MLP fusion
#   - Output: layout token sequence H_l = [h_1^l, ..., h_N^l]
#
# Multi-Modal Decoder (MM-DiT style):
#   - Joint attention for bidirectional interaction
#   - Three branches: visual, topology, geometry
#   - Fusion: H_final = Concat(H_l^vis, H_l^topo, H_l^geo) * W_fuse
#   - MLP prediction head for bbox coordinates
# ========================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class FourierFeatureEncoding(nn.Module):
    """
    Fourier feature mapping for bounding box coordinates.

    Fourier(b_i) = [sin(2π * B * b_i), cos(2π * B * b_i)]

    where B is a fixed Gaussian random matrix.
    This mapping enhances the model's expressiveness for
    fine-grained position offsets and scale variations.

    Reference: GLIGEN (li2023gligen)
    """

    def __init__(self, coord_dim=4, embed_dim=128, scale=10.0):
        super().__init__()
        self.coord_dim = coord_dim
        self.embed_dim = embed_dim
        self.scale = scale

        # Fixed Gaussian random matrix B
        B = torch.randn(coord_dim, embed_dim // 2) * scale
        self.register_buffer('B', B)

    def forward(self, coords):
        """
        Args:
            coords: (..., coord_dim) bounding box coordinates (cx, cy, w, h)

        Returns:
            (..., embed_dim * 2) Fourier features
        """
        x_proj = 2 * math.pi * coords @ self.B  # (..., embed_dim//2)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class LayoutEncoder(nn.Module):
    """
    Layout Token encoder.

    For each foreground element i:
      - Text embedding: t_i = τ(c_i) via CLIP text encoder
      - Geometry encoding: Fourier(b_i)
      - Layout token: h_i^l = MLP([t_i || Fourier(b_i)])

    Output: H_l = [h_1^l, ..., h_N^l] ∈ R^(N×D)
    """

    def __init__(self, text_dim=512, coord_dim=4, hidden_dim=256,
                 fourier_dim=128, fourier_scale=10.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_dim = text_dim
        self.coord_dim = coord_dim

        # Fourier feature encoding for coordinates
        self.fourier_enc = FourierFeatureEncoding(
            coord_dim=coord_dim,
            embed_dim=fourier_dim,
            scale=fourier_scale,
        )
        fourier_out_dim = fourier_dim  # sin + cos

        # Text feature projection
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Geometry projection
        self.geo_proj = nn.Sequential(
            nn.Linear(fourier_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Fusion MLP: [t_i || Fourier(b_i)] -> h_i^l
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, text_embeddings, boxes):
        """
        Args:
            text_embeddings: (N, text_dim) or (B, N, text_dim)
            boxes: (N, coord_dim) or (B, N, coord_dim) normalized (cx, cy, w, h)

        Returns:
            H_l: (N, hidden_dim) or (B, N, hidden_dim) layout tokens
        """
        batch_mode = (text_embeddings.dim() == 3)

        # Text encoding
        t = self.text_proj(text_embeddings)

        # Fourier geometry encoding
        fourier_feat = self.fourier_enc(boxes)  # (..., fourier_dim)
        g = self.geo_proj(fourier_feat)

        # Concatenate and fuse
        fused = torch.cat([t, g], dim=-1)
        H_l = self.fusion_mlp(fused)

        return H_l


class JointCrossAttention(nn.Module):
    """
    MM-DiT style joint attention mechanism.

    Given layout tokens H_l and condition features H_c:
    [H_l', H_c'] = Attention([Q_l, Q_c], [K_l, K_c], [V_l, V_c])

    This allows bidirectional information exchange between layout tokens
    and condition tokens in a unified feature space.
    """

    def __init__(self, d_model=256, nhead=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # Projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, H_l, H_c):
        """
        Args:
            H_l: (B, N_l, d_model) layout tokens
            H_c: (B, N_c, d_model) condition tokens

        Returns:
            H_l': (B, N_l, d_model) enhanced layout tokens
            H_c': (B, N_c, d_model) enhanced condition tokens
        """
        B, N_l, D = H_l.shape
        N_c = H_c.size(1)

        # Concatenate along sequence dimension
        H_cat = torch.cat([H_l, H_c], dim=1)  # (B, N_l + N_c, D)

        # Self/cross-attention via multi-head attention
        attn_out, _ = self.attention(H_cat, H_cat, H_cat)

        # Split back
        H_l_out = attn_out[:, :N_l, :]
        H_c_out = attn_out[:, N_l:, :]

        # Residual + norm
        H_l_out = H_l + self.dropout(self.out_proj(H_l_out))
        H_c_out = H_c + self.dropout(self.out_proj(H_c_out))

        return H_l_out, H_c_out


class MultiModalInteractionBranch(nn.Module):
    """
    Single multi-modal interaction branch.

    Processes one condition modality (visual / topology / geometry)
    through joint cross-attention with layout tokens.
    """

    def __init__(self, d_model=256, nhead=8, dropout=0.1):
        super().__init__()
        self.cross_attn = JointCrossAttention(d_model, nhead, dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

        # Condition feature projection
        self.cond_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, H_l, H_cond):
        """
        Args:
            H_l: (B, N_l, d_model) layout tokens
            H_cond: (B, N_c, d_model) condition features

        Returns:
            H_l': (B, N_l, d_model) layout tokens enhanced by this condition
        """
        # Project condition features
        H_cond = self.cond_proj(H_cond)

        # Joint cross-attention
        H_l_out, H_cond_out = self.cross_attn(H_l, H_cond)

        # FFN on layout tokens
        H_l_out = self.ffn_norm(H_l_out + self.ffn(H_l_out))

        return H_l_out


class MultiModalDecoder(nn.Module):
    """
    Multi-modal layout decoder.

    Processes layout tokens with three parallel condition branches:
      1. Visual feature interaction: F_vis (background texture)
      2. Topology relation interaction: H_topo (RGCN embeddings)
      3. Geometry relation interaction: H_geo (geometric attention features)

    Final fusion:
      H_final = Concat(H_l^vis, H_l^topo, H_l^geo) * W_fuse

    Then MLP prediction head maps H_final to bbox coordinates.
    """

    def __init__(self, d_model=256, nhead=8, dropout=0.1,
                 fusion_mode="concat", num_pred_layers=3):
        super().__init__()
        self.d_model = d_model
        self.fusion_mode = fusion_mode

        # Three modality interaction branches
        self.visual_branch = MultiModalInteractionBranch(d_model, nhead, dropout)
        self.topo_branch = MultiModalInteractionBranch(d_model, nhead, dropout)
        self.geo_branch = MultiModalInteractionBranch(d_model, nhead, dropout)

        # Fusion
        if fusion_mode == "concat":
            fusion_in_dim = d_model * 3
        else:
            fusion_in_dim = d_model

        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_in_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        # MLP prediction head: layout tokens -> bbox coordinates
        pred_layers = []
        for i in range(num_pred_layers):
            in_dim = d_model if i == 0 else d_model
            pred_layers.extend([
                nn.Linear(in_dim, d_model),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
        pred_layers.append(nn.Linear(d_model, 4))  # (cx, cy, w, h)
        self.pred_head = nn.Sequential(*pred_layers)

    def forward(self, H_l, H_topo, H_geo, F_vis_spatial=None):
        """
        Args:
            H_l: (B, N_l, d_model) layout tokens
            H_topo: (B, N_topo, d_model) topology features from RGCN
            H_geo: (B, N_geo, d_model) geometry-enhanced features
            F_vis_spatial: (B, N_v, d_model) visual spatial features (optional)

        Returns:
            bbox_pred: (B, N_l, 4) predicted bounding box coordinates
            H_final: (B, N_l, d_model) final fused features
        """
        # Ensure batch dimension
        if H_l.dim() == 2:
            H_l = H_l.unsqueeze(0)
        if H_topo.dim() == 2:
            H_topo = H_topo.unsqueeze(0)
        if H_geo.dim() == 2:
            H_geo = H_geo.unsqueeze(0)

        # Three parallel interaction branches
        H_l_vis = self.visual_branch(
            H_l, H_topo
        ) if F_vis_spatial is None else self.visual_branch(H_l, F_vis_spatial)

        H_l_topo = self.topo_branch(H_l, H_topo)
        H_l_geo = self.geo_branch(H_l, H_geo)

        # Fusion
        if self.fusion_mode == "concat":
            H_final = torch.cat([H_l_vis, H_l_topo, H_l_geo], dim=-1)
        else:
            H_final = (H_l_vis + H_l_topo + H_l_geo) / 3.0

        H_final = self.fusion_layer(H_final)

        # Predict bounding boxes
        bbox_pred = self.pred_head(H_final)

        # Apply sigmoid to keep coordinates in [0, 1] for normalized coords
        bbox_pred = torch.sigmoid(bbox_pred)

        return bbox_pred, H_final


class LayoutGenerationHead(nn.Module):
    """
    Complete layout generation head combining:
    - Layout encoder
    - Multi-modal decoder
    - Bbox prediction
    """

    def __init__(self, d_model=256, nhead=8, dropout=0.1,
                 text_dim=512, fourier_dim=128, fourier_scale=10.0,
                 num_pred_layers=3, fusion_mode="concat"):
        super().__init__()
        self.d_model = d_model

        # Layout encoder
        self.layout_encoder = LayoutEncoder(
            text_dim=text_dim,
            coord_dim=4,
            hidden_dim=d_model,
            fourier_dim=fourier_dim,
            fourier_scale=fourier_scale,
        )

        # Multi-modal decoder
        self.decoder = MultiModalDecoder(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            fusion_mode=fusion_mode,
            num_pred_layers=num_pred_layers,
        )

    def forward(self, text_embeddings, boxes, H_topo, H_geo, F_vis=None):
        """
        Args:
            text_embeddings: (N, text_dim) or (B, N, text_dim)
            boxes: (N, 4) or (B, N, 4) current/noisy bbox coordinates
            H_topo: topology features from RGCN
            H_geo: geometry-enhanced features
            F_vis: visual features (optional)

        Returns:
            bbox_pred: predicted bbox coordinates
            H_final: final fused layout features
        """
        # Encode layout tokens
        H_l = self.layout_encoder(text_embeddings, boxes)

        # Multi-modal decode
        bbox_pred, H_final = self.decoder(H_l, H_topo, H_geo, F_vis)

        return bbox_pred, H_final
