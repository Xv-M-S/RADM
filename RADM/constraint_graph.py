# ========================================
# Constraint Graph Builder
# Builds the multi-modal constraint graph G = (V, E)
# where V = V_bg ∪ V_fg, E = R_BB ∪ R_BF ∪ R_FF
# ========================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Optional import - only needed for box format conversion
try:
    from .util.box_ops import box_cxcywh_to_xyxy
except ImportError:
    box_cxcywh_to_xyxy = None


class PositionalEncoding1D(nn.Module):
    """Sinusoidal position encoding for bounding box coordinates."""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, positions):
        """
        Args:
            positions: (N, 4) normalized bounding box (cx, cy, w, h)

        Returns:
            (N, d_model) position embeddings
        """
        N = positions.size(0)
        device = positions.device

        # For each of 4 coordinates, generate d_model/4 dims
        dim_per_coord = self.d_model // 4

        pe = []
        for i in range(4):
            # Encode each coordinate dimension
            x = positions[:, i:i + 1]  # (N, 1)
            div_term = torch.exp(
                torch.arange(0, dim_per_coord, 2, device=device).float() *
                (-math.log(10000.0) / dim_per_coord)
            )
            pe_sin = torch.sin(x * div_term)
            pe_cos = torch.cos(x * div_term)
            pe_i = torch.zeros(N, dim_per_coord, device=device)
            pe_i[:, 0::2] = pe_sin[:, :dim_per_coord // 2 + dim_per_coord % 2]
            pe_i[:, 1::2] = pe_cos[:, :dim_per_coord // 2]
            pe.append(pe_i)

        return torch.cat(pe, dim=-1)


class ConstraintGraphBuilder(nn.Module):
    """
    Builds the multi-modal constraint graph from structured design elements.

    Graph structure:
      - Nodes V = V_bg ∪ V_fg (background + foreground component nodes)
      - Edges E = R_BB ∪ R_BF ∪ R_FF (3 relation types)
      - Node features h_i^(0) = [e_i^text || e_i^pos || e_i^cls]
      - Edge features e_ij^rel from relative position labels

    Relation types:
      0: R_BB (background-to-background)
      1: R_BF (background-to-foreground)
      2: R_FF (foreground-to-foreground)
    """

    def __init__(self, text_dim=768, hidden_dim=256, num_classes=4,
                 num_relations=3, num_rel_labels=8):
        """
        Args:
            text_dim: dimension of text embeddings
            hidden_dim: output hidden dimension for node features
            num_classes: number of element classes
            num_relations: number of relation types (3: BB, BF, FF)
            num_rel_labels: number of discrete relative position labels
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.num_relations = num_relations
        self.num_rel_labels = num_rel_labels

        # Position encoder for absolute position embedding
        self.pos_encoder = PositionalEncoding1D(hidden_dim)

        # Class embedding (learnable)
        self.class_embed = nn.Embedding(num_classes + 1, hidden_dim)  # +1 for background

        # Text feature projection
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Position feature projection
        self.pos_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Class feature projection
        self.cls_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Relation encoder: maps discrete relative position label r_ij to continuous embedding
        self.relation_embedding = nn.Embedding(num_rel_labels, hidden_dim)
        self.relation_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Final node feature fusion
        self.node_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def get_relation_type(self, node_type_i, node_type_j):
        """
        Determine relation type between two nodes.
        0: background-background (R_BB)
        1: background-foreground (R_BF)
        2: foreground-foreground (R_FF)

        Node type: 0 = background, >0 = foreground class
        """
        is_bg_i = (node_type_i == 0).float()
        is_bg_j = (node_type_j == 0).float()

        # R_BB: both background
        rel_BB = (is_bg_i * is_bg_j).bool()
        # R_BF: one bg, one fg
        rel_BF = ((is_bg_i + is_bg_j) == 1).bool()
        # R_FF: both foreground
        rel_FF = (~is_bg_i.bool()) & (~is_bg_j.bool())

        return rel_BB, rel_BF, rel_FF

    def compute_relative_position_label(self, boxes_i, boxes_j):
        """
        Compute discrete relative position label between two boxes.

        Args:
            boxes_i: (N, 4) normalized (cx, cy, w, h)
            boxes_j: (N, 4) normalized (cx, cy, w, h)

        Returns:
            labels: (N,) integer labels in [0, num_rel_labels-1]
        """
        # Compute relative position features
        dx = (boxes_i[:, 0] - boxes_j[:, 0]) / (boxes_j[:, 2] + 1e-6)
        dy = (boxes_i[:, 1] - boxes_j[:, 1]) / (boxes_j[:, 3] + 1e-6)

        # Discretize into 8 relative positions:
        # 0: above-left, 1: above, 2: above-right
        # 3: left, 4: center/overlap
        # 5: right, 6: below-left, 7: below-right
        angle = torch.atan2(dy, dx) / math.pi  # [-1, 1]

        # 8 direction sectors
        labels = ((angle + 1.0) / 2.0 * self.num_rel_labels).long()
        labels = labels.clamp(0, self.num_rel_labels - 1)

        return labels

    def build_adjacency(self, relation_types, device):
        """
        Build adjacency matrices for each relation type.

        Args:
            relation_types: tuple of (rel_BB, rel_BF, rel_FF) masks
                           each of shape (N, N)
            device: torch device

        Returns:
            adj_mats: (num_relations, N, N) adjacency matrices
        """
        N = relation_types[0].size(0)
        adj_mats = torch.zeros(self.num_relations, N, N, device=device)

        for r_idx, rel_mask in enumerate(relation_types):
            adj_mats[r_idx] = rel_mask.float()

        return adj_mats

    def forward(self, text_features, boxes, class_ids, bg_indices=None):
        """
        Build the constraint graph and initialize node features.

        Args:
            text_features: (N, text_dim) text semantic embeddings from pre-trained encoder
            boxes: (N, 4) normalized bounding box coordinates (cx, cy, w, h)
            class_ids: (N,) class IDs for each element (0 for background)
            bg_indices: optional indices of background nodes

        Returns:
            node_features: (N, hidden_dim) initial node features h_i^(0)
            adj_mats: (num_relations, N, N) adjacency matrices
            edge_labels: (N, N) discrete relative position labels for supervision
        """
        N = boxes.size(0)
        device = boxes.device

        if bg_indices is None:
            bg_indices = (class_ids == 0)

        # 1. Build node features
        # Text embedding: e_i^text
        e_text = self.text_proj(text_features)  # (N, hidden_dim)

        # Position embedding: e_i^pos
        e_pos = self.pos_encoder(boxes)
        e_pos = self.pos_proj(e_pos)  # (N, hidden_dim)

        # Class embedding: e_i^cls
        e_cls = self.class_embed(class_ids.long().clamp(0, self.num_classes))
        e_cls = self.cls_proj(e_cls)  # (N, hidden_dim)

        # Initial node features: h_i^(0) = [e_i^text || e_i^pos || e_i^cls]
        node_features = torch.cat([e_text, e_pos, e_cls], dim=-1)
        node_features = self.node_fusion(node_features)  # (N, hidden_dim)

        # 2. Build adjacency matrices (relation types)
        # For all pairs, determine relation type
        node_types = (class_ids > 0).long()  # 0=bg, 1=fg

        adj_mats = torch.zeros(self.num_relations, N, N, device=device)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue  # No self-loop for relation edges
                type_i = node_types[i]
                type_j = node_types[j]

                if type_i == 0 and type_j == 0:
                    adj_mats[0, i, j] = 1.0  # R_BB
                elif (type_i == 0 and type_j == 1) or (type_i == 1 and type_j == 0):
                    adj_mats[1, i, j] = 1.0  # R_BF
                else:
                    adj_mats[2, i, j] = 1.0  # R_FF

        # 3. Compute edge labels for relation reconstruction supervision
        edge_labels = self.compute_relative_position_label(
            boxes.unsqueeze(1).expand(-1, N, -1).reshape(-1, 4),
            boxes.unsqueeze(0).expand(N, -1, -1).reshape(-1, 4)
        ).reshape(N, N)

        return node_features, adj_mats, edge_labels
