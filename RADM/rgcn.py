# ========================================
# RGCN: Relation-aware Graph Convolutional Network
# for heterogeneous constraint graph encoding
#
# Reference: yuanRGNNRecurrentGraph2024
# ========================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RGCNLayer(nn.Module):
    """
    Single layer of Relation-aware Graph Convolutional Network.
    Performs message passing for each relation type separately,
    then aggregates across relation types.
    """

    def __init__(self, in_dim, out_dim, num_relations, num_bases=None,
                 dropout=0.0, activation=F.relu, self_loop=True):
        super(RGCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.dropout_rate = dropout
        self.activation = activation
        self.self_loop = self_loop

        if num_bases is not None and num_bases > 0:
            # Basis decomposition for parameter efficiency
            self.weight_basis = nn.Parameter(torch.Tensor(num_bases, in_dim, out_dim))
            nn.init.xavier_uniform_(self.weight_basis)
            self.coeff = nn.Parameter(torch.Tensor(num_relations, num_bases))
            nn.init.xavier_uniform_(self.coeff)
        else:
            # Independent weight per relation type
            self.weight = nn.Parameter(torch.Tensor(num_relations, in_dim, out_dim))
            nn.init.xavier_uniform_(self.weight)

        if self_loop:
            self.self_weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
            nn.init.xavier_uniform_(self.self_weight)

        self.bias = nn.Parameter(torch.Tensor(out_dim))
        nn.init.zeros_(self.bias)

        self.dropout = nn.Dropout(dropout)

    def _get_relation_weights(self):
        """Get weight matrix for each relation type."""
        if self.num_bases is not None and self.num_bases > 0:
            # W_r = sum_b (c_rb * V_b)
            weight = torch.einsum('rb,bio->rio', self.coeff, self.weight_basis)
        else:
            weight = self.weight
        return weight

    def forward(self, node_features, adj_mats):
        """
        Args:
            node_features: (N, in_dim) node feature matrix
            adj_mats: (num_relations, N, N) adjacency matrices, one per relation type
                       adj_mats[r][i][j] = 1 if edge (j->i) of relation type r exists

        Returns:
            (N, out_dim) updated node features
        """
        N = node_features.size(0)
        weights = self._get_relation_weights()  # (num_relations, in_dim, out_dim)

        # Message passing per relation type
        messages = []
        for r in range(self.num_relations):
            # Normalize adjacency by in-degree (symmetric normalization)
            adj = adj_mats[r]  # (N, N)
            if adj.sum() > 0:
                # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
                degree = adj.sum(dim=1).clamp(min=1)  # out-degree
                deg_inv_sqrt = torch.diag(degree.pow(-0.5))
                # in-degree normalization
                degree_in = adj.sum(dim=0).clamp(min=1)
                deg_inv_sqrt_in = torch.diag(degree_in.pow(-0.5))

                norm_adj = deg_inv_sqrt @ adj @ deg_inv_sqrt_in

                # Message: norm_adj * H * W_r
                support = node_features @ weights[r]  # (N, out_dim)
                msg = norm_adj @ support
            else:
                msg = torch.zeros(N, self.out_dim, device=node_features.device)

            messages.append(msg)

        # Aggregate messages across relation types (sum aggregation)
        output = torch.stack(messages, dim=0).sum(dim=0)  # (N, out_dim)

        # Self-loop
        if self.self_loop:
            output = output + node_features @ self.self_weight

        output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)

        output = self.dropout(output)

        return output


class RGCN(nn.Module):
    """
    Multi-layer Relation-aware GCN for topological reasoning over constraint graphs.

    Given the constraint graph G = (V, E) where:
      - V = V_bg ∪ V_fg (background + foreground element nodes)
      - E = R_BB ∪ R_BF ∪ R_FF (3 relation types)
    This module performs L-layer message passing to obtain H_topo.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_relations=3, num_layers=2,
                 num_bases=4, dropout=0.1):
        super(RGCN, self).__init__()
        self.num_layers = num_layers
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Layer stack
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        # Input layer
        self.layers.append(
            RGCNLayer(in_dim, hidden_dim, num_relations, num_bases, dropout)
        )
        self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(
                RGCNLayer(hidden_dim, hidden_dim, num_relations, num_bases, dropout)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_dim)

        # Node feature projection (from initial multi-modal features)
        self.node_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, node_features, adj_mats):
        """
        Args:
            node_features: (N, in_dim) initial node features h_i^(0)
                           = [e_i^text || e_i^pos || e_i^cls]
            adj_mats: (num_relations, N, N) adjacency for each relation type

        Returns:
            H_topo: (N, out_dim) final node representations after L-layer reasoning,
                    encoding both semantic context and topological constraints
        """
        h = self.node_proj(node_features)

        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            h_res = h
            h = layer(h, adj_mats)
            h = norm(h + h_res)  # Residual connection + LayerNorm

        H_topo = self.output_proj(h)

        return H_topo

    def get_intermediate_features(self, node_features, adj_mats):
        """Get features from all layers for auxiliary supervision."""
        intermediates = []
        h = self.node_proj(node_features)

        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            h_res = h
            h = layer(h, adj_mats)
            h = norm(h + h_res)
            intermediates.append(h)

        return intermediates, self.output_proj(h)


class RelationClassifier(nn.Module):
    """
    Relation reconstruction head.
    Given a pair of node features (h_i, h_j), predicts the relative position category.
    Used for auxiliary relation reconstruction loss L_rel.
    """

    def __init__(self, in_dim, num_rel_categories=8, hidden_dim=128):
        super(RelationClassifier, self).__init__()
        self.num_rel_categories = num_rel_categories

        # Pair feature: concatenate h_i and h_j, plus element-wise product
        pair_dim = in_dim * 3  # [h_i, h_j, h_i * h_j]
        self.classifier = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_rel_categories),
        )

    def forward(self, h_i, h_j):
        """
        Args:
            h_i: (N, in_dim) source node features
            h_j: (N, in_dim) target node features

        Returns:
            logits: (N, num_rel_categories)
        """
        pair_feat = torch.cat([h_i, h_j, h_i * h_j], dim=-1)
        return self.classifier(pair_feat)


class GridPositionPredictor(nn.Module):
    """
    Grid position classification head.
    Given node features h_i, predicts the S×S grid index of the node's absolute position.
    Used for grid position classification loss L_pos.
    """

    def __init__(self, in_dim, grid_size=8, hidden_dim=128):
        super(GridPositionPredictor, self).__init__()
        self.grid_size = grid_size
        self.num_cells = grid_size * grid_size

        self.predictor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.num_cells),
        )

    def forward(self, h):
        """
        Args:
            h: (N, in_dim) node features

        Returns:
            logits: (N, grid_size^2) grid cell prediction logits
        """
        return self.predictor(h)

    def get_grid_labels(self, positions, canvas_size=1.0):
        """
        Convert continuous positions to discrete grid indices.

        Args:
            positions: (N, 2) center (cx, cy) normalized to [0, 1]
            canvas_size: default 1.0 (normalized coordinates)

        Returns:
            grid_indices: (N,) integer grid index [0, grid_size^2 - 1]
        """
        cx = positions[:, 0].clamp(0.0, canvas_size - 1e-6)
        cy = positions[:, 1].clamp(0.0, canvas_size - 1e-6)

        cell_x = (cx * self.grid_size).long().clamp(0, self.grid_size - 1)
        cell_y = (cy * self.grid_size).long().clamp(0, self.grid_size - 1)

        grid_indices = cell_y * self.grid_size + cell_x
        return grid_indices
