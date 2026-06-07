# ========================================
# Background Visual Prior Encoder
# DINO-ViT based visual feature extraction + Linear Adapter + RoIAlign
#
# Reference: DINOv2 (oquab2024dinov), RoIAlign (he2017mask)
# ========================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DinoViTEncoder(nn.Module):
    """
    Self-supervised pre-trained DINO-ViT as visual feature extraction backbone.

    Input: background image I_bg ∈ R^(H_img × W_img × 3)
    Output: Z_spatial = [z_1, z_2, ..., z_Np] spatial tokens

    Uses DINOv2 by default. The ViT divides the image into P×P patches,
    yielding N_p = (H_img/P) × (W_img/P) patch tokens.
    """

    def __init__(self, model_name="dinov2_vits14", freeze_backbone=True,
                 feature_dim=384, patch_size=14):
        super().__init__()
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.freeze_backbone = freeze_backbone

        # We use torch.hub to load DINOv2, but also support a placeholder
        # for when the model isn't available - users should install it
        self.encoder = None  # Lazy loading
        self._init_encoder(model_name, freeze_backbone)

    def _init_encoder(self, model_name, freeze_backbone):
        """Initialize the DINO-ViT encoder."""
        try:
            self.encoder = torch.hub.load('facebookresearch/dinov2', model_name)
            if freeze_backbone:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder.eval()
        except Exception:
            # Placeholder: create a minimal ViT-like encoder
            # Users should install dinov2 via: pip install dinov2
            print(f"Warning: Could not load {model_name} from torch.hub. "
                  "Using placeholder encoder. Install dinov2 for full functionality.")
            self.encoder = self._build_placeholder()

    def _build_placeholder(self):
        """Build a placeholder ViT encoder when DINOv2 is not available."""
        return PlaceholderViT(
            img_size=224,
            patch_size=self.patch_size,
            embed_dim=self.feature_dim,
            depth=12,
            num_heads=6,
        )

    @torch.no_grad()
    def forward(self, image):
        """
        Args:
            image: (B, 3, H, W) input background images

        Returns:
            Z_spatial: (B, N_p, feature_dim) spatial patch tokens (without CLS)
        """
        if self.freeze_backbone and self.encoder is not None:
            self.encoder.eval()

        # DINOv2 returns patch tokens; we extract spatial tokens
        if hasattr(self.encoder, 'get_intermediate_layers'):
            # DINOv2 API
            features = self.encoder.get_intermediate_layers(
                image, n=1, return_class_token=False
            )
            spatial_tokens = features[0]  # (B, N_p, feature_dim)
        elif hasattr(self.encoder, 'forward_features'):
            # Standard ViT API
            spatial_tokens = self.encoder.forward_features(image)
        else:
            # Placeholder
            spatial_tokens = self.encoder(image)

        return spatial_tokens


class PlaceholderViT(nn.Module):
    """Minimal ViT placeholder when DINOv2 is unavailable."""

    def __init__(self, img_size=224, patch_size=14, embed_dim=384, depth=12, num_heads=6):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=min(depth, 3))

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        H_p, W_p = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, N_p, embed_dim)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)
        return x


class LinearAdapter(nn.Module):
    """
    Lightweight linear adapter for dimension mapping.
    Maps DINO-ViT feature dimension to target dimension:

    F_vis = W_proj * F + b_proj
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x):
        """
        Args:
            x: (B, N_p, in_channels) or (B, C_in, H_f, W_f)

        Returns:
            (B, out_channels, H_f, W_f) feature map
        """
        if x.dim() == 3:
            B, N, C = x.shape
            x = self.proj(x)
            # Reshape to 2D feature map
            H_f = W_f = int(N ** 0.5)
            x = x.transpose(1, 2).view(B, -1, H_f, W_f)
        elif x.dim() == 4:
            B, C, H_f, W_f = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, H_f*W_f, C)
            x = self.proj(x)
            x = x.transpose(1, 2).view(B, -1, H_f, W_f)

        return x


class RoIExtractor(nn.Module):
    """
    RoIAlign-based region visual feature extraction.

    Given visual feature map F_vis and candidate layout boxes x,
    extracts region features: V = RoIAlign(F_vis, x)
    """

    def __init__(self, output_size=7, spatial_scale=1.0, sampling_ratio=2):
        """
        Args:
            output_size: (H_r, W_r) output resolution
            spatial_scale: scale factor from feature map to input image
            sampling_ratio: number of sampling points per bin
        """
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, feature_map, boxes, image_size=None):
        """
        Args:
            feature_map: (B, C, H_f, W_f) visual feature map
            boxes: list of (N_i, 4) boxes in (x1, y1, x2, y2) format, normalized [0, 1]
            image_size: optional (H, W) of original image

        Returns:
            roi_features: list of (N_i, C, H_r, W_r) RoI features
        """
        if image_size is not None:
            self.spatial_scale = feature_map.size(-1) / max(image_size)

        roi_features = []
        B = feature_map.size(0)

        for b in range(B):
            if isinstance(boxes, list):
                boxes_b = boxes[b]  # (N_i, 4) or (N_i, 5) with batch_idx
            else:
                boxes_b = boxes[b]

            if boxes_b.ndim == 2 and boxes_b.size(1) >= 4:
                # Convert normalized [0,1] to feature map coordinates
                boxes_scaled = boxes_b[:, :4].clone()
                boxes_scaled[:, [0, 2]] *= feature_map.size(-1)
                boxes_scaled[:, [1, 3]] *= feature_map.size(-2)

                # RoIAlign
                roi_feat = torchvision.ops.roi_align(
                    feature_map[b:b + 1],
                    [boxes_scaled],
                    output_size=self.output_size,
                    spatial_scale=1.0,
                    sampling_ratio=self.sampling_ratio,
                    aligned=True,
                )
                roi_features.append(roi_feat)
            else:
                roi_features.append(None)

        return roi_features


class VisualPriorEncoder(nn.Module):
    """
    Full background visual prior encoding pipeline.

    1. DINO-ViT extracts spatial tokens Z_spatial
    2. Reshape to 2D feature map F = Reshape(Z_spatial)
    3. Linear Adapter maps dimension: F_vis = W_proj * F + b_proj
    4. RoIAlign extracts region features: V = RoIAlign(F_vis, x)
    """

    def __init__(self, model_name="dinov2_vits14", freeze_backbone=True,
                 feature_dim=384, out_channels=256, roi_output_size=7,
                 patch_size=14):
        super().__init__()
        self.out_channels = out_channels
        self.feature_dim = feature_dim
        self.roi_output_size = roi_output_size

        # DINO-ViT backbone
        self.vit_encoder = DinoViTEncoder(
            model_name=model_name,
            freeze_backbone=freeze_backbone,
            feature_dim=feature_dim,
            patch_size=patch_size,
        )

        # Linear adapter
        self.adapter = LinearAdapter(feature_dim, out_channels)

        # RoIAlign extractor
        self.roi_extractor = RoIExtractor(
            output_size=roi_output_size,
        )

    def forward(self, images, boxes=None, return_feature_map=True):
        """
        Args:
            images: (B, 3, H, W) background images
            boxes: optional list of (N_i, 4) boxes for RoI extraction
            return_feature_map: whether to return the full feature map

        Returns:
            F_vis: (B, C, H_f, W_f) visual feature map
            roi_features: list of (N_i, C, H_r, W_r) RoI features (if boxes provided)
        """
        # Step 1: Extract spatial tokens
        Z_spatial = self.vit_encoder(images)  # (B, N_p, feature_dim)

        # Step 2: Reshape to 2D feature map
        B, N_p, _ = Z_spatial.shape
        H_f = W_f = int(N_p ** 0.5)

        # Step 3: Linear adapter
        F_vis = self.adapter(
            Z_spatial.transpose(1, 2).view(B, self.feature_dim, H_f, W_f)
        )  # (B, out_channels, H_f, W_f)

        # Step 4: RoIAlign extraction
        roi_features = None
        if boxes is not None:
            roi_features = []
            for b in range(B):
                if boxes[b] is not None and boxes[b].numel() > 0:
                    boxes_b = boxes[b][:, :4].clone()
                    boxes_b[:, [0, 2]] *= F_vis.size(-1)
                    boxes_b[:, [1, 3]] *= F_vis.size(-2)

                    if boxes_b.size(0) > 0:
                        import torchvision.ops as tv_ops
                        roi_feat = tv_ops.roi_align(
                            F_vis[b:b + 1], [boxes_b],
                            output_size=self.roi_output_size,
                            spatial_scale=1.0,
                            sampling_ratio=-1,
                            aligned=True,
                        )
                        roi_features.append(roi_feat)
                    else:
                        roi_features.append(
                            torch.zeros(0, self.out_channels,
                                        self.roi_output_size, self.roi_output_size,
                                        device=F_vis.device)
                        )
                else:
                    roi_features.append(None)

        if return_feature_map:
            return F_vis, roi_features
        return roi_features


# Import torchvision for roi_align
import torchvision.ops
