import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math

from RADM.util.box_ops import box_xyxy_to_cxcywh

# ========================================
# DDPO (Diffusion-DPO) Components for RADM
# ========================================
class PreferenceModel(nn.Module):
    """
    偏好模型：评估生成布局的质量
    输入：布局特征 + 文本特征
    输出：质量评分 (0-1之间的概率)
    """
    def __init__(self, layout_dim=256, text_dim=768, hidden_dim=512):
        super().__init__()
        self.layout_encoder = nn.Sequential(
            nn.Linear(layout_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.preference_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出0-1之间的偏好分数
        )

    def forward(self, layout_features, text_features):
        """
        Args:
            layout_features: [batch_size, num_proposals, layout_dim]
            text_features: [batch_size, text_dim]
        Returns:
            preference_scores: [batch_size, num_proposals]
        """
        # 编码布局特征
        layout_encoded = self.layout_encoder(layout_features)  # [B, N, H]

        # 编码文本特征
        text_encoded = self.text_encoder(text_features)  # [B, H]
        text_encoded = text_encoded.unsqueeze(1).repeat(1, layout_features.size(1), 1)  # [B, N, H]

        # 融合特征
        combined = torch.cat([layout_encoded, text_encoded], dim=-1)  # [B, N, 2H]

        # 预测偏好分数
        scores = self.preference_head(combined).squeeze(-1)  # [B, N]

        return scores


class DDPOTrainer:
    """
    修正的DDPO (Diffusion-DPO) 训练器
    实现真正的轨迹概率计算和成对偏好优化
    """
    def __init__(self, model, preference_model, reference_model=None, beta=0.1, sample_size=4):
        self.model = model
        self.preference_model = preference_model
        # 参考模型可以是早期版本的模型，或者使用固定的噪声预测
        self.reference_model = reference_model if reference_model is not None else model
        self.beta = beta  # DPO的beta参数
        self.sample_size = sample_size  # 每次采样数量
        # 确保所有模型在同一设备上
        self.device = next(model.parameters()).device

    def compute_denoising_confidence_score(self, x_0, text_features, txt_mask, model):
        """
        计算模型对生成样本的"置信度"评分
        使用去噪一致性作为概率质量的代理指标

        这是对传统DDPO概率计算的实用替代方案：
        - 使用模型的去噪能力来评估生成质量
        - 计算多时间步的一致性得分
        - 避免复杂的轨迹概率积分
        """
        batch_size = x_0.shape[0]

        # 重新参数化到扩散空间
        x_0_norm = (x_0 * 2. - 1.) * model.scale  # [B, N, 4]

        # 选择多个时间步进行一致性检查
        timesteps = torch.tensor([100, 500, 900], device=x_0.device)  # 早期、中期、晚期
        consistency_scores = []

        for t in timesteps:
            t_batch = t.unsqueeze(0).repeat(batch_size, 1)

            # 正向扩散：添加噪声
            noise_gt = torch.randn_like(x_0_norm)
            x_t = model.q_sample(x_start=x_0_norm, t=t_batch, noise=noise_gt)

            # 逆向预测：模型预测噪声
            with torch.no_grad():
                pred_noise, _, _ = model.model_predictions(
                    None,  # backbone feats placeholder
                    torch.tensor([[1.0, 1.0, 1.0, 1.0]] * batch_size, device=x_0.device),  # image_whwh placeholder
                    x_t, text_features, txt_mask, t_batch
                )

            # 计算预测一致性 (越小越好)
            denoising_error = torch.mean((pred_noise - noise_gt) ** 2, dim=[1, 2])  # [B]

            # 转换为置信度得分 (去噪误差越小，置信度越高)
            confidence = -denoising_error  # 负误差作为"质量"指标

            consistency_scores.append(confidence)

        # 计算多时间步的一致性平均
        consistency_score = torch.stack(consistency_scores, dim=0).mean(dim=0)  # [B]

        return consistency_score

    def compute_simple_density_proxy(self, x_0, text_features, txt_mask, model):
        """
        计算简化的密度代理 (承认这不是真正的概率密度)
        使用去噪重建质量作为生成质量的代理

        注意：这个方法主要用于演示和快速原型，不建议用于生产环境
        """
        batch_size = x_0.shape[0]

        # 重新参数化到扩散空间
        x_0_norm = (x_0 * 2. - 1.) * model.scale

        # 选择一个合理的时间步 (不是T/2，因为那只是启发式)
        t = torch.tensor([model.num_timesteps // 3], device=x_0.device).repeat(batch_size)

        # 添加噪声并重建
        noise = torch.randn_like(x_0_norm)
        x_t = model.q_sample(x_start=x_0_norm, t=t, noise=noise)

        # 使用模型重建
        with torch.no_grad():
            pred_noise, _, _ = model.model_predictions(
                None,  # backbone feats
                torch.tensor([[1.0, 1.0, 1.0, 1.0]] * batch_size, device=x_0.device),  # image_whwh
                x_t, text_features, txt_mask, t
            )

            # 从预测噪声重建x_0
            pred_x0 = (x_t - (1 - model.alphas_cumprod[t[0]]).sqrt() * pred_noise) / model.alphas_cumprod[t[0]].sqrt()
            pred_x0 = torch.clamp(pred_x0, -model.scale, model.scale)

        # 计算重建质量 (这是合理的代理指标)
        reconstruction_error = torch.mean((pred_x0 - x_0_norm) ** 2, dim=[1, 2])  # [B]

        # 转换为"密度"代理 (重建误差越小，"密度"越高)
        density_proxy = -reconstruction_error

        return density_proxy


    def compute_ddpo_loss(self, samples_with_trajectory, batch_inputs, text_features, txt_mask, rewards, chosen_indices=None, rejected_indices=None):
        """
        计算真正的DDPO损失：使用replay模式比较同一个y在不同策略下的概率

        Args:
            samples_with_trajectory: [(layout, trajectory), ...] 从sample_layouts_with_trajectory返回的列表
            batch_inputs: 原始batch输入
            text_features: [batch_size, text_dim] 文本特征
            txt_mask: [batch_size, seq_len] 文本mask
            rewards: [sample_size, batch_size] 奖励分数 (如果不使用成对偏好)
            chosen_indices: Optional[List[int]] 每个batch item的优胜样本索引 [batch_size]
            rejected_indices: Optional[List[int]] 每个batch item的劣质样本索引 [batch_size]
        """
        sample_size = len(samples_with_trajectory)
        if sample_size == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        batch_size = samples_with_trajectory[0][0].shape[0]  # 从第一个layout获取batch_size

        # 计算当前模型在轨迹上的log prob
        log_p_theta = torch.zeros(sample_size, batch_size, device=self.device)
        for i, (_, trajectory) in enumerate(samples_with_trajectory):
            # 当前模型在自己的轨迹上的log prob (应该接近真实轨迹)
            log_p_theta[i] = self.compute_log_prob_for_trajectory(self.model, trajectory)

        # 计算参考模型在相同轨迹上的log prob (DDPO replay模式!)
        log_p_ref = torch.zeros(sample_size, batch_size, device=self.device)
        for i, (_, trajectory) in enumerate(samples_with_trajectory):
            # 参考模型在当前模型生成的轨迹上的log prob
            log_p_ref[i] = self.compute_log_prob_for_trajectory(self.reference_model, trajectory)

        # 处理成对偏好数据 (标准DDPO格式)
        if chosen_indices is not None and rejected_indices is not None:
            # 修正维度：chosen_indices是每个batch item对应的样本索引
            chosen_indices = torch.tensor(chosen_indices, device=self.device, dtype=torch.long)  # [batch_size]
            rejected_indices = torch.tensor(rejected_indices, device=self.device, dtype=torch.long)  # [batch_size]

            # 选择对应的log probs: [sample_size, batch_size] -> [batch_size] (选择正确的样本)
            log_p_theta_chosen = log_p_theta[chosen_indices, torch.arange(batch_size)]  # [batch_size]
            log_p_theta_rejected = log_p_theta[rejected_indices, torch.arange(batch_size)]  # [batch_size]
            log_p_ref_chosen = log_p_ref[chosen_indices, torch.arange(batch_size)]  # [batch_size]
            log_p_ref_rejected = log_p_ref[rejected_indices, torch.arange(batch_size)]  # [batch_size]

            # 标准DDPO目标:
            # max_θ E[log σ(β(log p_θ(y_w) - log p_θ(y_l) - log p_ref(y_w) + log p_ref(y_l)))]
            advantages = (log_p_theta_chosen - log_p_theta_rejected) - (log_p_ref_chosen - log_p_ref_rejected)
            ddpo_loss = -torch.log(torch.sigmoid(self.beta * advantages)).mean()

        else:
            # 回退到单样本奖励模式
            rewards = rewards.view(sample_size, -1)  # [sample_size, batch_size]

            # 获取偏好模型评分作为奖励信号
            layouts = [layout for layout, _ in samples_with_trajectory]
            layout_samples = torch.stack(layouts, dim=0)  # [sample_size, batch_size, num_proposals, 4]

            # 调试：检查布局特征维度
            import json
            import time
            debug_data = {
                "layout_samples_shape": list(layout_samples.shape),
                "sample_size": sample_size,
                "batch_size": layout_samples.shape[1],
                "num_proposals": layout_samples.shape[2],
                "layout_dim": layout_samples.shape[3]  # 应该是4
            }

            log_entry = json.dumps({
                "id": f"log_{int(time.time()*1000)}_layout",
                "timestamp": int(time.time()*1000),
                "location": "RADM/layers.py:compute_ddpo_loss",
                "message": "DDPO layout features dimensions",
                "data": debug_data,
                "sessionId": "debug-session",
                "runId": "run2",
                "hypothesisId": "FIX_VERIFICATION"
            })

            with open("/home/sxm/flux-workspace/text-to-layout-zhuanlan/BASE-RADM/RADM/.cursor/debug.log", "a") as f:
                f.write(log_entry + "\n")

            num_proposals = layout_samples.shape[2]
            layout_flat = layout_samples.view(-1, num_proposals, 4)  # [sample_size*batch_size, num_proposals, 4]
            # 调试：检查输入维度
            # print(f"text_features.shape: {text_features.shape}")
            # print(f"txt_mask.shape: {txt_mask.shape}")
            # print(f"sample_size: {sample_size}")

            # 处理text_features：如果是3D [batch_size, seq_len, hidden_dim]，需要池化到 [batch_size, hidden_dim]
            if text_features.dim() == 3:
                # 对序列维度进行平均池化，得到 [batch_size, hidden_dim]
                text_features = text_features.mean(dim=1)  # [1, 20, 768] -> [1, 768]

            # 处理txt_mask：如果是3D，需要相应调整
            if txt_mask.dim() == 3:
                # 对序列维度进行平均，得到 [batch_size, 1]
                txt_mask = txt_mask.float().mean(dim=1)  # [1, 20, 1] -> [1, 1]

            text_flat = text_features.unsqueeze(0).repeat(sample_size, 1, 1).view(-1, text_features.size(-1))
            txt_mask_flat = txt_mask.unsqueeze(0).repeat(sample_size, 1, 1).view(-1, txt_mask.size(-1))

            preference_scores = self.preference_model(layout_flat, text_flat)
            preference_scores = preference_scores.view(sample_size, batch_size, num_proposals)
            avg_preferences = preference_scores.mean(dim=-1)  # [sample_size, batch_size]

            combined_rewards = avg_preferences + rewards

            # 对于单样本DDPO，使用简化的优势计算
            # 鼓励生成高质量布局：更高的奖励 = 更好的生成
            advantages = self.beta * combined_rewards  # 只使用奖励信号

            # 如果有log概率信息，也可以使用它
            if log_p_theta is not None and log_p_ref is not None:
                advantages = advantages + (log_p_theta - log_p_ref)

            ddpo_loss = -torch.log(torch.sigmoid(advantages)).mean()

            # 调试：记录优势统计
            import json
            import time
            debug_data = {
                "advantages_mean": advantages.mean().item(),
                "advantages_std": advantages.std().item(),
                "combined_rewards_mean": combined_rewards.mean().item(),
                "avg_preferences_mean": avg_preferences.mean().item(),
                "rewards_mean": rewards.mean().item()
            }

            log_entry = json.dumps({
                "id": f"log_{int(time.time()*1000)}_ddpo_debug",
                "timestamp": int(time.time()*1000),
                "location": "RADM/layers.py:compute_ddpo_loss",
                "message": "DDPO advantages and rewards debug",
                "data": debug_data,
                "sessionId": "debug-session",
                "runId": "run3",
                "hypothesisId": "DDPO_LOSS_DEBUG"
            })

            with open("/home/sxm/flux-workspace/text-to-layout-zhuanlan/BASE-RADM/RADM/.cursor/debug.log", "a") as f:
                f.write(log_entry + "\n")

        return ddpo_loss

    def sample_layouts_with_trajectory(self, batch_inputs, text_features, txt_mask, num_samples=4):
        """
        从当前模型采样多个布局，并记录完整轨迹用于DDPO replay
        Args:
            batch_inputs: detectron2 batch (list of dicts)
            text_features: [batch_size, text_dim]
            txt_mask: [batch_size, seq_len]
            num_samples: 采样次数
        返回: (layouts, trajectories) 元组列表，其中trajectory包含每一步的状态
        """
        # batch_inputs 是 detectron2 的 batch 格式 (list of dicts)
        # 我们使用第一个batch元素进行采样
        batch_list = batch_inputs if isinstance(batch_inputs, list) else [batch_inputs]

        # 预处理图像数据 (一次性完成，避免重复计算)
        images, images_whwh = self.model.preprocess_image(batch_list)

        # 按照RADM的标准方式处理backbone特征
        src = self.model.backbone(images.tensor)
        backbone_feats = list()
        for f in self.model.in_features:
            feature = src[f]
            backbone_feats.append(feature)

        samples_with_trajectory = []

        for _ in range(num_samples):
            # 调用轨迹记录采样方法
            results, trajectory = self.model.ddim_sample_with_trajectory(
                batch_list, backbone_feats, images_whwh, images,
                text_features, txt_mask, return_trajectory=True
            )

            # 将结果转换为布局特征
            sample_layout = self.results_to_layout_features(results, images_whwh)

            samples_with_trajectory.append((sample_layout, trajectory))

        return samples_with_trajectory

    def results_to_layout_features(self, results, images_whwh):
        """
        将推理结果转换为布局特征表示
        Args:
            results: 推理结果列表，每个元素包含预测的boxes, scores, classes
            images_whwh: 图像尺寸 [batch_size, 4] (w, h, w, h)
        Returns:
            layout_features: [batch_size, num_proposals, 4] 布局框坐标 (cx, cy, w, h)
        """
        batch_size = len(results)
        num_proposals = self.model.num_proposals
        layout_features = torch.zeros(batch_size, num_proposals, 4, device=self.device)

        for i, result in enumerate(results):
            if isinstance(result, dict) and "instances" in result:
                instances = result["instances"]
            else:
                instances = result

            if len(instances) > 0:
                # 获取预测框 (x1, y1, x2, y2)
                boxes = instances.pred_boxes.tensor  # [num_instances, 4]

                # 转换为中心坐标格式 (cx, cy, w, h)
                boxes_center = box_xyxy_to_cxcywh(boxes)  # [num_instances, 4]

                # 归一化到0-1范围
                img_whwh = images_whwh[i]  # [4] (w, h, w, h)
                boxes_center = boxes_center / img_whwh[:4]

                # 填充到固定大小
                num_instances = min(len(boxes_center), num_proposals)
                layout_features[i, :num_instances] = boxes_center[:num_instances]

                # 剩余位置用零填充 (已经在初始化时完成)

        return layout_features

    def compute_log_prob_for_trajectory(self, model, trajectory):
        """
        对给定的轨迹，用指定模型计算log prob (修正版：计算轨迹实际发生的概率)
        关键修正：计算轨迹中实际发生的x_prev在当前模型下的概率密度
        """
        steps = trajectory['steps']
        batch_size = trajectory['image_whwh'].shape[0]

        total_log_prob = torch.zeros(batch_size, device=self.device)

        # 遍历轨迹的每一步
        for step in steps:
            x_t = step['x_t']
            t = step['t']
            time_next = step['time_next']

            # 关键修正1：获取轨迹中真实发生的下一步 x_prev_actual
            # sample_layouts_with_trajectory必须保存这一项！
            if 'x_prev' not in step:
                # 如果轨迹没有保存x_prev，跳过这一步
                continue

            x_prev_actual = step['x_prev']  # 轨迹中实际发生的x_{t-1}

            # Replay: 用当前模型预测噪声
            with torch.no_grad():
                preds, _, _ = model.model_predictions(
                    None,  # backbone feats placeholder
                    trajectory['image_whwh'],
                    x_t, trajectory['text_feature'], trajectory['txt_mask'], t
                )
                pred_noise = preds.pred_noise

            # 计算条件概率 log p(x_{prev} | x_t, θ)
            if time_next >= 0:
                alpha = step['alpha']
                alpha_next = step['alpha_next']
                eta = step['eta']

                # DDIM参数
                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()

                if sigma.sum() == 0:
                    # 如果是确定性采样(DDIM eta=0)，概率密度是无穷大(Dirac delta)
                    # 跳过这一步或给一个大的常数值
                    continue

                c = (1 - alpha_next - sigma ** 2).sqrt()

                # 关键修正2：基于当前模型的预测，推导当前的高斯均值 mu_theta
                # 先反推 x_0 (reparameterization)
                pred_x_start = (x_t - (1 - alpha).sqrt() * pred_noise) / alpha.sqrt()
                pred_x_start = torch.clamp(pred_x_start, -model.scale, model.scale)

                # 计算均值 mu (根据DDIM更新公式，去掉噪声项sigma*z)
                mu_theta = pred_x_start * alpha_next.sqrt() + c * pred_noise

                # 关键修正3：计算实际轨迹点x_prev_actual在该分布下的Log Probability
                # p(x_prev | x_t) ~ N(mu_theta, sigma^2 * I)
                # Log Gaussian: -0.5 * Σ((x - mu)/sigma)^2 - Σ log(sigma * sqrt(2π))

                # 计算方差 (对每个维度)
                variance = sigma ** 2 + 1e-8  # 避免除零

                # 计算平方差
                squared_diff = (x_prev_actual - mu_theta) ** 2

                # 对数概率密度 (逐元素)
                step_log_prob = -0.5 * (squared_diff / variance).sum(dim=[1, 2])

                # 减去归一化常数 log(Z) = Σ log(σ * sqrt(2π))
                num_elements = x_prev_actual.shape[1] * x_prev_actual.shape[2]
                normalization = num_elements * torch.log(sigma * (2 * torch.pi).sqrt() + 1e-8)
                step_log_prob = step_log_prob - normalization

                total_log_prob += step_log_prob

        return total_log_prob

    def sample_layouts(self, batch_inputs, text_features, txt_mask, num_samples=4):
        """
        从当前模型采样多个布局 (向后兼容)
        """
        layouts = []
        for _ in range(num_samples):
            # 这里调用模型的推理过程
            with torch.no_grad():
                results = self.model.ddim_sample(
                    batch_inputs, self.model.backbone(batch_inputs["image"]),
                    self.model.preprocess_image(batch_inputs)[1],  # images_whwh
                    self.model.preprocess_image(batch_inputs)[0],  # images
                    text_features, txt_mask
                )
                layouts.append(results)

        return layouts


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