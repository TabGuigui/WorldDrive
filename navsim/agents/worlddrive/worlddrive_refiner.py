import torch.nn as nn
import torch.nn.functional as F
import torch
from timm.models.layers import Mlp
from diffusers.models.embeddings import get_2d_sincos_pos_embed

class TrajWorldRefiner(nn.Module):
    def __init__(self,
                 traj_vocab_dim=512,
                 traj_len=8,
                 traj_dim=3,
                 hidden_size=1024,
                 h=32, 
                 w=64,
                 visual_dim=512
                 ):
        super().__init__()

        self.future_scene_query = nn.Embedding(h*w, visual_dim)
        self.traj_len = traj_len
        self.traj_dim = traj_dim
        self.traj_vocab_dim = traj_vocab_dim
        self.traj_offset_dim = traj_vocab_dim
        self.visual_dim = visual_dim
        self.traj_encoder = Mlp(
            in_features=self.traj_len * self.traj_dim,
            hidden_features=hidden_size,
            out_features=traj_vocab_dim,
            norm_layer=nn.LayerNorm
        )


        decoder_layer = nn.TransformerDecoderLayer(
                    d_model=traj_vocab_dim,
                    nhead=8,
                    dim_feedforward=traj_vocab_dim*2,
                    dropout=0.3,
                    batch_first=True,
                )
        self.refine_decoder = nn.TransformerDecoder(decoder_layer, 4)

        traj_decoder_layer = nn.TransformerDecoderLayer(
                    d_model=traj_vocab_dim,
                    nhead=8,
                    dim_feedforward=traj_vocab_dim*2,
                    dropout=0.0,
                    batch_first=True,
                )
        # self.traj_refine_decoder = nn.TransformerDecoder(traj_decoder_layer, 2)
        self.traj_refine_decoder = nn.TransformerDecoder(traj_decoder_layer, 4)

        self.refine_reward_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(traj_vocab_dim, traj_vocab_dim//4),
                    nn.ReLU(),
                    nn.Linear(traj_vocab_dim//4, 1),
                ) for _ in range(3)
            ])
        
        self.visual_proj = nn.Linear(traj_vocab_dim, 16)

        self.reward_loss =PairwiseRankingLoss(diff_threshold=0.0005)
        self.apply(self._init_weights)
        
        self.h, self.w = h, w
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _get_positional_embeddings(
        self, height: int, width: int, device
    ) -> torch.Tensor:
        post_patch_height = height # 最终的h
        post_patch_width = width
        
        pos_embedding = get_2d_sincos_pos_embed(
            self.traj_vocab_dim,
            (post_patch_width, post_patch_height),
            device=device,
            output_type="pt",
        )
        pos_embedding = pos_embedding

        return pos_embedding

    def forward(self, topk_traj, topk_traj_embed, visual_token, rewards_target):
        bs, traj_num = topk_traj.shape[:2] 
        # visual_token: bs c h w
        bs, visual_num = visual_token.shape[:2]
        visual_token = visual_token[:, None].repeat(1, traj_num, 1, 1)
        visual_token = visual_token.reshape(bs*traj_num, visual_num, -1)
        # future_scene_query 512 16
        future_query = self.future_scene_query.weight[None, None, ...].repeat(bs, traj_num, 1, 1) # bs 5 16*32

        traj_pos_token = topk_traj_embed
        traj_pos_token = traj_pos_token[:, :, None] # token dim
        traj_pos_token = traj_pos_token.reshape(bs * traj_num, 1, -1)
        
        future_query = future_query.reshape(bs * traj_num, self.h * self.w, self.visual_dim)
        pos_embedding = self._get_positional_embeddings(self.h, self.w, device=visual_token.device)
        future_query = future_query + pos_embedding[None, :].to(visual_token.dtype)

        future_scene_embed = self.refine_decoder(future_query, 
                                             torch.cat([visual_token, traj_pos_token], dim = 1)
                                             ) # 未来场景latent
        
        traj_pos_embed = self.traj_refine_decoder(traj_pos_token, future_scene_embed) # future scene aware
        traj_pos_embed = traj_pos_embed.reshape(bs, traj_num, -1) + future_scene_embed.mean(dim=1).reshape(bs, traj_num, -1)
        # refine im/sim score
        dis_weight, nc_reward, dac_reward = [sim_reward_head(traj_pos_embed) for sim_reward_head in self.refine_reward_heads]
        final_weight = dis_weight

        predict_traj = self.select_best_trajectory(final_weight.squeeze(-1), topk_traj, bs)

        future_scene_embed = self.visual_proj(future_scene_embed)
        if not self.training:
            return predict_traj, future_scene_embed.reshape(bs, traj_num, visual_num, -1), 0, final_weight
        
        bs, n = final_weight.squeeze(-1).shape
        pair_weights = torch.ones(bs, n, n, device=final_weight.device)
        is_unsafe = torch.logical_or(rewards_target[0] < 0.6, rewards_target[0] > 0.98).float() 
        is_unsafe_matrix = is_unsafe.unsqueeze(1).repeat(1, n, 1)
        penalty_weight = 5.0
        pair_weights = pair_weights + is_unsafe_matrix * (penalty_weight - 1.0)

        loss_dis = self.reward_loss(dis_weight, rewards_target[0].detach(), pair_weights)
        constraint = 0

        return predict_traj, future_scene_embed.reshape(bs, traj_num, visual_num, -1), loss_dis, final_weight, constraint
    
    def select_best_trajectory(self, final_rewards, trajectory_anchors, batch_size):
        best_trajectory_idx = torch.argmax(final_rewards, dim=-1)[:, None]  # Shape: [batch_size]
        batch_index = torch.arange(batch_size, device=best_trajectory_idx.device)[:, None]
        poses = trajectory_anchors[batch_index, best_trajectory_idx]  # Shape: [batch_size, 24]
        poses = poses.squeeze(1)  # Reshape to [batch_size, 8, 3]
        return poses

class PairwiseRankingLoss(nn.Module):
    def __init__(self, diff_threshold=0.05):
        """
        diff_threshold: 只有当真实 Reward 的差值超过这个阈值时，才计算 Loss。
                        防止模型强行去拟合那些质量非常接近的轨迹（噪声）。
        """
        super().__init__()
        self.diff_threshold = diff_threshold

    def forward(self, pred_scores, gt_rewards, pair_weights=None):
        """
        输入:
        pred_scores: Reward Model 预测的分数, shape [Batch, N, 1] 或 [Batch, N]
        gt_rewards:  Simulator 算出的真实分数, shape [Batch, N]
        
        注意：N=5 在你的场景里
        """
        # 确保维度一致
        if pred_scores.dim() == 3:
            pred_scores = pred_scores.squeeze(-1) # [B, N]
            
        B, N = pred_scores.shape
        
        # 1. 构建预测分数的差值矩阵 (Broadcasting)
        # matrix[b, i, j] = score[b, i] - score[b, j]
        # shape: [B, N, N]
        pred_diff = pred_scores.unsqueeze(2) - pred_scores.unsqueeze(1)
        
        # 2. 构建真实 Reward 的差值矩阵
        # shape: [B, N, N]
        gt_diff = gt_rewards.unsqueeze(2) - gt_rewards.unsqueeze(1)
        
        # 3. 生成 Mask：找出哪一对 (i, j) 确实是 i 比 j 好
        # 只有当 gt_diff > threshold 时，我们才认为 i 是 winner，j 是 loser
        # 这会自动处理掉对角线 (i=i) 和反向对 (j>i)
        valid_pair_mask = (gt_diff > self.diff_threshold).float()
        
        # 4. 计算 Log-Sigmoid Loss
        # 公式: -log(sigmoid(s_i - s_j))
        # 也就是 -log_sigmoid(pred_diff)
        loss_matrix = -F.logsigmoid(pred_diff)

        if pair_weights is not None:
            # pair_weights 形状必须是 [B, N, N]
            # 我们只关心 valid_pair_mask 为 1 的那些位置的权重
            loss_matrix = loss_matrix * pair_weights
        
        # 5. 应用 Mask 并求平均
        # 只保留 valid_pair_mask 为 1 的位置的 loss
        masked_loss = loss_matrix * valid_pair_mask
        
        # 计算平均值 (除以有效配对的数量)
        # 加一个极小值 eps 防止除以 0
        num_valid_pairs = valid_pair_mask.sum() + 1e-8
        total_loss = masked_loss.sum() / num_valid_pairs
        
        return total_loss

