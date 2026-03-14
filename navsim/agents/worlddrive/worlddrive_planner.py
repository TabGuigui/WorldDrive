from dataclasses import asdict, dataclass, field
from einops import rearrange

import torch
from torch import nn

from timm.models.layers import Mlp
from diffusers.models.embeddings import get_3d_sincos_pos_embed


SCORE_HEAD_HIDDEN_DIM = 128
SCORE_HEAD_OUTPUT_DIM = 1
NUM_SCORE_HEADS = 5

class TrajWorldPlanner(nn.Module):
    def __init__(self,
                 traj_vocab_size=256,
                 traj_vocab_dim=512,
                 topk=1,
                 hidden_size=1024,
                 decoder_head=8,
                 decoder_dropout=0.3,
                 num_dec_layer=8, #4
                 traj_len=8,
                 traj_dim=3,
                 visual_embd=16,
                 patch_size=2,
                 with_wm_proj=True,
                 gt_version=1,
                 with_sim_reward=True
                 ):
        super().__init__()
        self.traj_vocab_size = traj_vocab_size
        self.traj_vocab_dim = traj_vocab_dim
        self.traj_len = traj_len
        self.traj_dim = traj_dim 
        self.topk = topk
        self.gt_version = gt_version    

        self.ego_status_encoder = nn.Linear(8, traj_vocab_dim)
        
        # inherit from TA-DWM, wm_proj & traj_wm_proj
        self.wm_feature_dim = 1920
        self.with_wm_proj = with_wm_proj
        self.wm_proj = nn.Conv2d(
            visual_embd*2, self.wm_feature_dim, kernel_size=(patch_size, patch_size), stride=patch_size
        )
        self.visual_projector = Mlp(
            in_features=self.wm_feature_dim,
            hidden_features=self.wm_feature_dim//2,
            out_features=traj_vocab_dim,
            norm_layer=nn.BatchNorm2d,
            drop=decoder_dropout,
            use_conv=True
        )

        self.traj_wm_proj = nn.Linear(traj_vocab_dim, self.wm_feature_dim)
        self.traj_projector = Mlp(
            in_features=self.wm_feature_dim,
            hidden_features=self.wm_feature_dim//2,
            out_features=traj_vocab_dim,
            norm_layer=nn.LayerNorm,
            drop=0.0,
            use_conv=False
        )
        
        # traj & visual fusion
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=traj_vocab_dim,
            nhead=decoder_head,
            dim_feedforward=hidden_size,
            dropout=decoder_dropout,
            batch_first=True,
        )
        self.traj_visual_decoder = nn.TransformerDecoder(decoder_layer, num_dec_layer)


        self.cls_branch = nn.Sequential(
            nn.Linear(traj_vocab_dim, traj_vocab_dim//8),
            nn.ReLU(inplace=True),
            nn.Linear(traj_vocab_dim//8, 1)
        )
        self.offset_branch = nn.Sequential(
            nn.Linear(traj_vocab_dim, traj_vocab_dim//4),
            nn.ReLU(),
            nn.Linear(traj_vocab_dim//4, traj_len * traj_dim),
        )
        self.traj_offset_encoder = Mlp(
            in_features=self.traj_len * self.traj_dim,
            hidden_features=traj_vocab_dim*2,
            out_features=traj_vocab_dim,
            norm_layer=nn.LayerNorm
        )

        self.reg_loss = nn.L1Loss(reduction="none")
        self.sim_reward = with_sim_reward
        if self.sim_reward:
            self.sim_reward_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(traj_vocab_dim, SCORE_HEAD_HIDDEN_DIM),
                    nn.ReLU(),
                    nn.Linear(SCORE_HEAD_HIDDEN_DIM, SCORE_HEAD_OUTPUT_DIM),
                ) for _ in range(NUM_SCORE_HEADS)
            ])
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _get_traj_gt_v3(self, traj_vocab, traj_targets):
        assert traj_targets.shape[-1] == traj_vocab.shape[-1]
        bs = traj_targets.shape[0]
        num_trajs = traj_vocab.shape[0]
        trajectory_anchors = traj_vocab.reshape(num_trajs, -1).unsqueeze(0).repeat(bs, 1, 1)
        trajectory_targets = traj_targets.reshape(bs, -1).unsqueeze(1).float()
        l2_distances = torch.cdist(trajectory_anchors, trajectory_targets, p=2)  # Shape: [batch_size, 256]
        l2_distances = l2_distances.squeeze(-1)
        # Apply softmax to L2 distances to get reward targets
        reward_targets = torch.softmax(-l2_distances, dim=-1)  # Shape: [batch_size, 256]

        positive_id = torch.argsort(l2_distances, dim = 1)[:, :1]# pos id
        base_trajectory = traj_vocab[positive_id] 
        positive_offset = traj_targets[:, None] - base_trajectory        
        return reward_targets, positive_id, positive_offset
    
    def _get_positional_embeddings(
        self, height: int, width: int, frames: int, device
    ) -> torch.Tensor:
        post_patch_height = height # 最终的h
        post_patch_width = width
        post_time_compression_frames = frames #预测的帧数
        
        pos_embedding = get_3d_sincos_pos_embed(
            self.traj_vocab_dim,
            (post_patch_width, post_patch_height),
            post_time_compression_frames,
            device=device,
            output_type="pt",
        )
        pos_embedding = pos_embedding.flatten(0, 1)

        return pos_embedding
    

    def forward(self, visual_latent, ego_status, traj_vocab_feature, traj_vocab, targets=None, eval_mode=False):

        traj_vocab = traj_vocab.to(visual_latent.device)
        height, width = visual_latent.shape[-2:]
        # with wm proj
        visual_latent = self.wm_proj(torch.cat([visual_latent[:, -1], visual_latent[:, -1]], dim = 1)) # refer to TA-DiT
        visual_token = self.visual_projector(visual_latent)
        visual_token = rearrange(visual_token, "b c h w -> b (h w) c") # bf16
        pos_embedding = self._get_positional_embeddings(
                    height // 2, width // 2, 1, device=visual_token.device
                )
        # -------- token compute ---------
        visual_token = visual_token + pos_embedding[None, :].to(dtype=visual_token.dtype) # bs hxw 512
        ego_token = self.ego_status_encoder(ego_status.to(visual_latent.device))[:, None] # bs 1 512 

        traj_vocab_feature = self.traj_wm_proj(traj_vocab_feature)
        traj_vocab_feature = self.traj_projector(traj_vocab_feature) # for planner
        traj_vocab_feature = self.traj_visual_decoder(traj_vocab_feature, torch.cat([visual_token, ego_token], dim = 1))

        bs = ego_token.shape[0]
        traj_offset = self.offset_branch(traj_vocab_feature[:, :self.traj_vocab_size]) # offset predict
        traj_offset_feature = self.traj_offset_encoder(traj_offset)
        traj_score = self.cls_branch(traj_vocab_feature+traj_offset_feature)

        if self.sim_reward:
            sim_rewards = [sim_reward_head(traj_vocab_feature) for sim_reward_head in self.sim_reward_heads]
            if not eval_mode:
                sim_rewards_sigmoid = torch.cat(sim_rewards, dim=-1).permute(0, 2, 1).sigmoid()  # Concatenate metric rewards
                if "sim_reward" in targets:
                    sim_rewards_loss = compute_sim_reward_loss(targets["sim_reward"], sim_rewards_sigmoid)
                else:
                    sim_rewards_loss = 0

        if eval_mode:
            im_rewards_softmax = torch.softmax(traj_score.squeeze(-1), dim=-1)
            sim_rewards = [reward.sigmoid() for reward in sim_rewards]  
            final_rewards = self.weighted_reward_calculation(im_rewards_softmax, sim_rewards)
            offset = traj_offset.squeeze(0)
            trajectory_anchors = traj_vocab + offset.reshape(-1, self.traj_len, self.traj_dim)
            bs = im_rewards_softmax.shape[0]
            predict_traj = self.select_best_trajectory(final_rewards, trajectory_anchors, bs)

            return predict_traj, final_rewards, traj_offset, (visual_token,ego_token)
        # -------- loss compute ---------
        traj_targets = targets["trajectory"]
        # cls loss
        traj_pos, pos_id, pos_offset = self._get_traj_gt_v3(traj_vocab,  traj_targets)
        cls_loss = -torch.sum(traj_pos *  torch.softmax(traj_score.squeeze(-1), dim = -1).log()) / bs
        # reg loss
        pos_traj_offsets = []
        for b in range(bs):
            pos_index = pos_id[b]
            pos_traj_offsets.append(traj_offset[b][pos_index][None])
        pos_traj_offsets = torch.cat(pos_traj_offsets, dim = 0)
        traj_offset_gt = pos_offset
        pos_traj_offsets = pos_traj_offsets.reshape(bs, pos_id.shape[1], 8, 3)
        reg_loss = self.reg_loss(pos_traj_offsets, traj_offset_gt)
        reg_loss = reg_loss.mean()

        if not self.training: # eval mode for val loss
            predict_traj_index = torch.argmax(traj_score, dim =1).squeeze(1)
            batch_idx = torch.arange(bs, dtype=torch.int)
            predict_traj_mode = traj_vocab[predict_traj_index.cpu()]
            predict_traj_offset = traj_offset[batch_idx, predict_traj_index.cpu()].reshape(predict_traj_mode.shape)
            predict_traj = predict_traj_mode + predict_traj_offset
            reg_loss = self.reg_loss(predict_traj, traj_targets)
            reg_loss = reg_loss.mean()
        if self.sim_reward:
            im_rewards_softmax = torch.softmax(traj_score.squeeze(-1), dim=-1)
            sim_rewards = [reward.sigmoid() for reward in sim_rewards]  
            final_rewards = self.weighted_reward_calculation(im_rewards_softmax, sim_rewards)
        else:
            final_rewards = torch.softmax(traj_score.squeeze(-1), dim=-1)
        return final_rewards, traj_offset, cls_loss, reg_loss, traj_pos, (visual_token, ego_token), sim_rewards_loss if self.sim_reward else 0
    
    def weighted_reward_calculation(self, im_rewards, sim_rewards) -> torch.Tensor:
        """
        Calculate the final reward for each trajectory based on the given weights.

        Args:
            im_rewards (torch.Tensor): Imitation rewards for each trajectory. Shape: [batch_size, num_traj]
            sim_rewards (List[torch.Tensor]): List of metric rewards for each trajectory. Each tensor shape: [batch_size, num_traj]
            w (List[float]): List of weights for combining the rewards.

        Returns:
            torch.Tensor: Final weighted reward for each trajectory. Shape: [batch_size, num_traj]
        """
        assert len(sim_rewards) == 5, "Expected 4 metric rewards: S_NC, S_DAC, S_TTC, S_EP, S_COMFORT"
        # Extract metric rewards
        w = [0.1, 0.5, 0.5, 1.0]  
        S_NC, S_DAC, S_EP, S_TTC, S_COMFORT = sim_rewards
        S_NC, S_DAC, S_EP, S_TTC, S_COMFORT = S_NC.squeeze(-1), S_DAC.squeeze(-1), S_EP.squeeze(-1), S_TTC.squeeze(-1), S_COMFORT.squeeze(-1)
        #self.metric_keys = ['no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress', 'time_to_collision_within_bound', 'comfort']
        # Calculate assembled cost based on the provided formula
        assembled_cost = (
            w[0] * torch.log(im_rewards)+
            w[1] * torch.log(S_NC) +
            w[2] * torch.log(S_DAC) +
            w[3] * torch.log(5 * S_TTC + 2 * S_COMFORT + 5 * S_EP)
        )
        return assembled_cost
    
    def select_best_trajectory(self, final_rewards, trajectory_anchors, batch_size):
        best_trajectory_idx = torch.argmax(final_rewards, dim=-1)  # Shape: [batch_size]
        poses = trajectory_anchors[best_trajectory_idx]  # Shape: [batch_size, 24]
        poses = poses.view(batch_size, 8, 3)  # Reshape to [batch_size, 8, 3]
        return poses

def compute_sim_reward_loss(
    targets,
    predicted_rewards: torch.Tensor,
) -> torch.Tensor:
    epsilon = 1e-6
    # Load precomputed target rewards
    batch_size = targets.shape[0]
    target_rewards = targets[:, -1] # the last frame

    # Compute loss using binary cross-entropy # 5 is the number of metrics
    sim_reward_loss = -torch.mean(
        target_rewards * (predicted_rewards + epsilon).log() + (1 - target_rewards) * (1 - predicted_rewards + epsilon).log()
    ) * 5

    return sim_reward_loss


class TrajEncoder(nn.Module):
    def __init__(self,
                 traj_vocab_size=256,
                 traj_vocab=None,
                 traj_vocab_dim=512,
                 traj_offset_dim=512,
                 traj_len=8,
                 traj_dim=3,
                 hidden_size=1024,
                 condition_frames=5,
                 topk=5
                 ):
        """
        traj encoder with traj vocab trajecotry and traj offset
        """
        super().__init__()

        self.traj_vocab_size = traj_vocab_size
        self.traj_vocab = nn.Parameter(
            torch.tensor(traj_vocab, dtype=torch.float32),
            requires_grad=False,
        )
        assert self.traj_vocab.shape[0] == self.traj_vocab_size

        self.traj_len = traj_len
        self.traj_dim = traj_dim # default 3 for NAVSIM
        self.hidden_size = hidden_size
        self.traj_vocab_dim = traj_vocab_dim
        self.traj_offset_dim = traj_offset_dim

        self.traj_vocab_encoder = Mlp(
            in_features=self.traj_len * self.traj_dim,
            hidden_features=hidden_size,
            out_features=self.traj_vocab_dim,
            norm_layer=nn.LayerNorm
        )

        self.traj_offset_encoder = Mlp(
            in_features=self.traj_len * self.traj_dim,
            hidden_features=hidden_size,
            out_features=self.traj_offset_dim,
            norm_layer=nn.LayerNorm
        )

        self.topk = topk

    def norm_odo(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Normalizes trajectory coordinates and heading to the range [-1, 1]."""
        x = 2 * (trajectory[..., 0:1] + 1.57) / 66.74 - 1
        y = 2 * (trajectory[..., 1:2] + 19.68) / 42 - 1
        heading = 2 * (trajectory[..., 2:3] + 1.67) / 3.53 - 1
        return torch.cat([x, y, heading], dim=-1)
    
    def denorm_odo(self, normalized_trajectory: torch.Tensor) -> torch.Tensor:
        """Denormalizes trajectory from [-1, 1] back to original coordinate space."""
        x = (normalized_trajectory[..., 0:1] + 1) / 2 * 66.74 - 1.57
        y = (normalized_trajectory[..., 1:2] + 1) / 2 * 42 - 19.68
        heading = (normalized_trajectory[..., 2:3] + 1) / 2 * 3.53 - 1.67
        return torch.cat([x, y, heading], dim=-1)

    def get_traj_indices_offset(self, trajectory):
        assert trajectory.shape[-1] == self.traj_vocab.shape[-1]
        diffs = trajectory[:, None] - self.traj_vocab[None, ...]
        distance = torch.sum(torch.sqrt((diffs[..., :2] ** 2).sum(dim=-1)), dim = -1)

        trajectory_id = torch.argsort(distance, dim = 1)[:, :self.topk] # bs topk
        base_trajectory = self.traj_vocab[trajectory_id]
        traj_offset = trajectory[:, None] - base_trajectory
        return trajectory_id, traj_offset
    
    def forward(self, trajectory: torch.Tensor=None, traj_train: bool=True):
        """
        traj_train: use planner or worldmodel mode, if traj_train==True, select the planner mode
        """
        if traj_train:
            all_traj = self.norm_odo(self.traj_vocab)
            all_traj_vocab_embed = self.traj_vocab_encoder(
                all_traj.reshape(self.traj_vocab_size, -1)
            )
            traj_embed = all_traj_vocab_embed
        else:
            batch_size = trajectory.shape[0]
            traj_vocab_id, traj_vocab_offset = self.get_traj_indices_offset(trajectory)
            traj_vocab_id = traj_vocab_id[..., None]

            all_traj = self.norm_odo(self.traj_vocab)
            select_traj = all_traj[traj_vocab_id]
            traj_vocab_embed = self.traj_vocab_encoder(select_traj.reshape(batch_size, self.topk, -1))
            traj_offset_embed = self.traj_offset_encoder(traj_vocab_offset.reshape(batch_size, self.topk, -1)) # bs 1 1024

            traj_embed = traj_vocab_embed + traj_offset_embed

        return traj_embed



