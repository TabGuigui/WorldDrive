import torch
import torch.nn as nn

from timm.models.layers import Mlp
from einops import rearrange

def extract_yaw_from_rotation_matrices(rot_matrices):
    return torch.atan2(rot_matrices[..., 1, 0], rot_matrices[..., 0, 0])

def get_rel_traj_radians(poses, condition_frames, traj_len):
    B, F, N = poses.shape[0], condition_frames, traj_len
    rot_A = poses[:, :F, None][..., :3, :3]
    rot_B = poses[:, None, :F+N][..., :3, :3]
    trans_A = poses[:, :F, None][..., :3, 3:]
    trans_B = poses[:, None, :F+N][..., :3, 3:]
    rel_pose_rot = torch.linalg.inv(rot_A) @ rot_B
    rel_pose_trans = torch.linalg.inv(rot_A) @ (trans_B - trans_A)
    indices = torch.arange(F).unsqueeze(1) + torch.arange(1, N+1).unsqueeze(0)  # [F, N]
    rel_pose_trans = rel_pose_trans[:, torch.arange(F).unsqueeze(1), indices]
    rel_poses = rel_pose_trans[..., :2, 0]
    rel_pose_rot = rel_pose_rot[:, torch.arange(F).unsqueeze(1), indices]
    rel_yaws = extract_yaw_from_rotation_matrices(rel_pose_rot[..., :3, :3]).unsqueeze(-1)
    return rel_poses, rel_yaws

class TrajEncoder(nn.Module):
    def __init__(self,
                 traj_vocab_size=128,
                 traj_vocab=None,
                 traj_vocab_dim=512,
                 traj_offset_dim=512,
                 traj_len=15,
                 traj_dim=3,
                 hidden_size=1024,
                 condition_frames=4,
                 topk=5
                 ):
        """
        traj encoder with traj vocab trajecotry and traj offset
        """
        super().__init__()

        self.traj_vocab_size = traj_vocab_size
        self.traj_vocab = torch.tensor(traj_vocab, device = "cuda").to(torch.bfloat16)
        assert self.traj_vocab.shape[0] == self.traj_vocab_size

        self.norm_traj_vocab = self.norm_odo(self.traj_vocab)

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
        self.condition_frames = condition_frames

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
        distance = torch.sqrt((diffs ** 2).sum(dim=-1))[..., -1]
        trajectory_id = torch.argsort(distance, dim = 1)[:, :self.topk] # bs topk
        base_trajectory = self.traj_vocab[trajectory_id]
        traj_offset = trajectory[:, None] - base_trajectory
        return trajectory_id, traj_offset
    
    def forward(self, trajectory: torch.Tensor, traj_train: bool=False, is_traj=False):
        """
        traj_train: use planner or worldmodel mode, if traj_train==True, select the planner mode
        """
        batch_size = trajectory.shape[0]
        if not is_traj:
            traj_poses, traj_yaws = get_rel_traj_radians(trajectory, self.condition_frames, self.traj_len)
            traj_targets = torch.cat([traj_poses, traj_yaws], dim=-1)   # (B, F, N, 3)
            trajectory = traj_targets[:, -1]
        else:
            trajectory = trajectory

        traj_vocab_id, traj_vocab_offset = self.get_traj_indices_offset(trajectory)
        traj_vocab_id = traj_vocab_id[..., None]
        select_traj = self.norm_traj_vocab[traj_vocab_id]
        traj_vocab_embed = self.traj_vocab_encoder(select_traj.reshape(batch_size, self.topk, -1))
        traj_vocab_offset = traj_vocab_offset.reshape(batch_size, self.topk, -1)
        traj_offset_embed = self.traj_offset_encoder(traj_vocab_offset) # bs 1 1024
        traj_embed = traj_vocab_embed + traj_offset_embed

        return traj_embed

class Adapters(nn.Module):
    def __init__(self,
                 inchannel_size=512,
                 hidden_size=1024,
                 ):
        """
        traj encoder with traj vocab trajecotry and traj offset
        """
        super().__init__()

        self.adapters = Mlp(
            in_features=inchannel_size,
            hidden_features=hidden_size,
            out_features=inchannel_size,
            norm_layer=nn.BatchNorm2d,
            use_conv=True
        )
    
    def forward(self, visual_latents):
        bs, f, c, h, w = visual_latents.shape
        visual_latents = rearrange(visual_latents, "b f c h w -> b (f c) h w")
        visual_latents = self.adapters(visual_latents)
        visual_latents = rearrange(visual_latents, "b (f c) h w -> b f c h w", f = f, c = c, h = h, w = w)
        return visual_latents