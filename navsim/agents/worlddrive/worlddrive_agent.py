from typing import Any, List, Dict, Optional, Union
from dataclasses import asdict
import lzma
import torch
import copy
import pickle
import cv2
import os

import numpy as np
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from diffusers import AutoencoderKLCogVideoX

from navsim.evaluate.pdm_score import pdm_score
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
    PDMScorerConfig,
)
from navsim.common.dataloader import MetricCacheLoader

from .blocks.lr_scheduler import WarmupCosLR
from .worlddrive_features import TrajWorldFeatureBuilder, TrajectoryTargetBuilder
from .worlddrive_generator import TrajWorldModel, TrajWorldModelConfig, randn_tensor
from .worlddrive_adapters import Adapters
from .worlddrive_planner import TrajEncoder, TrajWorldPlanner
from .worlddrive_refiner import TrajWorldRefiner


class WorldDriveAgent(AbstractAgent):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        checkpoint_path: Optional[str] = None,
        worldmodel_checkpoint_path: Optional[str] = None,
        pretrained_vae_path: Optional[str] = None,
        cache_vae_token: bool = True, 
        cache_mode: bool = False,
        lr: float = 1e-4,
        training_mode = "traj",
        with_dit=False,
        with_wm_proj=False,
        gt_version=1,
        vocab_path="",
        sim_reward_path="",
        topk=5,
        reg_loss_weight = 1,
        cls_loss_weight = 1,
        sim_loss_weight = 1,
        select_version = 1,
        visualize=False
    ):
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        self._checkpoint_path = checkpoint_path
        self._worldmodel_checkpoint_path = worldmodel_checkpoint_path
        self._lr = lr

        self.pretrained_vae_path = pretrained_vae_path
        self.cache_vae_token = cache_vae_token
        self.cache_mode = cache_mode

        if not self.cache_vae_token and not self.cache_mode:
            print("Agent running in 'no-cache' mode, runnning CogvideoX VAE")

        # worlddrive planner 
        self.gt_version = gt_version
        self.vocab_path = vocab_path
        self.traj_vocab = torch.tensor(np.load(self.vocab_path)).to(torch.float)
        self.sim_reward_path = sim_reward_path
        self.trajencoder = TrajEncoder(traj_vocab=self.traj_vocab)
        self.trajplanner = TrajWorldPlanner(with_wm_proj=with_wm_proj,
                                            gt_version=gt_version)

        # worlddrive wm
        self.cfg = TrajWorldModelConfig(image_size_width = 256,
                                        image_size_height = 512)
        self.adapters = Adapters(inchannel_size=self.cfg.vae_embed_dim*2, 
                                 hidden_size=self.cfg.vae_embed_dim*4
                                 )
        
        # freeze world model related params
        self._freeze_model()
            
        if training_mode == "wm": # world model refinement
            self.topk = topk
            self.visualize = visualize
            if self.visualize:
                self.vae_backbone = AutoencoderKLCogVideoX.from_pretrained(
                    pretrained_vae_path,
                    subfolder="vae", 
                )
                self.vae_backbone.eval()
                self.cfg = TrajWorldModelConfig(image_size_width = 512, image_size_height = 1024)
            self.trajworldmodel = TrajWorldModel(self.cfg)
            self.trajworldmodel.eval()
            for param in self.trajworldmodel.parameters():
                param.requires_grad = False
            self.trajencoder_wm = copy.deepcopy(self.trajencoder)
            self.adapters_wm = copy.deepcopy(self.adapters)
            self.trajencoder_wm.eval()
            for param in self.trajencoder_wm.parameters():
                param.requires_grad = False
            self.adapters_wm.eval()
            for param in self.adapters_wm.adapters.parameters():
                param.requires_grad = False
            self.trajplanner.eval() # freeze planner
            for param in self.trajplanner.parameters():
                param.requires_grad = False
            self.adapters.adapters.eval()
            for param in self.adapters.adapters.parameters():
                param.requires_grad = False
            self.trajencoder.eval()
            for param in self.trajencoder.parameters():
                param.requires_grad = False
            
            
            
            self.select_version = select_version
            if select_version == 1:
                self.trajrefiner = TrajWorldRefiner()

            proposal_sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
            self.simulator = PDMSimulator(proposal_sampling)
            self.train_scorer = PDMScorer(proposal_sampling)
            self.metric_cache_loader = MetricCacheLoader(Path("/mnt/gxt-share-navsim/dataset/trajworld_1024/metric_cache/metric_cache"))

            
        
        self.train_mode = training_mode
        self.reg_loss_weight = reg_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.sim_loss_weight = sim_loss_weight

    def _freeze_model(self):
        # visual adapter
        self.adapters.eval()
        for param in self.adapters.parameters():
            param.requires_grad = False
        # traj encoder
        self.trajencoder.eval()
        for param in self.trajencoder.parameters():
            param.requires_grad = False
        # trajplanner projector(see ta-dwm)
        self.trajplanner.wm_proj.eval()
        for param in self.trajplanner.wm_proj.parameters():
            param.requires_grad = False
        self.trajplanner.traj_wm_proj.eval()
        for param in self.trajplanner.traj_wm_proj.parameters():
            param.requires_grad = False
        self.trajplanner.traj_offset_encoder.eval()
        for param in self.trajplanner.traj_offset_encoder.parameters():
            param.requires_grad = False
    
    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__
    
    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=[0, 1, 2, 3])
    
    def initialize(self, from_world_model=False) -> None:
        """Inherited, see superclass."""

        if from_world_model: # training, weights from TA-DWM
            if self.train_mode == "wm": # world model refinement，需要从stage1直接读所有权重
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
                load_state_dict = {k.replace("agent.", ""): v for k, v in state_dict.items()}
            else: # planning stage1
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))["model_state_dict"] # for traj planner
                load_state_dict = dict()
                for key, val in state_dict.items():
                    adapt_key = key.replace("module.", "")
                    if "traj_encoder" in adapt_key:
                        new_adapt_key = adapt_key.replace("traj_encoder", "trajencoder")
                        load_state_dict[new_adapt_key] = val
                        if "traj_offset_encoder" in adapt_key:
                            new_adapt_key = adapt_key.replace("traj_encoder", "trajplanner")
                            load_state_dict[new_adapt_key] = val
                    if "adapters" in adapt_key:
                        load_state_dict[adapt_key] = val
                    elif "patch_embed.proj" in adapt_key:
                        new_adapt_key = adapt_key.replace("transformer.patch_embed.proj", "trajplanner.wm_proj")
                        load_state_dict[new_adapt_key] = val
                    elif "patch_embed.cond_proj" in adapt_key:
                        new_adapt_key = adapt_key.replace("transformer.patch_embed.cond_proj", "trajplanner.traj_wm_proj")
                        load_state_dict[new_adapt_key] = val
                self.from_world_model = True
            
            if self.train_mode == "wm": # world model refinement 额外再读一个freeze住的模型
                worldmodel_state_dict = torch.load(self._worldmodel_checkpoint_path, map_location=torch.device("cpu"))["model_state_dict"] # for world model
                for key, val in worldmodel_state_dict.items():
                    adapt_key = key.replace("module.", "")
                    if "traj_encoder" in adapt_key:
                        if self.train_mode == "wm": 
                            adapt_key = adapt_key.replace("traj_encoder.", "trajencoder_wm.")
                            load_state_dict[adapt_key] = val
                    if "adapters" in adapt_key:
                        if self.train_mode == "wm": 
                            adapt_key = adapt_key.replace("adapters.a", "adapters_wm.a")
                            load_state_dict[adapt_key] = val
                    if self.train_mode == "wm":
                        if "transformer" in adapt_key:
                            adapt_key = "trajworldmodel." + adapt_key
                            load_state_dict[adapt_key] = val
            for key, val in self.state_dict().items():
                if key not in load_state_dict:
                    print("{} not in the load state key".format(key))
            self.load_state_dict(load_state_dict, strict=False)
        else: # evaluation
            if torch.cuda.is_available():
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
            else:
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                    "state_dict"
                ]
            if self._worldmodel_checkpoint_path: # use TA-DWM checkpoint
                worldmodel_state_dict = torch.load(self._worldmodel_checkpoint_path, map_location=torch.device("cpu"))["model_state_dict"]
                for key, val in worldmodel_state_dict.items():
                    adapt_key = key.replace("module.", "")
                    if "transformer" in adapt_key:
                        adapt_key = "trajworldmodel." + adapt_key
                        state_dict[adapt_key] = val

            missing, unexpected = self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()}, strict=False)
            bad_missing = [k for k in missing if not k.startswith("vae_backbone.")]
            bad_unexpected = [k for k in unexpected if not k.startswith("vae_backbone.")]
            # assert not bad_missing and not bad_unexpected, (bad_missing, bad_unexpected)
    
    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling,
                                        sim_reward_dict_path=self.sim_reward_path)]
    
    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [TrajWorldFeatureBuilder(
            cache_vae_token=self.cache_vae_token,
            cache_mode=self.cache_mode,
            pretrained_vae_path=self.pretrained_vae_path
        )]
    
    def forward_wm(self, features, targets, unique_tokens):
        visual_latents = features["visual_token"]
        status_latents = features["status_feature"]

        low_visual_latents = features["low_visual_token"]
            # visual token adapter
        ori_visual_latents = self.adapters_wm(low_visual_latents)

        self.adapters.adapters.eval() # freeze bn
        visual_latents = self.adapters(visual_latents)
        traj_embed = self.trajencoder(traj_train=True)
        bs = visual_latents.shape[0]
        traj_embed = traj_embed[None].repeat(bs, 1, 1)

        self.trajplanner.eval()
        traj_score, traj_offset, cls_loss, reg_loss, traj_pos, (visual_token, ego_token), sim_reward_loss = self.trajplanner(
            visual_latents, status_latents, traj_embed, self.traj_vocab , targets
        )


        topk = 16
        _, pos_traj_index = traj_score.topk(k=topk, dim=1)
        pos_traj_index = pos_traj_index.squeeze(-1)


        select_traj = self.trajencoder.traj_vocab[pos_traj_index]
        norm_select_traj = self.trajencoder.norm_odo(select_traj)
        traj_vocab_embed = self.trajencoder.traj_vocab_encoder(norm_select_traj.reshape(bs, topk, -1))
            
        batch_index = torch.arange(bs, device=pos_traj_index.device)[:, None]
        predict_traj_offset = traj_offset[batch_index, pos_traj_index]
        traj_offset_embed = self.trajencoder.traj_offset_encoder(predict_traj_offset.reshape(bs, topk, -1))
        
        topk_traj = select_traj + predict_traj_offset.reshape(bs, topk, 8, 3)
        topk_traj_embed = traj_vocab_embed + traj_offset_embed

        if not self.training:
            loss = {}
            loss["re_refine_loss"] = 0
            loss["re_consist_loss"] = 0
            loss["loss"] = 0 + 0
            return loss
    
        tokens_rep = [tok for tok in unique_tokens for _ in range(topk)]
        metric_cache = {}
        for token in unique_tokens:
            path = self.metric_cache_loader.metric_cache_paths[token]
            with lzma.open(path, 'rb') as f:
                metric_cache[token] = pickle.load(f)
        pdms_rewards, nc_rewards, dac_rewards = self.reward_fn(topk_traj.reshape(bs*topk, 8, 3), tokens_rep, metric_cache)
        
        rewards_matrix = pdms_rewards.view(bs, topk)

        final_topk = 3
        value, indices = rewards_matrix.topk(k=final_topk, dim = 1, largest=False)
        winner_matrix = rewards_matrix[batch_index, indices]
        winner_traj = topk_traj[batch_index, indices]

        traj_winner_embed = topk_traj_embed[batch_index, indices]
        middle_topk = 10
        value, candidate_indices = rewards_matrix.topk(k=middle_topk, dim = 1, largest=True)
        rand_noise = torch.rand(bs, middle_topk, device=rewards_matrix.device)
        _, sample_cols = rand_noise.topk(k=final_topk, dim=1) 
        indices = torch.gather(candidate_indices, 1, sample_cols)

        loser_matrix = rewards_matrix[batch_index, indices]
        loser_traj = topk_traj[batch_index, indices]

        traj_loser_embed = topk_traj_embed[batch_index, indices]

        rewards_matrix = torch.cat([winner_matrix, loser_matrix], dim = 1)
        topk_traj = torch.cat([winner_traj, loser_traj], dim = 1)

        topk_traj_embed = torch.cat([traj_winner_embed, traj_loser_embed], dim = 1)


        pos_traj, future_scene_embed, refine_loss, refine_weight, constraint, constraint_nc , constraint_dac = self.trajrefiner(topk_traj, topk_traj_embed, visual_token, [rewards_matrix])

        topk = final_topk * 2
        with torch.no_grad():
            h, w = ori_visual_latents.shape[-2:]
            shape = (bs, self.cfg.predict_frames // self.cfg.temporal_compression_ratio + 2 + 1, self.cfg.vae_embed_dim, h, w)
            rand_latents = randn_tensor(shape, generator=None, device=traj_score.device, dtype=traj_score.dtype)
            multi_latents = rand_latents[:,None].repeat(1, topk, 1, 1, 1, 1)
            multi_latents = rearrange(multi_latents, "b n f c h w -> (b n) f c h w")
            multi_ori_latents = ori_visual_latents[:,None].repeat((1, topk, 1, 1, 1, 1))
            multi_ori_latents = rearrange(multi_ori_latents, "b n f c h w -> (b n) f c h w")
            neg_latents = self.world_model_refine(multi_ori_latents, topk_traj.reshape(-1, 8, 3), latents=multi_latents, no_grad=True, timesteps=1)
            neg_latents = rearrange(neg_latents, "(b n) f c h w -> b n f c h w", b = bs, n = topk)
        neg = rearrange(neg_latents[:, :, 2:], "b n f c h w -> b n c f h w")
        neg = neg.mean(dim = 3)
        future_scene_embed = future_scene_embed.permute(0, 1, 3, 2).reshape(bs, topk, self.cfg.vae_embed_dim, h, w)

        latent_consist_loss = F.mse_loss(future_scene_embed, neg)
        # latent_consist_loss = 0

        loss = {}
        loss["re_bt_loss"] = refine_loss
        loss["re_consist_loss"] = latent_consist_loss
        loss["loss"] = loss["re_consist_loss"] + loss["re_bt_loss"]
        return loss

    def forward_traj(self, features, targets):
        visual_latents = features["visual_token"]
        status_latents = features["status_feature"]

        # visual adapter
        visual_latents = self.adapters(visual_latents)
        bs = visual_latents.shape[0]
        # traj vocab encoder
        traj_embed = self.trajencoder(traj_train=True)
        traj_embed = traj_embed[None].repeat(bs, 1, 1)

        traj_score, traj_offset, cls_loss, reg_loss, traj_pos, visual_token, sim_reward_loss = self.trajplanner(
            visual_latents, 
            status_latents, 
            traj_embed, 
            self.traj_vocab, 
            targets
        )

        # loss computation
        loss = {}
        loss["cls_loss"] = cls_loss * self.cls_loss_weight
        loss["reg_loss"] = reg_loss * self.reg_loss_weight
        loss["sim_loss"] = sim_reward_loss * self.sim_loss_weight if self.trajplanner.sim_reward else 0
        loss["loss"] = loss["cls_loss"] + loss["reg_loss"] + loss["sim_loss"]
        return loss
    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None, token_list=None, eval=False) -> Dict[str, torch.Tensor]:
        if self.training and self.train_mode=="wm":
            loss = self.forward_wm(features, targets, token_list)
            return loss
        elif self.training and self.train_mode=="traj":
            loss = self.forward_traj(features, targets)
            return loss
        # --------evaluation-----------
        elif not self.training and eval: 
            visual_latents = features["visual_token"][None]
            status_latents = features["status_feature"][None]
            visual_latents = self.adapters(visual_latents)
            bs = visual_latents.shape[0]

            traj_embed = self.trajencoder(traj_train=True)
            traj_embed = traj_embed[None].repeat(bs, 1, 1)
            predict_traj, traj_score, traj_offset, (visual_token, ego_token) = self.trajplanner(
                visual_latents, status_latents, traj_embed, self.traj_vocab, targets, eval_mode=True
            )

            if self.train_mode == "wm":
                topk = self.topk
                distance_score, pos_traj_index = traj_score.topk(k=topk, dim=1)
                pos_traj_index = pos_traj_index.squeeze(-1)

                # select vocab
                select_traj = self.trajencoder_wm.traj_vocab[pos_traj_index]
                norm_select_traj = self.trajencoder_wm.norm_odo(select_traj)
                traj_vocab_embed = self.trajencoder_wm.traj_vocab_encoder(norm_select_traj.reshape(bs, topk, -1))
                
                # select offset
                batch_index = torch.arange(bs, device=pos_traj_index.device)[:, None]
                predict_traj_offset = traj_offset[batch_index, pos_traj_index]
                traj_offset_embed = self.trajencoder_wm.traj_offset_encoder(predict_traj_offset.reshape(bs, topk, -1))

                # select traj / traj embed
                topk_traj = select_traj + predict_traj_offset.reshape(bs, topk, 8, 3)
                topk_traj_embed = traj_vocab_embed + traj_offset_embed

                pos_traj, future_scene_embed, refine_loss, refine_weight = self.trajrefiner(topk_traj, 
                                                                                            topk_traj_embed, 
                                                                                            visual_token, 
                                                                                            None)
                predict_traj = pos_traj.reshape(bs, 8, 3)

                # visualization 
                if self.visualize:
                    with torch.no_grad():
                        ori_visual_latents = self.adapters(features["visual_token"][None])
                        h, w = ori_visual_latents.shape[-2:]
                        shape = (bs, self.cfg.predict_frames // self.cfg.temporal_compression_ratio + 3, self.cfg.vae_embed_dim, h, w)
                        rand_latents = randn_tensor(shape, generator=None, device=traj_score.device, dtype=traj_score.dtype)

                        latents = self.world_model_refine(ori_visual_latents, pos_traj, latents=rand_latents, timesteps=100)
                        self.scene_generate(latents, f"visual/pred_traj_{token_list[0]}", "worlddrive_far")

                        latents = self.world_model_refine(ori_visual_latents, topk_traj[:, 0], latents=rand_latents, timesteps=100)
                        self.scene_generate(latents, f"visual/pred_traj_{token_list[0]}", "worlddrive_top1")

                        latents = self.world_model_refine(ori_visual_latents, targets["trajectory"][None].to(torch.float), latents=rand_latents, timesteps=100)
                        self.scene_generate(latents, f"visual/pred_traj_{token_list[0]}", "expert")

                return {
                        "pred_traj_0": topk_traj[:, 0], # for visual
                        "pred_traj_1": topk_traj[:, 1], 
                        "pred_traj_2": topk_traj[:, 2],
                        "pred_traj_3": predict_traj, 
                }


            return {"pred_traj": predict_traj}
        
        elif not self.training and self.train_mode=="traj":
            visual_latents = features["visual_token"]
            status_latents = features["status_feature"]
            visual_latents = self.adapters(visual_latents)

            traj_embed = self.trajencoder(traj_train=True)
            bs = visual_latents.shape[0]
            traj_embed = traj_embed[None].repeat(bs, 1, 1)
            traj_score, traj_offset, cls_loss, reg_loss, traj_pos, visual_token, sim_reward_loss = self.trajplanner(
                visual_latents, status_latents, traj_embed, self.traj_vocab , targets
            )
            loss = {}
            loss["cls_loss"] = cls_loss * self.cls_loss_weight
            loss["reg_loss"] = reg_loss * 10
            loss["sim_loss"] = sim_reward_loss if self.trajplanner.sim_reward else 0
            loss["loss"] = cls_loss * self.cls_loss_weight + reg_loss * 10 + sim_reward_loss
            return loss

        elif not self.training and self.train_mode=="wm":
            loss = self.forward_wm(features, targets, token_list)
            return loss

    def scene_generate(self, latents, save_path, save_name, fps = 2):
        latents = latents.permute(0, 2, 1, 3, 4)
        cond, preds = latents[:, :, :2], latents[:, :, 2:]
        latents = 1 / self.vae_backbone.config.scaling_factor * preds
        frames = self.vae_backbone.decode(latents[:1]).sample
        frames = frames.permute(0, 2, 3, 4, 1)
        frames = torch.clamp(frames, -1, 1)
        pred = ((frames[0].cpu().detach().numpy() / 2 + 0.5) * 255).astype('uint8')[:, :, :, ::-1]

        T, H, W, _ = pred.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, f"{save_name}.mp4")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (W, H))
        
        if not writer.isOpened():
            raise RuntimeError(
                f"VideoWriter failed to open. save_path={save_path}, fps={fps}, size={(W,H)}. "
                f"Try a different codec (e.g., 'avc1') or ensure directory exists."
            )

        for t in range(T):
            frame_rgb = pred[t]
            writer.write(frame_rgb)
        writer.release()
    
    def world_model_refine(self, cond_latents, trajectory, latents, no_grad=False, guidance_scale=1, timesteps=1):
        do_classifier_free_guidance = guidance_scale > 1
        traj_embed = self.trajencoder(trajectory, traj_train=False) # bs f l c
        neg_traj_embed = self.trajencoder(torch.zeros_like(trajectory), traj_train=False)
        if do_classifier_free_guidance:
            traj_embed = torch.cat([traj_embed, neg_traj_embed], dim = 0)
        
        latents = self.trajworldmodel.step_eval(cond_latents, latents, traj_embed, no_grad=no_grad, guidance_scale=guidance_scale, timesteps=timesteps)
        return latents

    def compute_loss(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.train_mode == "wm":
            return predictions
        elif self.train_mode == "traj":
            return predictions

    def get_optimizers(self) -> Union[Optimizer, Dict[str, LRScheduler]]:
        return self.get_coslr_optimizers()
    
    def get_step_lr_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self._lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 70], gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def get_coslr_optimizers(self):
        # import ipdb; ipdb.set_trace()
        optimizer_cfg = dict(type="AdamW", 
                            lr=self._lr, 
                            weight_decay=1e-4,
                            betas=(0.9, 0.95)
                            )

        optimizer_cfg = DictConfig(optimizer_cfg)
        
        # if getattr(self, "from_world_model"):
        opt_paramwise_cfg = {
        "name":{
            "trajencoder":{
                    "lr_mult": 0.1
                },
            "adapters":{
                    "lr_mult": 0.1
                },
            "wm_proj":{
                    "lr_mult": 0.1
            },
            },    
        }

        params = []
        pgs = [[] for _ in opt_paramwise_cfg['name']]

        for k, v in self.named_parameters():
            in_param_group = True
            for i, (pattern, pg_cfg) in enumerate(opt_paramwise_cfg['name'].items()):
                if pattern in k:
                    pgs[i].append(v)
                    in_param_group = False
            if in_param_group:
                params.append(v)
        # else:
        #     params = self.parameters()
        
        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        # import ipdb; ipdb.set_trace()
        # if getattr(self, "from_world_model"):
        for pg, (_, pg_cfg) in zip(pgs, opt_paramwise_cfg['name'].items()):
            cfg = {}
            if 'lr_mult' in pg_cfg:
                cfg['lr'] = pg_cfg['lr_mult'] * optimizer_cfg["lr"]
            optimizer.add_param_group({'params': pg, **cfg})
        
        # scheduler = build_from_configs(optim.lr_scheduler, scheduler_cfg, optimizer=optimizer)
        if self.train_mode == "wm":
            wm_epochs = 0
            scheduler = WarmupCosLR(
                optimizer=optimizer,
                lr=self._lr,
                min_lr=1e-6,
                epochs=10,
                warmup_epochs=wm_epochs,
            )
        else:
            wm_epochs = 10 # 10
            scheduler = WarmupCosLR(
                optimizer=optimizer,
                lr=self._lr,
                min_lr=1e-6,
                epochs=60,
                warmup_epochs=wm_epochs,
            )

        
        
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def compute_trajectory(self, features: Dict[str, torch.Tensor], targets, tokens=None) -> Trajectory:
        self.eval()
        with torch.no_grad():
            predictions = self.forward(features, targets, tokens, eval=True)
            if "pred_traj_1" in predictions:
                pose_list = [Trajectory(predictions[f"pred_traj_{i}"].float().cpu().squeeze(0)) for i in range(len(predictions))]
                return pose_list
            else:
                poses = predictions["pred_traj"].float().cpu().squeeze(0)
                if self.train_mode == "wm":
                    if "ori_traj" not in predictions:
                        return Trajectory(poses)
                    ori_poses = predictions["ori_traj"].float().cpu().squeeze(0)
                    if "score" in predictions:
                        return [Trajectory(poses), Trajectory(ori_poses), predictions["score"]]
                    return [Trajectory(poses), Trajectory(ori_poses)]
        return Trajectory(poses)

    # def compute_trajectory(self, agent_input) -> Trajectory:
    #     self.eval()
    #     features: Dict[str, torch.Tensor] = {}
    #     # build features
    #     for builder in self.get_feature_builders():
    #         features.update(builder.compute_features(agent_input))
    #     features = {k: v.to("cuda")  for k, v in features.items()}
    #     with torch.no_grad():
    #         now = datetime.now()
    #         predictions = self.forward(features, None, None, eval=True)
    #         print("planner", datetime.now() - now)
    #         pose_list = [Trajectory(predictions[f"pred_traj_{i}"].float().cpu().squeeze(0)) for i in range(len(predictions))]
    #     return pose_list

    def reward_fn(
        self,
        pred_traj: torch.Tensor,
        tokens_list,
        cache_dict,
    ) -> torch.Tensor:
        """Calculates PDM scores for a batch of predicted trajectories."""
        pred_np = pred_traj.detach().cpu().numpy()
        pdms_rewards = []
        for i, token in enumerate(tokens_list):
            trajectory = Trajectory(pred_np[i])
            metric_cache = cache_dict[token]
            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=self.simulator.proposal_sampling,
                simulator=self.simulator,
                scorer=self.train_scorer,
            )
            pdms_rewards.append(asdict(pdm_result)["score"])
            
        return [torch.tensor(pdms_rewards, device=pred_traj.device, dtype=pred_traj.dtype).detach(),
                ]
def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)

def format_number(n, decimal_places=2):
    return f"{n:+.{decimal_places}f}" if abs(round(n, decimal_places)) > 1e-2 else "0.0"
    