from typing import Dict, Optional
import cv2
import numpy as np
import numpy.typing as npt

import torch
from torchvision import transforms

from navsim.agents.abstract_agent import AgentInput
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.common.dataclasses import Scene, Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from diffusers import AutoencoderKLCogVideoX
from datetime import datetime

class TrajWorldFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self,
                 cache_vae_token: bool = True,
                 pretrained_vae_path: Optional[str] = None,
                 device: str = "cuda",
                 cache_mode: bool = False, ):
        super().__init__()

        self.cache_vae_token = cache_vae_token
        self.cache_mode = cache_mode
        if self.cache_vae_token and self.cache_mode:
            self.vae_backbone = AutoencoderKLCogVideoX.from_pretrained(
                pretrained_vae_path,
                subfolder="vae", 
            )
            self.vae_backbone.to(device)
            self.vae_backbone.eval()
    
    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "trajworld_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:

        features = {}
        # ego status feature
        ego_statuses = agent_input.ego_statuses
        status_feature = torch.cat([
            torch.tensor(ego_statuses[-1].driving_command, dtype=torch.float32),
            torch.tensor(ego_statuses[-1].ego_velocity, dtype=torch.float32),
            torch.tensor(ego_statuses[-1].ego_acceleration, dtype=torch.float32)
        ], dim=-1)
        features["status_feature"] = status_feature

        images, camera_feature, low_camera_feature = self._get_camera_feature(agent_input)
        features["visual_token"] = camera_feature.cpu().detach()
        features["low_visual_token"] = low_camera_feature.cpu().detach()

        return features
    
    def normalize_imgs(self, imgs):
        imgs = imgs / 255.0
        imgs = (imgs - 0.5)*2
        return imgs
        
    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """
        # history 0 - 3
        prev_images = []
        low_res_prev_images = []
        for i in range(4):
            cameras = agent_input.cameras[i]
            f0 = cameras.cam_f0.image[28:-28]
            resized_image = cv2.resize(f0, (1024, 512))
            low_res_resized_image = cv2.resize(f0, (512, 256))
            prev_images.append(resized_image)
            low_res_prev_images.append(low_res_resized_image)

        prev_images = np.array(prev_images)
        tensor_image = torch.tensor(prev_images)

        if tensor_image.shape[0] == 4:
            padding_image = torch.zeros_like(tensor_image)
            tensor_image = torch.cat([padding_image, tensor_image], dim = 0) # padding 0 image to 8 frame
        tensor_image = self.normalize_imgs(tensor_image).cuda() 
        tensor_image = tensor_image[None].permute(0, 4, 1, 2, 3) # b c f h w
        with torch.no_grad():
            latents = self.vae_backbone.encode(tensor_image).latent_dist
            visual_token = latents.sample() * self.vae_backbone.config.scaling_factor
        visual_token = visual_token.permute(0, 2, 1, 3, 4) # b f c h w
        visual_token = visual_token[0]

        low_res_prev_images = np.array(low_res_prev_images)
        low_res_tensor_image = torch.tensor(low_res_prev_images)

        if low_res_tensor_image.shape[0] == 4:
            padding_image = torch.zeros_like(low_res_tensor_image)
            low_res_tensor_image = torch.cat([padding_image, low_res_tensor_image], dim = 0) # padding 0 image to 8 frame
        low_res_tensor_image = self.normalize_imgs(low_res_tensor_image).cuda() 
        low_res_tensor_image = low_res_tensor_image[None].permute(0, 4, 1, 2, 3) # b c f h w
        with torch.no_grad():
            latents = self.vae_backbone.encode(low_res_tensor_image).latent_dist
            low_res_visual_token = latents.sample() * self.vae_backbone.config.scaling_factor
        low_res_visual_token = low_res_visual_token.permute(0, 2, 1, 3, 4) # b f c h w
        low_res_visual_token = low_res_visual_token[0]
        return tensor_image[0], visual_token, low_res_visual_token
    

class TrajectoryTargetBuilder(AbstractTargetBuilder):
    def __init__(self, 
                 trajectory_sampling: TrajectorySampling,
                 slice_indices=[3],
                 sim_reward_dict_path=None,):
        self._trajectory_sampling = trajectory_sampling
        self.slice_indices = slice_indices
        self.sim_reward_dict_path = sim_reward_dict_path
        if self.sim_reward_dict_path is not None:
            self.sim_reward_dict = np.load(self.sim_reward_dict_path, allow_pickle=True).item() 
            self.sim_keys = ['no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress', 'time_to_collision_within_bound', 'comfort']

    def get_unique_name(self) -> str:
        # return "trajectory_target"
        return "trajectory_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)

        index = self.slice_indices[0]
        frame_offset = index - 3 # max is 3
        
        sim_reward_list = []
        if self.sim_reward_dict_path is not None:
            token = scene.frames[index].token
            try:
                sim_reward_dict_single = self.sim_reward_dict[token]['trajectory_scores'][0]
            except:
                return {"trajectory": torch.tensor(future_trajectory.poses)}
            # dict_keys(['no_at_fault_collisions', 'drivable_area_compliance', 'driving_direction_compliance', 'ego_progress', 'time_to_collision_within_bound', 'comfort', 'score'])
            combined_sim_reward = np.vstack([sim_reward_dict_single[key] for key in self.sim_keys])
            combined_sim_reward_tensor = torch.tensor(combined_sim_reward, dtype=torch.float32)
            sim_reward_list.append(combined_sim_reward_tensor)
            sim_reward_stacked = torch.stack(sim_reward_list) 
        
            return {"trajectory": torch.tensor(future_trajectory.poses), "sim_reward": sim_reward_stacked}