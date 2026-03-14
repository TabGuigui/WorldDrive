from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.planning.training.dataset import Dataset


from typing import Dict, Optional, Tuple
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

import torch
import cv2
import numpy as np
import logging
import random

logger = logging.getLogger(__name__)

class TrajWorldFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self,
                 cache_mode: bool = False, ):
        super().__init__()
        self.cache_mode = cache_mode

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "trajworld_feature"

    def compute_features(self, agent_input) -> Dict[str, torch.Tensor]:

        features = {}
        # ego status feature
        ego_statuses = agent_input.ego_statuses
        status_feature = torch.cat([
            torch.tensor(ego_statuses[-1].driving_command, dtype=torch.float32),
            torch.tensor(ego_statuses[-1].ego_velocity, dtype=torch.float32),
            torch.tensor(ego_statuses[-1].ego_acceleration, dtype=torch.float32)
        ], dim=-1)
        features["status_feature"] = status_feature

        images = self._get_camera_feature(agent_input)
        features["images"] = images.cpu()

        return features
        
    def _get_camera_feature(self, agent_input) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """
        # history 0 - 3
        prev_images = []
        for i in range(13):
            cameras = agent_input.cameras[i]
            f0 = cameras.cam_f0.image[28:-28]
            resized_image = cv2.resize(f0, (1024, 512))
            prev_images.append(resized_image)

        prev_images = np.array(prev_images)
        tensor_image = torch.tensor(prev_images)

        return tensor_image
    
class TrajectoryTargetBuilder(AbstractTargetBuilder):
    def __init__(self, trajectory_sampling: TrajectorySampling):
        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        return "trajectory_target"

    def compute_targets(self, scene) -> Dict[str, torch.Tensor]:
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)
        return {"trajectory": torch.tensor(future_trajectory.poses)}
    

class NavSimDataset(Dataset):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Get features or targets either from cache or computed on-the-fly.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """

        token = self._scene_loader.tokens[idx]
        features: Dict[str, torch.Tensor] = {}
        targets: Dict[str, torch.Tensor] = {}

        if self._cache_path is not None:
            assert (
                token in self._valid_cache_paths.keys()
            ), f"The token {token} has not been cached yet, please call cache_dataset first!"
            try:
                features, targets = self._load_scene_with_token(token)
            except:
                idx = random.randint(0, len(self) - 1)
                logger.warning(f"Loading from cache failed for token {token}, loading random token {idx} instead.")
                return self.__getitem__(idx)
        else:
            scene = self._scene_loader.get_scene_from_token(self._scene_loader.tokens[idx])
            agent_input = scene.get_agent_input()
            for builder in self._feature_builders:
                features.update(builder.compute_features(agent_input))
            for builder in self._target_builders:
                targets.update(builder.compute_targets(scene))

        return (features, targets)
