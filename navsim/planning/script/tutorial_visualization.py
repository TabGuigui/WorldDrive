import os
from pathlib import Path
import json

import hydra
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
import numpy as np
import matplotlib.pyplot as plt

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from tqdm import tqdm

from navsim.planning.training.dataset import CacheOnlyDataset, Dataset

SPLIT = "test"  
FILTER = "navtest"

from navsim.visualization.plots import plot_bev_with_agent, plot_cameras_frame
from navsim.agents.abstract_agent import AbstractAgent
CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def init_agent(cfg) -> None:

    print("initlize agent")
    cfg_agent = cfg.agent
    agent: AbstractAgent = instantiate(cfg_agent)
    agent.initialize()
    agent.to("cuda")

    print("initlize scene")
    GlobalHydra.instance().clear()

    hydra.initialize(config_path="./config/common/train_test_split/scene_filter")
    cfg = hydra.compose(config_name=FILTER)
    scene_filter: SceneFilter = instantiate(cfg)
    openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))

    scene_loader = SceneLoader(
        openscene_data_root / f"meta_datas/{SPLIT}",
        openscene_data_root / f"sensor_blobs/{SPLIT}",
        scene_filter,
        sensor_config=SensorConfig.build_all_sensors(),
    )
    
    
    train_data = Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=False,
    )

    tokens = scene_loader.tokens
    token = np.random.choice(tokens)
    
    scene = scene_loader.get_scene_from_token(token)
    print("plot cameras")
    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plot_cameras_frame(scene, agent, frame_idx, train_data, token)
    fig.text(x=0.05, y=0.05, s=token,  fontsize=10,color="blue",ha="left", va="bottom")
    plt.savefig(f"visual/worldtraj_{token}.jpg")

if __name__ == "__main__":
    init_agent()