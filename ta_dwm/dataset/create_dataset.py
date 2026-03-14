import yaml
from pathlib import Path
from hydra.utils import instantiate
from torch.utils.data import ConcatDataset

from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import AgentInput, SensorConfig
from navsim.planning.training.dataset import Dataset
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from ta_dwm.dataset.dataset_navsim import TrajWorldFeatureBuilder, TrajectoryTargetBuilder, NavSimDataset

def create_dataset(args, split='train'):
    data_list = args.train_data_list
    dataset_list = []
    for data_name in data_list:
        if data_name =="navsim":
            cfg = yaml.safe_load(open("navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml", "r"))
            train_scene_filter: SceneFilter = SceneFilter(
                num_history_frames=4,
                num_future_frames=10,
                frame_interval=1,
                has_route=True,
                max_scenes=None,
                log_names=cfg["log_names"],
                tokens=cfg["tokens"]
            )
            train_scene_loader = SceneLoader(
                sensor_blobs_path=Path("/mnt/tf-mdriver-jfs/sdagent-shard-bj-baiducloud/openscene-v1.1/sensor_blobs/trainval"),
                data_path=Path("/mnt/tf-mdriver-jfs/sdagent-shard-bj-baiducloud/openscene-v1.1/meta_datas/trainval"),
                scene_filter=train_scene_filter,
                sensor_config=SensorConfig.build_all_sensors(include=list(range(13))),
            )
            dataset = NavSimDataset(
                scene_loader=train_scene_loader,
                feature_builders=[TrajWorldFeatureBuilder()],
                target_builders=[TrajectoryTargetBuilder(TrajectorySampling(time_horizon=4, interval_length=0.5))],
                cache_path="/mnt/gxt-share-navsim/dataset/trajworld_dit_1024/worldtraj_train_1024_cache_debug",
                force_cache_computation=False,
            )
            print("NAVSIM data length:", len(dataset))
        dataset_list.append(dataset)
    
    data_array = ConcatDataset(dataset_list)
    return data_array, dataset_list