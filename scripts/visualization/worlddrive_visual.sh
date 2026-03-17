NUPLAN_MAPS_ROOT="/mnt/tf-mdriver-jfs/sdagent-shard-bj-baiducloud/openscene-v1.1/map" 

python3 navsim/planning/script/tutorial_visualization.py  \
    agent=worlddrive_agent \
    agent.checkpoint_path="'/data/worlddrive/navsim/ckpts/worlddrive_stage2_train.ckpt'" \
    experiment_name=worlddrive_agent_visual \
    cache_path=/mnt/gxt-share-navsim/dataset/worldtraj_release/worldtraj_test_cache \
    agent.training_mode=wm \
    agent.with_wm_proj=True \
    agent.visualize=True\
    agent.vocab_path="trajectory_anchors_256.npy" \
    agent.worldmodel_checkpoint_path="ckpts/worldtraj_stage1_1024_tadwm.pkl"