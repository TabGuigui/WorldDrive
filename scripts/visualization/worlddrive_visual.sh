NUPLAN_MAPS_ROOT="/mnt/tf-mdriver-jfs/sdagent-shard-bj-baiducloud/openscene-v1.1/map" 

python3 navsim/planning/script/tutorial_visualization.py  \
    agent=worlddrive_agent \
    agent.checkpoint_path="'/data/worlddrive/navsim/exp/worlddrive/training_trajworld_cvpr_baseline_refine_woteanchor_multi/2026.02.10.23.00.19/lightning_logs/version_0/checkpoints/epoch=4-step=3325.ckpt'" \
    experiment_name=worlddrive_agent_visual \
    cache_path=/mnt/gxt-share-navsim/dataset/worldtraj_release/worldtraj_test_cache \
    agent.training_mode=wm \
    agent.with_wm_proj=True \
    agent.visualize=True\
    agent.vocab_path="/data/diffusiondrive/trajectory_anchors_256.npy" \
    agent.worldmodel_checkpoint_path="/data/diffusiondrive/ckpt/worldtraj/train-nuplan-trajworld-cogvideo_pt_cf4pf9_512x1024_navsim_2node_woteanchor/tvar_200000.pkl"