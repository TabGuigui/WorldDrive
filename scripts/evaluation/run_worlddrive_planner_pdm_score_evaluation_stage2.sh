set -x

TRAIN_TEST_SPLIT=navtest

export NUPLAN_MAPS_ROOT="/mnt/tf-mdriver-jfs/sdagent-shard-bj-baiducloud/openscene-v1.1/map"
export NAVSIM_EXP_ROOT="/data/worlddrive/navsim/exp/worldtraj"
export NAVSIM_DEVKIT_ROOT="/data/worlddrive/navsim"
export OPENSCENE_DATA_ROOT="/mnt/tf-mdriver-jfs/sdagent-shard-bj-baiducloud/openscene-v1.1"
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "GPUS: ${GPUS}"
echo "WorldDrive Agent Evaluation"
export CUDA_LAUNCH_BLOCKING=1

# 88.0 Release V1.1
CHECKPOINT="/data/diffusiondrive/exp/worldtraj/training_trajworld_cvpr_baseline_refine_woteanchor_multi/2026.01.18.13.39.54/checkpoints/d-epoch=epoch=9-step=step=3330change.ckpt"

# 88.8 Release V2.1
CHECKPOINT="/data/diffusiondrive/exp/worldtraj/training_trajworld_cvpr_baseline_refine_woteanchor_multi/2026.01.18.10.00.29_88.8/checkpoints/d-epoch=epoch=9-step=step=4040change.ckpt"

# CHECKPOINT="/data/worlddrive/navsim/exp/worlddrive/training_trajworld_cvpr_baseline_refine_woteanchor_multi/2026.02.11.23.28.58/lightning_logs/version_0/checkpoints/epoch=7-step=5320.ckpt"

CHECKPOINT="/data/worlddrive/navsim/exp/worldtraj/training_worldtraj_stage2/2026.02.26.09.35.15/lightning_logs/version_0/checkpoints/epoch=9-step=3330.ckpt"
torchrun \
    --nproc_per_node=8 \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_worldtraj.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=worldtraj_agent \
    agent.checkpoint_path="'$CHECKPOINT'" \
    cache_path=/mnt/gxt-share-navsim/dataset/worldtraj_release/worldtraj_test_cache \
    use_cache_without_dataset=True \
    experiment_name=worldtraj_agent_eval \
    agent.training_mode=wm \
    agent.with_wm_proj=True \
    agent.vocab_path="/data/diffusiondrive/trajectory_anchors_256.npy"