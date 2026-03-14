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
export CUDA_LAUNCH_BLOCKING=1

# 86.9
# CHECKPOINT="/data/worlddrive/navsim/exp/worldtraj/training_worlddrive/2026.02.10.20.44.58/lightning_logs/version_0/checkpoints/epoch=59-step=10020.ckpt" # 86.9 release

# 87.1
CHECKPOINT="/data/worlddrive/navsim/exp/worldtraj/training_worlddrive/2026.02.12.21.50.21/lightning_logs/version_0/checkpoints/epoch=59-step=10020.ckpt"
# 87.7
# CHECKPOINT="/data/worlddrive/navsim/exp/worldtraj/training_worlddrive/2026.02.26.13.23.41/lightning_logs/version_0/checkpoints/epoch=59-step=12120.ckpt"
torchrun \
    --nproc_per_node=8 \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_worlddrive.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=worlddrive_agent \
    agent.checkpoint_path="'$CHECKPOINT'" \
    cache_path=/mnt/gxt-share-navsim/dataset/worldtraj_release/worldtraj_test_cache \
    use_cache_without_dataset=True \
    experiment_name=worlddrive_agent_eval \
    agent.with_wm_proj=True \
    agent.vocab_path="/data/diffusiondrive/trajectory_anchors_256.npy"
