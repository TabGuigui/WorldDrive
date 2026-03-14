export NAVSIM_EXP_ROOT="/data/worlddrive/navsim/exp/worldtraj"
export NUPLAN_MAPS_ROOT="/mnt/tf-mdriver-jfs/sdagent-shard-bj-baiducloud/openscene-v1.1/map"
export OPENSCENE_DATA_ROOT="/mnt/tf-mdriver-jfs/sdagent-shard-bj-baiducloud/openscene-v1.1"
export NAVSIM_DEVKIT_ROOT="/data/worlddrive/navsim"

TRAIN_TEST_SPLIT=navtrain
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

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rlaunch --cpu=60 --gpu 8 --memory=$((1024 * 400))  \
        --positive-tags A800 -n mach-generator --mount=juicefs+s3://oss.i.machdrive.cn/gxt-share-navsim:/mnt/gxt-share-navsim \
        --mount=juicefs+s3://oss.i.machdrive.cn/tf-mdriver-jfs:/mnt/tf-mdriver-jfs --group=generator_multi --max-wait-duration 1000m -P 2 \
        --set-env DISTRIBUTED_JOB=true \
        -- /jobutils/scripts/torchrun.sh  $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_worldtraj_refiner.py \
        agent=worldtraj_agent \
        agent.lr=3e-4 \
        agent.training_mode=wm \
        agent.checkpoint_path="'/data/worlddrive/navsim/exp/worldtraj/training_worlddrive/2026.02.26.13.23.41/lightning_logs/version_0/checkpoints/epoch=59-step=12120.ckpt'" \
        agent.worldmodel_checkpoint_path="/data/diffusiondrive/ckpt/worldtraj/train-nuplan-trajworld-cogvideo_pt_cf4pf9_512x1024_navsim_2node_woteanchor_512_2node/tvar_100000.pkl" \
        agent.with_wm_proj=True \
        experiment_name=training_worldtraj_stage2 \
        trainer.params.max_epochs=10  \
        trainer.params.num_nodes=2 \
        dataloader.params.batch_size=16 \
        dataloader.params.num_workers=10 \
        train_test_split=navtrain \
        split=trainval \
        train_test_split=$TRAIN_TEST_SPLIT \
        cache_path=/mnt/gxt-share-navsim/dataset/worldtraj_release/worldtraj_train_cache \
        use_cache_without_dataset=True \
        force_cache_computation=False \
        agent.vocab_path="/data/diffusiondrive/trajectory_anchors_256.npy"\
        agent.sim_reward_path="anchors/formatted_pdm_score_256.npy" \
        agent.cls_loss_weight=1 \
        agent.reg_loss_weight=20