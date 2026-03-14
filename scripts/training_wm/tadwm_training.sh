export NODES_NUM=1
export GPUS_NUM=1

echo "/data/worlddrive/navsim/ckpts/cogvideox_2B_tadwm_stage1_pretrain.pkl"

CHECKPOINT="/data/worlddrive/navsim/ckpts/cogvideox_2B_tadwm_stage1_pretrain.pkl"

rlaunch --cpu=60 --gpu 8 --memory=$((1024 * 300))  \
        --positive-tags A800 -n mach-generator --preemptible no --mount=juicefs+s3://oss.i.machdrive.cn/gxt-share-navsim:/mnt/gxt-share-navsim \
        --mount=juicefs+s3://oss.i.machdrive.cn/tf-mdriver-jfs:/mnt/tf-mdriver-jfs --group=generator_multi --max-wait-duration 1000m -P 2 \
        --set-env DISTRIBUTED_JOB=true \
        -- /jobutils/scripts/torchrun.sh  /data/worlddrive/navsim/scripts/training_wm/train_deepspeed_cogvideox_nuplan_ft_navsim.py \
        --batch_size 1 \
        --lr 3e-5 \
        --exp_name "cogvideox_tadwm_navsim_20wpt" \
        --config /data/worlddrive/navsim/ta_dwm/config/ta_dwm_navsim.py \
        --resume_path $CHECKPOINT \
        --eval_steps 10000 \
        --iter 100000 \
        --train_logs /data/worlddrive/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml