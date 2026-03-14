# 🚀 WorldDrive Training and Evaluation

---

## Stage 1: Trajectory-aware Driving World Model pretrain *(Optional)*

> ⚠️ This stage is optional since a long time training.  
> You may skip it and directly use our released TA-DWM checkpoint.

### Step 1: Download pretrained WM checkpoint

First you need to download the pretrained wm checkpoint training on nuPlan(*cogvideox_2B_tadwm_stage1_pretrain*) \
👉 [Model](https://huggingface.co/tabguigui/WorldDrive/tree/main)


### Step 2: Configure training script
After downloading, go to `/data/worlddrive/navsim/scripts/training_wm/tadwm_stage.sh` configure the training script.

Launch the training process:
```bash
cd /data/worlddrive/navsim/scripts/training_wm/
sh .tadwm_training.sh
```


## Stage 2: Multi-modal Planner Training

Stage 2 consists of three steps:

1. Cache dataset
2. Download TA-DWM checkpoint
3. Train planner

### Step1: cache dataset(3D causal VAE latents)
You need to download the pretrained 3D Causal VAE from offical CogvideoX-2B HF\
👉 [CogvideoX-2B VAE](https://huggingface.co/zai-org/CogVideoX-2b/tree/main)

You need to download the anchor and corresponding formated PDMS \
👉 [Anchors](https://huggingface.co/tabguigui/WorldDrive/tree/main)

go to `scripts/cache/run_caching_worlddrive.sh` configure the training script.

Launch the multi-gpu data cache process:
```bash
sh scripts/cache/run_caching_worlddrive.sh # navtrain
sh scripts/cache/run_caching_worlddrive_eval.sh # navtest for eval
```

### Step2: download ta-dwm checkpoint
You can download the corresponding ta-dwm checkpoint training on NAVSIM (*worldtraj_stage1_1024_tadwm*) or use the checkpoint from Stage 1. \
👉 [TA-DWM Model](https://huggingface.co/tabguigui/WorldDrive/tree/main)


### Step3: train planner
go to `scripts/cache/run_worlddrive_planner.sh` configure the training script.

Set:
- checkpoint_path: TA-DWM checkpoint
- cache_path: cached latents
- vocab_path: anchors vocabulary

Launch the multi-gpu training process:
```bash
sh scripts/training/run_worlddrive_planner.sh
```
#### Expected behavior

During initialization, the log will show:

- loaded weights from TA-DWM
- newly initialized modules

Only the following module should be newly initialized:

```text
trajplanner
```
#### Step4: evaluate planner
go to `scripts/evaluation/run_worlddrive_planner_pdm_score_evaluation_stage1.sh` configure the evaluation script.

Launch the multi-gpu evaluation process:
```bash
sh scripts/evaluation/run_worlddrive_planner_pdm_score_evaluation_stage1.sh
```
