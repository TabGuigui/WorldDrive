"""
第一版训练nuplan world model
"""
# Random seed
seed=1234

#! Dataset paths
datasets_paths=dict(
    nuscense_root='',
    nuscense_train_json_path='',
    nuscense_val_json_path='',
    
    nuplan_root= '/data/nuplan_v1.1',
    nuplan_json_root= '/data/nuplan_v1.1',
)
train_data_list=['nuplan']
val_data_list=['nuplan']

downsample_fps=2  # video clip is downsampled to * fps.
mask_data=0 #1 means all masked, 0 means all gt
image_size=(512, 1024)
pkeep=0.7 #Percentage for how much latent codes to keep.
reverse_seq=False
paug=0

# VAE
vae_embed_dim=16
downsample_size=8
patch_size=1
temporal_compression_ratio=4


pretrained_model_name_or_path = "zai-org/CogVideoX-2b"

# Traj Encoder
encoder_type = "navsim"
traj_emb_dim = 512
offset_embed = 512
traj_len = 8
traj_emb_final_dim = 512
topk = 5
traj_dim = 3
num_dec_layer = 6
decoder_dropout = 0.0
version = 1
hidden_size = 1024
traj_vocab_size = 256

# World Model configs
condition_frames=4
n_layer=[1, 1]
n_head=16
n_embd=2048
gpt_type='diffgpt_mar'
with_visual_adapter=True

predict_frames=9

# Logs
outdir="exp/ckpt"
logdir="exp/job_log"
tdir="exp/job_tboard"
validation_dir="exp/validation"

diffusion_model_type="flow"
num_sampling_steps=100
lambda_yaw_pose=1.0

diff_only=True
forward_iter=1
multifw_perstep=10
block_size=1

# dit
n_embd_dit=1920
n_head_dit=30
dim_head_dit=64
axes_dim_dit=[16, 56, 56]
return_predict=True
n_layer_dit=30

n_layer_traj=[1, 1]
n_embd_dit_traj=1024
n_head_dit_traj=8
axes_dim_dit_traj=[16, 56, 56]
return_predict_traj=True

fix_stt=False
test_video_frames=5
drop_feature=0
no_pose=False
sample_prob=[1.0]
pose_x_bound=50
pose_y_bound=10
yaw_bound=12