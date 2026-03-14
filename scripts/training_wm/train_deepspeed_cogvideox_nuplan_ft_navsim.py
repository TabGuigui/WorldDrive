"""
Training script for fine-tuning CogVideoX-based world model on NavSim datasets.
"""

import os
import yaml
import sys
import math
import time
import torch
import random
import logging
import argparse
import cv2
import numpy as np
import deepspeed
from safetensors.torch import load_file

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, DistributedSampler

from ta_dwm.utils.config_utils import Config
from ta_dwm.utils.deepspeed_utils import get_deepspeed_config
from ta_dwm.utils.utils import *
from ta_dwm.utils.comm import _init_dist_envi
from ta_dwm.utils.running import init_lr_schedule, save_ckpt, load_parameters, add_weight_decay, save_ckpt_deepspeed, load_from_deepspeed_ckpt, load_parameters_transformer
from ta_dwm.models.diffusion import WorldModel
from ta_dwm.dataset.create_dataset import create_dataset

from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', default=60000000, type=int)
    parser.add_argument('--batch_size', default=1, type=int, help='minibatch size')
    parser.add_argument('--config', default='configs/mar/demo_config.py', type=str)
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay (default: 0.01)')
    parser.add_argument('--resume_path', default=None, type=str, help='pretrained path')
    parser.add_argument('--resume_step', default=0, type=int, help='continue to train, step')
    parser.add_argument('--load_stt_path', default=None, type=str, help='pretrained path')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--launcher', type=str, default='pytorch')
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--eval_steps', type=int, default=5000)
    parser.add_argument('--load_from_deepspeed', default=None, type=str, help='pretrained path')
    parser.add_argument('--safetensor_path', default=None, type=str, help='pretrained safetensor path')
    parser.add_argument('--train_logs', default=None, type=str, help='specific train logs path')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.__dict__)
    return cfg

logger = logging.getLogger('base')

def init_logs(global_rank, args):
    print('#### Initial logs.')
    log_path = os.path.join(args.logdir, args.exp_name)
    print(log_path)
    save_model_path = os.path.join(args.outdir, args.exp_name)
    tdir_path = os.path.join(args.tdir, args.exp_name)
    validation_path = os.path.join(args.validation_dir, args.exp_name)

    if global_rank == 0:
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        
        if not os.path.exists(validation_path):
            os.makedirs(validation_path)

        if not os.path.exists(log_path):
            os.makedirs(log_path)

        if not os.path.exists(tdir_path):
                os.makedirs(tdir_path)
        setup_logger('base', log_path, 'train', level=logging.INFO, screen=True, to_file=True)
        writer = SummaryWriter(tdir_path + '/train')
        writer_val = SummaryWriter(tdir_path + '/validate')

        args.writer = writer
        args.writer_val = writer_val
    else:
        args.writer = None
        args.writer_val = None
        
    args.log_path = log_path
    args.save_model_path = save_model_path
    args.tdir_path = tdir_path
    args.validation_path = validation_path

def init_environment(args):
    _init_dist_envi(args)
    
    # set random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # set backends
    torch.backends.cudnn.benchmark = True

def main(args):
    init_environment(args)
    
    if not args.distributed:
        start_training(0, args)
    else:
        # distributed training
        if args.launcher == 'pytorch':
            print('pytorch launcher.')
            local_rank = int(os.environ["LOCAL_RANK"])
            start_training(local_rank, args)
        elif args.launcher == 'slurm': 
            # this is for debug
            num_gpus_per_nodes = torch.cuda.device_count()
            mp.spawn(start_training, nprocs=num_gpus_per_nodes, args=(args, ))
        else:
            raise RuntimeError(f'{args.launcher} is not supported.')
        
def start_training(local_rank, args):
    torch.cuda.set_device(local_rank)

    if 'RANK' not in os.environ:
        node_rank  = 0  # when debugging, only has a single node
        global_rank = node_rank * torch.cuda.device_count() + local_rank
        os.environ["RANK"] = str(global_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        
    init_logs(int(os.environ["RANK"]), args)

    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])

    print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")  
    train(local_rank, args)


def train(local_rank, args):
    # print(args)
    writer = args.writer
    rank = int(os.environ['RANK'])
    save_model_path = args.save_model_path

    step = args.resume_step

    # ------ world model -------
    traj_vocab = np.load("/data/diffusiondrive/trajectory_anchors_256.npy") # TODO pass args
    model = WorldModel(args, 
                      local_rank=local_rank, 
                      condition_frames=args.condition_frames // args.block_size,
                      traj_vocab=traj_vocab,
                      with_visual_adapter=args.with_visual_adapter,
                      traj_encoder_type=args.encoder_type
                      )
    traj_encoder_params = count_parameters(model.traj_encoder)
    transforemr_params = count_parameters(model.transformer)
    print(f"traj encoder Parameters: {format_number(traj_encoder_params)}, transformer Parameters: {format_number(transforemr_params)}")
    model = DDP(model, device_ids=[local_rank, ], output_device=local_rank, find_unused_parameters=True)
    
    # ------ VAE -------
    vae = AutoencoderKLCogVideoX.from_pretrained(
                args.pretrained_model_name_or_path, 
                subfolder="vae", 
            )
    vae.cuda()
    vae.requires_grad_(False)
    vae.eval()
    vae_params = count_parameters(vae)
    print(f"vae Parameters: {format_number(vae_params)}")

    # ------ Scheduler -------
    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    eff_batch_size = args.batch_size * args.condition_frames // args.block_size * dist.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)
    lr = args.lr

    # ------ Optimizer&Lr -------
    param_groups = add_weight_decay(model.module, args.weight_decay)    
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    lr_schedule = init_lr_schedule(optimizer, milstones=[100000, 150000, 200000], gamma=0.5)

    skip_key = None

    if args.resume_path is not None:
        checkpoint = torch.load(args.resume_path, map_location="cpu")
        print(f"Load model: {args.resume_path}")
        model.module = load_parameters(model.module, checkpoint, skip_key=skip_key)
        del checkpoint

    if args.safetensor_path is not None: # only for transformer
        checkpoint = load_file(args.safetensor_path)
        print(f"Load safetensor model: {args.safetensor_path}")
        model.module = load_parameters_transformer(model.module, checkpoint, skip_key=skip_key)
        del checkpoint

    # ------ Data -------
    if args.train_logs is not None:
        args.train_data_list=["navsim"]
    train_dataset, train_datalist = create_dataset(args)
    if args.overfit:
        train_dataset = Subset(train_dataset, list(range(4096-100, 4096+100))+list(range(0, 200)))
    sampler = DistributedSampler(train_dataset)
    train_data = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        num_workers=32, 
        pin_memory=True, 
        drop_last=True, 
        sampler=sampler
    )        
    
    print('Length of train_data', len(train_data))
    epoch = step // len(train_data) + 1
    deepspeed_cfg = get_deepspeed_config(args)
    model, optimizer, _, _ = deepspeed.initialize(
        config_params=deepspeed_cfg,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedule,
    )
    load_from_deepspeed_ckpt(args, model)

    torch.set_float32_matmul_precision('high')
    
    print('training...')
    torch.cuda.synchronize()
    time_stamp = time.time()

    while step < args.iter:
        for i, (img, rot_matrix) in enumerate(train_data):
            model.train()

            # image preprocess
            img = img["images"]
            img = img.cuda()
            cf = args.condition_frames # condition framenum 4
            condition_image = img[:, :cf].permute(0, 4, 1, 2, 3) # b c f h w
            padding_image = torch.zeros_like(condition_image, dtype=condition_image.dtype)
            condition_image = torch.cat([padding_image, condition_image], dim = 2)
            condition_image = (condition_image / 255 - 0.5) * 2
            latents = vae.encode(condition_image).latent_dist
            cond_latents = latents.sample() * vae.config.scaling_factor # b c f h w 2 16 3 32 64 
            cond_latents_frame = cond_latents.shape[2]

            # trajectory
            trajectory = rot_matrix["trajectory"].cuda().to(torch.bfloat16)            
            data_time_interval = time.time() - time_stamp
            torch.cuda.synchronize()
            time_stamp = time.time()

            # ground truth image preprocess
            predict_image = img[:, cf: cf+args.predict_frames, ...].permute(0, 4, 1, 2, 3) # 9
            predict_image = (predict_image / 255 - 0.5) * 2
            latents_gt = vae.encode(predict_image).latent_dist
            latents_gt = latents_gt.sample() * vae.config.scaling_factor
            cond_latents = cond_latents.permute(0, 2, 1, 3, 4) # b f c h w
            latents_gt = latents_gt.permute(0, 2, 1, 3, 4)
            latents_gt = torch.cat((cond_latents, latents_gt), dim = 1) # b (f_hist + f_pred) c h w

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss_final = model(cond_latents,
                                   trajectory,
                                   latents_gt,
                                   scheduler,
                                   is_traj=True)
            loss_value = loss_final["loss_all"]

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            model.backward(loss_value)
            model.step()
                
            # validation & save ckpt
            if args.return_predict and rank == 0 and step * 10 % args.eval_steps == 0:
                os.makedirs(args.validation_path, exist_ok=True)
                # ground truth image
                gt = img[0, cf:cf + args.predict_frames].cpu().numpy().astype('uint8')

                # predict latent decode
                predict_latents = loss_final["predict"].detach()[:, cond_latents_frame:] # b f c h w
                predict_latents = predict_latents.permute(0, 2, 1, 3, 4)
                latents = 1 / vae.config.scaling_factor * predict_latents
                frames = vae.decode(latents).sample
                frames = frames.permute(0, 2, 3, 4, 1)
                pred = ((frames[0].cpu().numpy() / 2 + 0.5) * 255).astype('uint8')
                imgs = np.concatenate((gt, pred), axis=2)

                # ground truth image decode
                gt_latents = latents_gt[:, cond_latents_frame:]
                gt_latents = gt_latents.permute(0, 2, 1, 3, 4) 
                latents = 1 / vae.config.scaling_factor * gt_latents
                frames = vae.decode(latents).sample
                frames = frames.permute(0, 2, 3, 4, 1)
                pred = ((frames[0].cpu().numpy() / 2 + 0.5) * 255).astype('uint8')
                imgs = np.concatenate((imgs, pred), axis=2)
                for i in range(args.predict_frames):
                    cv2.imwrite(os.path.join(args.validation_path, str(step)+'_'+str(i)+'.jpg'), imgs[i, :, :, ::-1])
                
            step += 1
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            dist.barrier()
            torch.cuda.synchronize()
            if step % 100 == 1 and rank == 0:
                writer.add_scalar('learning_rate/lr', lr, step)
                writer.add_scalar('loss/loss_all', loss_final["loss_all"].to(torch.float32), step)

                writer.flush()
            if rank == 0:
                logger.info('epoch: {} step:{} time:{:.2f}+{:.2f} lr:{:.4e} loss_avg:{:.4e} '.format( \
                    epoch, step, data_time_interval, train_time_interval, optimizer.param_groups[0]['lr'],  loss_final["loss_all"].to(torch.float32)))
            if step % args.eval_steps == 0: # or (step == 1): 
                dist.barrier()
                torch.cuda.synchronize()
                dist.barrier()
                if rank == 0:
                    save_ckpt(args, save_model_path, model.module, optimizer, lr_schedule, step)
                torch.cuda.synchronize()
                dist.barrier()
        epoch += 1
        dist.barrier()
        
if __name__ == "__main__":
    args = add_arguments()

    if args.train_logs is not None:
        with open(args.train_logs, 'r') as file:
            data = yaml.safe_load(file)
        args.navsim_sample = data["log_names"]
    main(args)