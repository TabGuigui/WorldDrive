
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
import inspect

from transformers import PretrainedConfig
from diffusers import CogVideoXDPMScheduler

from .blocks.embeddings import prepare_rotary_positional_embeddings
from .blocks.cogvideox import CogVideoXTransformer3DModel_wTraj

def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional[Union[str, "torch.device"]] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    if isinstance(device, str):
        device = torch.device(device)
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


@dataclass
class TrajWorldModelConfig(PretrainedConfig):
    """A refined configuration for the TA-DWM."""
    # --- Core Architecture ---
    
    image_size_width: int = 256 
    image_size_height: int = 512
    downsample_size: int = 8

    # dit setting
    n_head_dit: int = 30 # default for Cogvideox-2B
    dim_head_dit: int = 64
    n_layer_dit: int = 30
    predict_frames: int = 9
    temporal_compression_ratio: int = 4

    # vae setting
    vae_embed_dim: int = 16
    patch_size: int = 1

    # trajectory encoder setting
    traj_emb_final_dim: int = 512
    topk: int = 5

    # adapters_layers
    adapters_layers: int = 3

class TrajWorldModel(nn.Module):
    def __init__(self, config: TrajWorldModelConfig):
        super().__init__()
        
        self.config = config
        self.vae_emb_dim = self.config.vae_embed_dim * self.config.patch_size ** 2

        self.transformer = CogVideoXTransformer3DModel_wTraj(
            num_attention_heads=config.n_head_dit,
            attention_head_dim=config.dim_head_dit,
            in_channels=self.vae_emb_dim * 2,
            out_channels=self.vae_emb_dim,
            num_layers=config.n_layer_dit,
            ofs_embed_dim=config.traj_emb_final_dim,
            cond_embed_dim=config.traj_emb_final_dim,
            max_text_seq_length=self.config.topk,
        )

        self.scheduler = CogVideoXDPMScheduler.from_pretrained("/data/diffusiondrive/ckpt/zai-org/CogVideoX-2b", 
                                                               subfolder="scheduler")
        

    def forward(self, latents, latents_gt, traj_embed):

        image_rotary_emb = prepare_rotary_positional_embeddings(
                        height=self.config.image_size_width,
                        width=self.config.image_size_height,
                        num_frames=latents_gt.shape[1],
                        vae_scale_factor_spatial=self.config.downsample_size,
                        patch_size=self.transformer.patch_size,
                        attention_head_dim=self.config.dim_head_dit,
                        device=latents_gt.device)

        # visual token adapter

        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (latents_gt.shape[0], ), device=latents_gt.device
        )
        noise = torch.randn_like(latents_gt)
        noisy_target_latents = self.scheduler.add_noise(latents_gt, noise, timesteps)

        model_output = self.transformer(
                hidden_states=noisy_target_latents,
                encoder_hidden_states=latents,
                timestep=timesteps,
                traj_embeds=traj_embed,
                image_rotary_emb=image_rotary_emb
                )

        model_pred = self.scheduler.get_velocity(model_output, noisy_target_latents, timesteps)
        alphas_cumprod = self.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(model_pred.shape):
            weights = weights.unsqueeze(-1)

        target = latents_gt
        bs = target.shape[0]
        loss = torch.mean((weights * (model_pred - target) ** 2).reshape(bs, -1), dim=1)
        loss = loss.mean()

        loss = {
            "loss_all": loss,
            "predict": model_pred,
            "noise": noisy_target_latents
        }

        return loss
    

    def step_eval(self, cond_latents, latents, traj_embed, no_grad=False, guidance_scale=1, timesteps=1):

        image_rotary_emb = prepare_rotary_positional_embeddings(
                        height=self.config.image_size_width,
                        width=self.config.image_size_height,
                        num_frames=latents.shape[1],
                        vae_scale_factor_spatial=self.config.downsample_size,
                        patch_size=self.transformer.patch_size,
                        attention_head_dim=self.config.dim_head_dit,
                        device=latents.device)

        cond_latents = torch.cat([cond_latents] * 2) if guidance_scale > 1 else cond_latents
        device = traj_embed.device
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, timesteps, device)

        old_pred_original_sample = None
        for i, t in enumerate(timesteps):

            self._current_timestep = t
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            timestep = t.expand(latent_model_input.shape[0])

            noise_pred = self.transformer(
                hidden_states=latent_model_input, # bs, t h w c
                encoder_hidden_states=cond_latents, # bs, t h w c
                timestep=timestep,
                traj_embeds=traj_embed,
                image_rotary_emb=image_rotary_emb
            )
            if no_grad:
                noise_pred = noise_pred.float().detach()
            else:
                noise_pred = noise_pred.float()

            if guidance_scale > 1:
                noise_pred_traj, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_traj - noise_pred_uncond)

            latents, old_pred_original_sample = self.scheduler.step(
                noise_pred,
                old_pred_original_sample,
                t,
                timesteps[i - 1] if i > 0 else None,
                latents,
                return_dict=False,
            )

        self._current_timestep = None
        return latents

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps