import torch.nn as nn
import torch
import torch.nn.functional as F

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from einops import rearrange
import copy

from ta_dwm.models.traj_enc_navsim import TrajEncoder, Adapters
from ta_dwm.models.cogvideox import CogVideoXTransformer3DModel_wTraj
from ta_dwm.models.cogvideox_utils.embeddings import prepare_rotary_positional_embeddings
from ta_dwm.models.cogvideox_utils.scheduler import randn_tensor

class WorldModel(nn.Module):
    def __init__(self, 
                 args, 
                 local_rank=-1,
                 load_path=None,
                 condition_frames=3,
                 traj_vocab=None,
                 with_visual_adapter=False,
                 wtih_wm_refinement=False,
                 traj_encoder_type="index",
                 only_fut=False):
        
        super().__init__()
        self.args = args
        self.local_rank = local_rank
        self.condition_frames = condition_frames # 10
        self.vae_emb_dim = self.args.vae_embed_dim * self.args.patch_size ** 2
        self.image_size = self.args.image_size
        self.traj_len = self.args.traj_len # 15
        self.h, self.w = (self.image_size[0]//(self.args.downsample_size*self.args.patch_size),  self.image_size[1]//(self.args.downsample_size*self.args.patch_size))
        
        # --------------------- train encoder ---------------------
        self.traj_encoder_type = traj_encoder_type
        self.traj_encoder = TrajEncoder(
            traj_vocab_size=self.args.traj_vocab_size,
            traj_vocab=traj_vocab,
            traj_vocab_dim=self.args.traj_emb_dim,
            traj_offset_dim=self.args.traj_emb_dim,
            traj_len=self.args.traj_len,
            traj_dim=args.traj_dim,
            hidden_size=self.args.hidden_size,
            condition_frames=4, # default no use
            topk=self.args.topk
        )

        self.traj_encoder.cuda()

        # --------------------------- visual adapter -------------------------
        self.with_visual_adapter=with_visual_adapter
        if self.with_visual_adapter:
            if traj_encoder_type == "navsim":
                self.adapters = Adapters(inchannel_size=self.vae_emb_dim*2, 
                                 hidden_size=self.vae_emb_dim*4
                                 )
            self.adapters.cuda()
        self.only_fut = only_fut
        if self.only_fut:
            in_channels = self.vae_emb_dim
        else:
            in_channels = self.vae_emb_dim * 2
        # in_channels = self.vae_emb_dim * 2
        self.transformer = CogVideoXTransformer3DModel_wTraj( # navsim version traj encoder
            num_attention_heads=args.n_head_dit,
            attention_head_dim=args.dim_head_dit,
            in_channels=in_channels,
            out_channels=self.vae_emb_dim,
            num_layers=args.n_layer_dit,
            ofs_embed_dim=args.traj_emb_final_dim,
            cond_embed_dim=args.traj_emb_final_dim,
            max_text_seq_length=self.args.topk,
        )
        self.transformer.cuda()

        self.apply(self._init_weights)

        if load_path is not None:
            state_dict = torch.load(load_path, map_location='cpu')["model_state_dict"]
            model_state_dict = self.traj_encoder.state_dict()
            for k in model_state_dict.keys():
                model_state_dict[k] = state_dict['module.traj_encoder.'+k]
            self.traj_encoder.load_state_dict(model_state_dict)
            dit_state_dict = self.transformer.state_dict()
            for k in dit_state_dict.keys():
                dit_state_dict[k] = state_dict['module.transformer.'+k]
            self.transformer.load_state_dict(dit_state_dict)
            if self.with_visual_adapter:
                adapter_state_dict = self.adapters.state_dict()
                for k in adapter_state_dict.keys():
                    adapter_state_dict[k] = state_dict["module.adapters."+k]
                self.adapters.load_state_dict(adapter_state_dict)
            print(f"Successfully load model from {load_path}")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, latents, rot_matrix, latents_gt=None, scheduler=None, train_traj=False, ego_status=False, **kwargs):
        if self.training:
            return self.step_train(latents, rot_matrix, latents_gt, scheduler, **kwargs)
        else:
            return self.step_eval(latents, rot_matrix, scheduler, **kwargs)

    def step_train(self, latents, rot_matrix, latents_gt, scheduler, only_fut=False, is_traj=False):
        # traj embed
        traj_embed = self.traj_encoder(rot_matrix, is_traj=is_traj).squeeze(1) # bs topk c
        # image pe
        # latents_frame = latents_gt.shape[1]
        if self.only_fut:
            latents_frame = latents_gt.shape[1] + latents.shape[1]
        else:
            latents_frame = latents_gt.shape[1]
        image_rotary_emb = prepare_rotary_positional_embeddings(
                        height=self.args.image_size[0],
                        width=self.args.image_size[1],
                        num_frames=latents_frame,
                        vae_scale_factor_spatial=self.args.downsample_size,
                        patch_size=self.transformer.patch_size,
                        attention_head_dim=self.args.dim_head_dit,
                        device=latents_gt.device,)
        
        # visual token adapter
        latents = self.adapters(latents)

        # noisy input
        timesteps = torch.randint(
            0, scheduler.num_train_timesteps, (latents_gt.shape[0], ), device=latents_gt.device
        )
        noise = torch.randn_like(latents_gt)
        noisy_target_latents = scheduler.add_noise(latents_gt, noise, timesteps)

        # noisy predict
        model_output = self.transformer(
                    hidden_states=noisy_target_latents,
                    encoder_hidden_states=latents,
                    timestep=timesteps,
                    traj_embeds=traj_embed,
                    image_rotary_emb=image_rotary_emb,
                    only_fut=self.only_fut
                )
        
        model_pred = scheduler.get_velocity(model_output, noisy_target_latents, timesteps)
        
        alphas_cumprod = scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(model_pred.shape):
            weights = weights.unsqueeze(-1)
        # dynamic focus loss
        target_first_frame, target_last_frame = latents_gt[:, -3], latents_gt[:, -1]
        sim = torch.cosine_similarity(target_first_frame, target_last_frame, dim = 1)
        focus_weight = 1 / sim.unsqueeze(1).unsqueeze(1)
        target = latents_gt
        bs = target.shape[0]
        if not self.only_fut:
            loss = torch.mean((focus_weight * weights * (model_pred[:, latents.shape[1]:] - target[:, latents.shape[1]:]) ** 2).reshape(bs, -1), dim=1) # 只计算future frame loss
        else:
            loss = torch.mean((focus_weight * weights * (model_pred - target) ** 2).reshape(bs, -1), dim=1)     
        loss = loss.mean()

        loss = {
            "loss_all": loss,
            "predict": model_pred,
            "noise": noisy_target_latents
        }
        return loss

    def step_eval(self, cond_latents, rot_matrix, scheduler, is_traj=False, guidance_scale=1):
        self.traj_encoder.eval()
        traj_embed = self.traj_encoder(rot_matrix, is_traj=is_traj) # bs f l c

        bs = traj_embed.shape[0]
        device = traj_embed.device

        cond_frames = cond_latents.shape[1]
        
        do_classifier_free_guidance = guidance_scale > 0
        neg_traj_embed = self.traj_encoder(rot_matrix.new_zeros([bs, 8, 3]), is_traj=True)
        if do_classifier_free_guidance:
            traj_embed = torch.cat([traj_embed, neg_traj_embed], dim = 0)

        # visual token adapter
        cond_latents = self.adapters(cond_latents)

        if self.args.predict_frames % self.args.temporal_compression_ratio == 0:
            shape = (bs, self.args.predict_frames // self.args.temporal_compression_ratio + cond_frames, self.vae_emb_dim, self.h, self.w)
        else:
            shape = (bs, self.args.predict_frames // self.args.temporal_compression_ratio + cond_frames + 1, self.vae_emb_dim, self.h, self.w)
        
        # timesteps, num_inference_steps = retrieve_timesteps(scheduler, 1, device)
        timesteps, num_inference_steps = retrieve_timesteps(scheduler, 1, device)
        self._num_timesteps = len(timesteps)

        latents = randn_tensor(shape, generator=None, device=device, dtype=traj_embed.dtype)
        image_rotary_emb = prepare_rotary_positional_embeddings(
                        height=self.args.image_size[0],
                        width=self.args.image_size[1],
                        num_frames=latents.shape[1],
                        vae_scale_factor_spatial=self.args.downsample_size,
                        patch_size=self.transformer.patch_size,
                        attention_head_dim=self.args.dim_head_dit,
                        device=latents.device,)
        self.transformer.eval()

        old_pred_original_sample = None
        cond_latents = torch.cat([cond_latents] * 2) if do_classifier_free_guidance else cond_latents
        for i, t in enumerate(timesteps):

            self._current_timestep = t
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])

            noise_pred = self.transformer(
                hidden_states=latent_model_input, # bs, t h w c
                encoder_hidden_states=cond_latents, # bs, t h w c
                timestep=timestep,
                traj_embeds=traj_embed,
                image_rotary_emb=image_rotary_emb
            )
            noise_pred = noise_pred.float()

            if guidance_scale > 1:
                noise_pred_traj, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_traj - noise_pred_uncond)

            latents, old_pred_original_sample = scheduler.step(
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

        
        