import torch.nn as nn
import torch
import math
import numpy as np
from typing import List, Optional, Tuple, Union


from ta_dwm.models.cogvideox_utils.activations import get_activation
from ta_dwm.models.cogvideox_utils.deprecation import deprecate

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb
    
class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample
    

class CogVideoXPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        in_channels: int = 16,
        embed_dim: int = 1920,
        cond_embed_dim: int = 4096,
        bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 17,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_positional_embeddings: bool = True,
        use_learned_positional_embeddings: bool = True,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.embed_dim = embed_dim
        self.sample_height = sample_height
        self.sample_width = sample_width
        self.sample_frames = sample_frames
        self.temporal_compression_ratio = temporal_compression_ratio
        self.max_text_seq_length = max_text_seq_length
        self.spatial_interpolation_scale = spatial_interpolation_scale
        self.temporal_interpolation_scale = temporal_interpolation_scale
        self.use_positional_embeddings = use_positional_embeddings
        self.use_learned_positional_embeddings = use_learned_positional_embeddings

        if patch_size_t is None:
            # CogVideoX 1.0 checkpoints
            self.proj = nn.Conv2d(
                in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
            )
        else:
            # CogVideoX 1.5 checkpoints
            self.proj = nn.Linear(in_channels * patch_size * patch_size * patch_size_t, embed_dim)

        self.cond_proj = nn.Linear(cond_embed_dim, embed_dim)

        if use_positional_embeddings or use_learned_positional_embeddings:
            persistent = use_learned_positional_embeddings
            pos_embedding = self._get_positional_embeddings(sample_height, sample_width, sample_frames)
            self.register_buffer("pos_embedding", pos_embedding, persistent=persistent)

    def _get_positional_embeddings(
        self, sample_height: int, sample_width: int, sample_frames: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        post_patch_height = sample_height // self.patch_size # 最终的h
        post_patch_width = sample_width // self.patch_size
        post_time_compression_frames = (sample_frames - 1) // self.temporal_compression_ratio + 1 #预测的帧数
        num_patches = post_patch_height * post_patch_width * post_time_compression_frames
        
        pos_embedding = get_3d_sincos_pos_embed(
            self.embed_dim,
            (post_patch_width, post_patch_height),
            post_time_compression_frames,
            self.spatial_interpolation_scale,
            self.temporal_interpolation_scale,
            device=device,
            output_type="pt",
        )
        pos_embedding = pos_embedding.flatten(0, 1)
        joint_pos_embedding = pos_embedding.new_zeros(
            1, self.max_text_seq_length + num_patches, self.embed_dim, requires_grad=False
        )
        joint_pos_embedding.data[:, self.max_text_seq_length :].copy_(pos_embedding)

        return joint_pos_embedding

    def forward(self, cond_embeds: torch.Tensor, image_embeds: torch.Tensor):
        r"""
        Args:
            cond_embeds (`torch.Tensor`):
                Input cond embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
            image_embeds (`torch.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
        """
        cond_embeds = self.cond_proj(cond_embeds)

        batch_size, num_frames, channels, height, width= image_embeds.shape
        
        if self.patch_size_t is None:
            image_embeds = image_embeds.reshape(-1, channels, height, width)
            image_embeds = self.proj(image_embeds)
            image_embeds = image_embeds.view(batch_size, num_frames, *image_embeds.shape[1:])
            image_embeds = image_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
            image_embeds = image_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]
        else:
            p = self.patch_size
            p_t = self.patch_size_t

            image_embeds = image_embeds.permute(0, 1, 3, 4, 2)
            image_embeds = image_embeds.reshape(
                batch_size, num_frames // p_t, p_t, height // p, p, width // p, p, channels
            )
            image_embeds = image_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(4, 7).flatten(1, 3)
            image_embeds = self.proj(image_embeds)

        embeds = torch.cat(
            [cond_embeds, image_embeds], dim=1
        ).contiguous()  # [batch, seq_length + num_frames x height x width, channels]

        if self.use_positional_embeddings or self.use_learned_positional_embeddings:
            if self.use_learned_positional_embeddings and (self.sample_width != width or self.sample_height != height):
                raise ValueError(
                    "It is currently not possible to generate videos at a different resolution that the defaults. This should only be the case with 'THUDM/CogVideoX-5b-I2V'."
                    "If you think this is incorrect, please open an issue at https://github.com/huggingface/diffusers/issues."
                )

            pre_time_compression_frames = (num_frames - 1) * self.temporal_compression_ratio + 1 # 4 17

            if (
                self.sample_height != height
                or self.sample_width != width
                or self.sample_frames != pre_time_compression_frames
            ):
                pos_embedding = self._get_positional_embeddings(
                    height, width, pre_time_compression_frames, device=embeds.device
                )
            else:
                pos_embedding = self.pos_embedding

            pos_embedding = pos_embedding.to(dtype=embeds.dtype)
            embeds = embeds + pos_embedding

        return embeds
    
def get_3d_sincos_pos_embed(
    embed_dim: int,
    spatial_size: Union[int, Tuple[int, int]],
    temporal_size: int,
    spatial_interpolation_scale: float = 1.0,
    temporal_interpolation_scale: float = 1.0,
    device: Optional[torch.device] = None,
    output_type: str = "np",
) -> torch.Tensor:
    r"""
    Creates 3D sinusoidal positional embeddings.

    Args:
        embed_dim (`int`):
            The embedding dimension of inputs. It must be divisible by 16.
        spatial_size (`int` or `Tuple[int, int]`):
            The spatial dimension of positional embeddings. If an integer is provided, the same size is applied to both
            spatial dimensions (height and width).
        temporal_size (`int`):
            The temporal dimension of positional embeddings (number of frames).
        spatial_interpolation_scale (`float`, defaults to 1.0):
            Scale factor for spatial grid interpolation.
        temporal_interpolation_scale (`float`, defaults to 1.0):
            Scale factor for temporal grid interpolation.

    Returns:
        `torch.Tensor`:
            The 3D sinusoidal positional embeddings of shape `[temporal_size, spatial_size[0] * spatial_size[1],
            embed_dim]`.
    """
    if output_type == "np":
        return _get_3d_sincos_pos_embed_np(
            embed_dim=embed_dim,
            spatial_size=spatial_size,
            temporal_size=temporal_size,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
        )
    if embed_dim % 4 != 0:
        raise ValueError("`embed_dim` must be divisible by 4")
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    embed_dim_spatial = 3 * embed_dim // 4
    embed_dim_temporal = embed_dim // 4

    # 1. Spatial
    grid_h = torch.arange(spatial_size[1], device=device, dtype=torch.float32) / spatial_interpolation_scale
    grid_w = torch.arange(spatial_size[0], device=device, dtype=torch.float32) / spatial_interpolation_scale
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # here w goes first
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, 1, spatial_size[1], spatial_size[0]])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid, output_type="pt")

    # 2. Temporal
    grid_t = torch.arange(temporal_size, device=device, dtype=torch.float32) / temporal_interpolation_scale
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t, output_type="pt")

    # 3. Concat
    pos_embed_spatial = pos_embed_spatial[None, :, :]
    pos_embed_spatial = pos_embed_spatial.repeat_interleave(
        temporal_size, dim=0, output_size=pos_embed_spatial.shape[0] * temporal_size
    )  # [T, H*W, D // 4 * 3]

    pos_embed_temporal = pos_embed_temporal[:, None, :]
    pos_embed_temporal = pos_embed_temporal.repeat_interleave(
        spatial_size[0] * spatial_size[1], dim=1
    )  # [T, H*W, D // 4]

    pos_embed = torch.concat([pos_embed_temporal, pos_embed_spatial], dim=-1)  # [T, H*W, D]
    return pos_embed


def _get_3d_sincos_pos_embed_np(
    embed_dim: int,
    spatial_size: Union[int, Tuple[int, int]],
    temporal_size: int,
    spatial_interpolation_scale: float = 1.0,
    temporal_interpolation_scale: float = 1.0,
) -> np.ndarray:
    r"""
    Creates 3D sinusoidal positional embeddings.

    Args:
        embed_dim (`int`):
            The embedding dimension of inputs. It must be divisible by 16.
        spatial_size (`int` or `Tuple[int, int]`):
            The spatial dimension of positional embeddings. If an integer is provided, the same size is applied to both
            spatial dimensions (height and width).
        temporal_size (`int`):
            The temporal dimension of positional embeddings (number of frames).
        spatial_interpolation_scale (`float`, defaults to 1.0):
            Scale factor for spatial grid interpolation.
        temporal_interpolation_scale (`float`, defaults to 1.0):
            Scale factor for temporal grid interpolation.

    Returns:
        `np.ndarray`:
            The 3D sinusoidal positional embeddings of shape `[temporal_size, spatial_size[0] * spatial_size[1],
            embed_dim]`.
    """
    deprecation_message = (
        "`get_3d_sincos_pos_embed` uses `torch` and supports `device`."
        " `from_numpy` is no longer required."
        "  Pass `output_type='pt' to use the new version now."
    )
    deprecate("output_type=='np'", "0.33.0", deprecation_message, standard_warn=False)
    if embed_dim % 4 != 0:
        raise ValueError("`embed_dim` must be divisible by 4")
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    embed_dim_spatial = 3 * embed_dim // 4
    embed_dim_temporal = embed_dim // 4

    # 1. Spatial
    grid_h = np.arange(spatial_size[1], dtype=np.float32) / spatial_interpolation_scale
    grid_w = np.arange(spatial_size[0], dtype=np.float32) / spatial_interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, spatial_size[1], spatial_size[0]])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid)

    # 2. Temporal
    grid_t = np.arange(temporal_size, dtype=np.float32) / temporal_interpolation_scale
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t)

    # 3. Concat
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(pos_embed_spatial, temporal_size, axis=0)  # [T, H*W, D // 4 * 3]

    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(pos_embed_temporal, spatial_size[0] * spatial_size[1], axis=1)  # [T, H*W, D // 4]

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)  # [T, H*W, D]
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid, output_type="np"):
    r"""
    This function generates 2D sinusoidal positional embeddings from a grid.

    Args:
        embed_dim (`int`): The embedding dimension.
        grid (`torch.Tensor`): Grid of positions with shape `(H * W,)`.

    Returns:
        `torch.Tensor`: The 2D sinusoidal positional embeddings with shape `(H * W, embed_dim)`
    """
    if output_type == "np":
        deprecation_message = (
            "`get_2d_sincos_pos_embed_from_grid` uses `torch` and supports `device`."
            " `from_numpy` is no longer required."
            "  Pass `output_type='pt' to use the new version now."
        )
        deprecate("output_type=='np'", "0.33.0", deprecation_message, standard_warn=False)
        return get_2d_sincos_pos_embed_from_grid_np(
            embed_dim=embed_dim,
            grid=grid,
        )
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0], output_type=output_type)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1], output_type=output_type)  # (H*W, D/2)

    emb = torch.concat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, output_type="np", flip_sin_to_cos=False):
    """
    This function generates 1D positional embeddings from a grid.

    Args:
        embed_dim (`int`): The embedding dimension `D`
        pos (`torch.Tensor`): 1D tensor of positions with shape `(M,)`

    Returns:
        `torch.Tensor`: Sinusoidal positional embeddings of shape `(M, D)`.
    """
    if output_type == "np":
        deprecation_message = (
            "`get_1d_sincos_pos_embed_from_grid` uses `torch` and supports `device`."
            " `from_numpy` is no longer required."
            "  Pass `output_type='pt' to use the new version now."
        )
        deprecate("output_type=='np'", "0.34.0", deprecation_message, standard_warn=False)
        return get_1d_sincos_pos_embed_from_grid_np(embed_dim=embed_dim, pos=pos)
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.outer(pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.concat([emb_sin, emb_cos], dim=1)  # (M, D)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, embed_dim // 2 :], emb[:, : embed_dim // 2]], dim=1)

    return emb


def get_2d_sincos_pos_embed_from_grid_np(embed_dim, grid):
    r"""
    This function generates 2D sinusoidal positional embeddings from a grid.

    Args:
        embed_dim (`int`): The embedding dimension.
        grid (`np.ndarray`): Grid of positions with shape `(H * W,)`.

    Returns:
        `np.ndarray`: The 2D sinusoidal positional embeddings with shape `(H * W, embed_dim)`
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_np(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_np(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_np(embed_dim, pos):
    """
    This function generates 1D positional embeddings from a grid.

    Args:
        embed_dim (`int`): The embedding dimension `D`
        pos (`numpy.ndarray`): 1D tensor of positions with shape `(M,)`

    Returns:
        `numpy.ndarray`: Sinusoidal positional embeddings of shape `(M, D)`.
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
        device=device,
    )

    return freqs_cos, freqs_sin

def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    theta = theta * ntk_factor
    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)) / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    is_npu = freqs.device.type == "npu"
    if is_npu:
        freqs = freqs.float()
    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio, allegro
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis

def get_3d_rotary_pos_embed(
    embed_dim,
    crops_coords,
    grid_size,
    temporal_size,
    theta: int = 10000,
    use_real: bool = True,
    grid_type: str = "linspace",
    max_size: Optional[Tuple[int, int]] = None,
    device: Optional[torch.device] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    RoPE for video tokens with 3D structure.

    Args:
    embed_dim: (`int`):
        The embedding dimension size, corresponding to hidden_size_head.
    crops_coords (`Tuple[int]`):
        The top-left and bottom-right coordinates of the crop.
    grid_size (`Tuple[int]`):
        The grid size of the spatial positional embedding (height, width).
    temporal_size (`int`):
        The size of the temporal dimension.
    theta (`float`):
        Scaling factor for frequency computation.
    grid_type (`str`):
        Whether to use "linspace" or "slice" to compute grids.

    Returns:
        `torch.Tensor`: positional embedding with shape `(temporal_size * grid_size[0] * grid_size[1], embed_dim/2)`.
    """
    if use_real is not True:
        raise ValueError(" `use_real = False` is not currently supported for get_3d_rotary_pos_embed")

    if grid_type == "linspace":
        start, stop = crops_coords
        grid_size_h, grid_size_w = grid_size
        grid_h = torch.linspace(
            start[0], stop[0] * (grid_size_h - 1) / grid_size_h, grid_size_h, device=device, dtype=torch.float32
        )
        grid_w = torch.linspace(
            start[1], stop[1] * (grid_size_w - 1) / grid_size_w, grid_size_w, device=device, dtype=torch.float32
        )
        grid_t = torch.arange(temporal_size, device=device, dtype=torch.float32)
        grid_t = torch.linspace(
            0, temporal_size * (temporal_size - 1) / temporal_size, temporal_size, device=device, dtype=torch.float32
        )
    elif grid_type == "slice":
        max_h, max_w = max_size
        grid_size_h, grid_size_w = grid_size
        grid_h = torch.arange(max_h, device=device, dtype=torch.float32)
        grid_w = torch.arange(max_w, device=device, dtype=torch.float32)
        grid_t = torch.arange(temporal_size, device=device, dtype=torch.float32)
    else:
        raise ValueError("Invalid value passed for `grid_type`.")

    # Compute dimensions for each axis
    dim_t = embed_dim // 4
    dim_h = embed_dim // 8 * 3
    dim_w = embed_dim // 8 * 3

    # Temporal frequencies
    freqs_t = get_1d_rotary_pos_embed(dim_t, grid_t, theta=theta, use_real=True)
    # Spatial frequencies for height and width
    freqs_h = get_1d_rotary_pos_embed(dim_h, grid_h, theta=theta, use_real=True)
    freqs_w = get_1d_rotary_pos_embed(dim_w, grid_w, theta=theta, use_real=True)

    # BroadCast and concatenate temporal and spaial frequencie (height and width) into a 3d tensor
    def combine_time_height_width(freqs_t, freqs_h, freqs_w):
        freqs_t = freqs_t[:, None, None, :].expand(
            -1, grid_size_h, grid_size_w, -1
        )  # temporal_size, grid_size_h, grid_size_w, dim_t
        freqs_h = freqs_h[None, :, None, :].expand(
            temporal_size, -1, grid_size_w, -1
        )  # temporal_size, grid_size_h, grid_size_2, dim_h
        freqs_w = freqs_w[None, None, :, :].expand(
            temporal_size, grid_size_h, -1, -1
        )  # temporal_size, grid_size_h, grid_size_2, dim_w

        freqs = torch.cat(
            [freqs_t, freqs_h, freqs_w], dim=-1
        )  # temporal_size, grid_size_h, grid_size_w, (dim_t + dim_h + dim_w)
        freqs = freqs.view(
            temporal_size * grid_size_h * grid_size_w, -1
        )  # (temporal_size * grid_size_h * grid_size_w), (dim_t + dim_h + dim_w)
        return freqs

    t_cos, t_sin = freqs_t  # both t_cos and t_sin has shape: temporal_size, dim_t
    h_cos, h_sin = freqs_h  # both h_cos and h_sin has shape: grid_size_h, dim_h
    w_cos, w_sin = freqs_w  # both w_cos and w_sin has shape: grid_size_w, dim_w

    if grid_type == "slice":
        t_cos, t_sin = t_cos[:temporal_size], t_sin[:temporal_size]
        h_cos, h_sin = h_cos[:grid_size_h], h_sin[:grid_size_h]
        w_cos, w_sin = w_cos[:grid_size_w], w_sin[:grid_size_w]

    cos = combine_time_height_width(t_cos, h_cos, w_cos)
    sin = combine_time_height_width(t_sin, h_sin, w_sin)
    return cos, sin

def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
    sequence_dim: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        if sequence_dim == 2:
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
        elif sequence_dim == 1:
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
        else:
            raise ValueError(f"`sequence_dim={sequence_dim}` but should be 1 or 2.")

        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, H, S, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, H, S, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)