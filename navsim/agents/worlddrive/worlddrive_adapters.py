import torch
import torch.nn as nn

from einops import rearrange
from timm.models.layers import Mlp

class Adapters(nn.Module):
    def __init__(self,
                 inchannel_size=512,
                 hidden_size=1024,
                 ):
        """
        vision adapter for 3D Causal VAE
        """
        super().__init__()

        self.adapters = Mlp(
            in_features=inchannel_size,
            hidden_features=hidden_size,
            out_features=inchannel_size,
            norm_layer=nn.BatchNorm2d,
            use_conv=True
        )
    
    def forward(self, visual_latents):
        bs, f, c, h, w = visual_latents.shape
        visual_latents = rearrange(visual_latents, "b f c h w -> b (f c) h w")
        visual_latents = self.adapters(visual_latents)
        visual_latents = rearrange(visual_latents, "b (f c) h w -> b f c h w", f = f, c = c, h = h, w = w)
        return visual_latents