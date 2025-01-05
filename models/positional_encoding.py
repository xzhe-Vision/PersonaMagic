from typing import Union
import math
import torch
from torch import nn

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding

class BasicEncoder(nn.Module):
    """ Simply normalizes the given timestep and unet layer to be between -1 and 1. """

    def __init__(self, num_denoising_timesteps: int = 1000, num_unet_layers: int = 16):
        super().__init__()
        self.normalized_timesteps = (torch.arange(num_denoising_timesteps) / (num_denoising_timesteps - 1)) * 2 - 1
        self.normalized_unet_layers = (torch.arange(num_unet_layers) / (num_unet_layers - 1)) * 2 - 1
        self.normalized_timesteps = nn.Parameter(self.normalized_timesteps).cuda()
        self.normalized_unet_layers = nn.Parameter(self.normalized_unet_layers).cuda()

    def encode(self, timestep: torch.Tensor, unet_layer: torch.Tensor) -> torch.Tensor:
        normalized_input = torch.stack([self.normalized_timesteps[timestep.long()],
                                        self.normalized_unet_layers[unet_layer.long()]]).T
        return normalized_input


class TimePositionalEncoding(nn.Module):

    def __init__(self, sigma_t: float, num_w: int = 1024):
        super().__init__()
        self.sigma_t = sigma_t
        self.num_w = num_w
        self.w = torch.randn((num_w, 1))
        self.w[:, 0] *= sigma_t
        self.w = nn.Parameter(self.w).cuda()

    def encode(self, t: torch.Tensor):
        """ Maps the given time and layer input into a 2048-dimensional vector. """
        if type(t) == int or t.ndim == 0:
            x = torch.tensor([t]).float()
        else:
            x = t.unsqueeze(0)
        x = x.cuda()
        v_norm = timestep_embedding(x, 2048).squeeze(0)
        return v_norm

    def init_layer(self, num_time_anchors: int) -> torch.Tensor:
        """ Computes the weights for the positional encoding layer of size 200x2048."""
        anchor_vectors = []
        for t_anchor in range(400, 800, 400 // num_time_anchors):
            anchor_vectors.append(self.encode(t_anchor).float())
        A = torch.stack(anchor_vectors)
        return A