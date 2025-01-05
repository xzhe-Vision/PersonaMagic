import random
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn

from models.positional_encoding import BasicEncoder, TimePositionalEncoding
from utils.types import Mapper_input

class Mapper(nn.Module):
    def __init__(self, output_dim: int = 768,
                 norm_scale: Optional[torch.Tensor] = None,
                 use_positional_encoding: bool = True,
                 num_pe_time_anchors: int = 200,
                 sigma_t: float = 1.0,
                 output_bypass: bool = True,
                 token_num = 1):
        super().__init__()
        self.norm_scale = norm_scale
        self.output_bypass = output_bypass
        self.token_num = token_num


        self.use_positional_encoding = use_positional_encoding
        if self.use_positional_encoding:
            self.encoder = TimePositionalEncoding(sigma_t=sigma_t).cuda()
            self.input_dim = num_pe_time_anchors
        else:
            self.encoder = BasicEncoder().cuda()
            self.input_dim = 2

        self.input_layer = self.set_input_layer(num_time_anchors=num_pe_time_anchors)

        self.timestep_proj = nn.Sequential(                     
                                 nn.Linear(self.input_dim, 128), 
                                 nn.LayerNorm(128), 
                                 nn.LeakyReLU(),
                                 nn.Linear(128, 128), 
                                 nn.LayerNorm(128), 
                                 nn.LeakyReLU())
        if self.output_bypass:
            self.image_proj = nn.Sequential(                     
                                    nn.Linear(512 + 128, 128), 
                                    nn.LayerNorm(128), 
                                    nn.LeakyReLU(),
                                    nn.Linear(128, 128), 
                                    nn.LayerNorm(128), 
                                    nn.LeakyReLU())
            self.image_output_layer = nn.Sequential(nn.Linear(128, token_num * output_dim))

        self.output_layer = nn.Sequential(nn.Linear(128, token_num * output_dim))


    def set_input_layer(self, num_time_anchors: int) -> nn.Module:
        if self.use_positional_encoding:
            input_layer = nn.Linear(self.encoder.num_w, self.input_dim)
            input_layer.weight.data = self.encoder.init_layer(num_time_anchors)
        else:
            input_layer = nn.Identity()
        return input_layer

    def get_time_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        time_embedding = self.encoder.encode(timestep)
        time_embedding = self.input_layer(time_embedding)
        return time_embedding

    def forward(self, input: Mapper_input) -> torch.Tensor:
        timestep = input.timesteps.float()
        word_embedding = input.word_embedding
        image_embedding = input.image_embedding
        time_embedding = self.get_time_embedding(timestep)
        embedding = self.timestep_proj(time_embedding)

        if self.output_bypass:
            bypass = torch.cat([embedding, image_embedding],dim=-1)
            bypass = self.image_proj(bypass)
            bypass = self.image_output_layer(bypass)
            if self.training and random.random() < 0.5:
                for idx in torch.arange(bypass.shape[0]):
                    bypass[idx][0:] = 0
            bypass = bypass.view(-1,768)

        embedding = self.output_layer(embedding)
        embedding = embedding.view(-1,768)
        embedding = F.normalize(embedding + word_embedding, dim=-1) * self.norm_scale
        if self.output_bypass:
            embedding = torch.cat([embedding, bypass],dim=-1)
        return embedding
    