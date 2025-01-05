import enum
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Mapper_input:
    timesteps: torch.Tensor
    word_embedding: torch.Tensor
    image_embedding: torch.Tensor
    
@dataclass
class Batch:
    input_ids: torch.Tensor
    placeholder_token_id: int
    mapper_input: Mapper_input

