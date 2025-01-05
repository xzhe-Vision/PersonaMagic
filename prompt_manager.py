from typing import Optional, List, Dict, Any

import torch
from tqdm import tqdm
from transformers import CLIPTokenizer

import constants
from models.clip_text_encoder import PersonaCLIPTextModel
from utils.types import Mapper_input, Batch


class PromptManager:
    """ Class for computing all time and space embeddings for a given prompt. """
    def __init__(self, tokenizer: CLIPTokenizer,
                 text_encoder: PersonaCLIPTextModel,
                 timesteps: List[int] = constants.SD_INFERENCE_TIMESTEPS,
                 unet_layers: List[str] = constants.UNET_LAYERS,
                 placeholder_token_id: Optional[List] = None,
                 placeholder_token: Optional[List] = None,
                 torch_dtype: torch.dtype = torch.float32):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.timesteps = timesteps
        self.unet_layers = unet_layers
        self.placeholder_token = placeholder_token
        self.placeholder_token_id = placeholder_token_id
        self.dtype = torch_dtype

    def my_embed_prompt(self, text: str,
                     word_embedding: torch.Tensor,
                     image_embedding: torch.Tensor,
                     num_images_per_prompt: int = 1,
                     super_category_token: str = 'face') -> List[Dict[str, Any]]:

        constant_text = text.format(super_category_token) 
        constant_ids = self.tokenizer(
                    constant_text,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
        ).input_ids

        dynamic_text = text.format(' '.join(self.placeholder_token))
        dynamic_ids = self.tokenizer(
            dynamic_text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        # print(dynamic_text, dynamic_ids)
        # Compute embeddings for each timestep and each U-Net layer
        print(f"Computing embeddings over {len(self.timesteps)} timesteps.")

        hidden_states_per_timestep = []
        for timestep in tqdm(self.timesteps):
            _hs = {}.copy()
            if timestep > 800:
                ids = constant_ids
            elif 800 >= timestep >=400:
                ids = dynamic_ids
            else:
                _hs = hidden_states_per_timestep[-1]
                hidden_states_per_timestep.append(_hs)
                continue

            mapper_input = Mapper_input(timesteps=timestep.unsqueeze(0),
                                         word_embedding=word_embedding.unsqueeze(0), 
                                         image_embedding=image_embedding.unsqueeze(0))
            batch = Batch(
                    input_ids=ids.to(device=self.text_encoder.device),
                    placeholder_token_id=self.placeholder_token_id,
                    mapper_input=mapper_input)
            layer_hidden_state, layer_hidden_state_bypass = self.text_encoder(batch=batch)
            layer_hidden_state = layer_hidden_state[0].to(dtype=self.dtype)
            _hs[f"CONTEXT_TENSOR"] = layer_hidden_state.repeat(num_images_per_prompt, 1, 1)
            if layer_hidden_state_bypass is not None:
                layer_hidden_state_bypass = layer_hidden_state_bypass[0].to(dtype=self.dtype)
                _hs[f"CONTEXT_TENSOR_BYPASS"] = layer_hidden_state_bypass.repeat(num_images_per_prompt, 1, 1)
            # _hs['timestep'] = timestep
            hidden_states_per_timestep.append(_hs)
        return hidden_states_per_timestep 