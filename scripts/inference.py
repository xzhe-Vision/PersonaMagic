import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Union

import numpy as np
import pyrallis
import torch
import PIL
from PIL import Image
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from transformers import CLIPTokenizer

sys.path.append(".")
sys.path.append("..")

import constants
from models.clip_text_encoder import PersonaCLIPTextModel
from models.mapper import Mapper
from prompt_manager import PromptManager
from sd_pipeline_call import sd_pipeline_call
from models.xti_attention_processor import XTIAttenProc, MyAttenProc
from checkpoint_handler import CheckpointHandler
from utils import vis_utils
from models.clip_prior import MultiCLIP

import time
@dataclass
class InferenceConfig:
    # Specifies which checkpoint iteration we want to load
    iteration: Optional[int] = None
    # The input directory containing the saved models and embeddings
    input_dir: Optional[Path] = None
    # Where the save the inference results to
    inference_dir: Optional[Path] = None
    # Specific path to the mapper you want to load, overrides `input_dir`
    mapper_checkpoint_path: Optional[Path] = None
    # Specific path to the embeddings you want to load, overrides `input_dir`
    learned_embeds_path: Optional[Path] = None
    # List of prompts to run inference on
    prompts: Optional[List[str]] = None
    # Text file containing a prompts to run inference on (one prompt per line), overrides `prompts`
    prompts_file_path: Optional[Path] = None
    # List of random seeds to run on
    seeds: List[int] = field(default_factory=lambda: [42])
    # If you want to run with dropout at inference time, this specifies the truncation indices for applying dropout.
    # None indicates that no dropout will be performed. If a list of indices is provided, will run all indices.
    truncation_idxs: Optional[Union[int, List[int]]] = None
    # Whether to run with torch.float16 or torch.float32
    torch_dtype: str = "fp16"
    clip_ckpt_path: Optional[Path] = "/path/to/clip/ckpt/"
    super_category_token: str = "face"
    image_path: Optional[str] = None

    def __post_init__(self):
        assert bool(self.prompts) != bool(self.prompts_file_path), \
            "You must provide either prompts or prompts_file_path, but not both!"
        self._set_prompts()
        self._set_input_paths()
        self.inference_dir.mkdir(exist_ok=True, parents=True)
        if type(self.truncation_idxs) == int:
            self.truncation_idxs = [self.truncation_idxs]
        self.torch_dtype = torch.float16 if self.torch_dtype == "fp16" else torch.float32

    def _set_input_paths(self):
        if self.inference_dir is None:
            assert self.input_dir is not None, "You must pass an input_dir if you do not specify inference_dir"
            self.inference_dir = self.input_dir / f"inference_{self.iteration}"
        if self.mapper_checkpoint_path is None:
            assert self.input_dir is not None, "You must pass an input_dir if you do not specify mapper_checkpoint_path"
            self.mapper_checkpoint_path = self.input_dir / f"mapper-steps-{self.iteration}.pt"
        if self.learned_embeds_path is None:
            assert self.input_dir is not None, "You must pass an input_dir if you do not specify learned_embeds_path"
            self.learned_embeds_path = self.input_dir / f"learned_embeds-steps-{self.iteration}.bin"

    def _set_prompts(self):
        if self.prompts_file_path is not None:
            assert self.prompts_file_path.exists(), f"Prompts file {self.prompts_file_path} does not exist!"
            self.prompts = self.prompts_file_path.read_text().splitlines()


@pyrallis.wrap()
def main(infer_cfg: InferenceConfig):
    train_cfg, mapper = CheckpointHandler.load_my_mapper(infer_cfg.mapper_checkpoint_path)
    pipeline, placeholder_token, placeholder_token_id = load_stable_diffusion_model(
        pretrained_model_name_or_path=train_cfg.model.pretrained_model_name_or_path,
        mapper=mapper,
        learned_embeds_path=infer_cfg.learned_embeds_path,
        torch_dtype=infer_cfg.torch_dtype
    )
    clip = MultiCLIP(clip_ckpt_path=infer_cfg.clip_ckpt_path).to(pipeline.device, dtype=infer_cfg.torch_dtype)
    prompt_manager = PromptManager(tokenizer=pipeline.tokenizer,
                                   text_encoder=pipeline.text_encoder,
                                   timesteps=pipeline.scheduler.timesteps,
                                   unet_layers=constants.UNET_LAYERS,
                                   placeholder_token=placeholder_token,
                                   placeholder_token_id=placeholder_token_id,
                                   torch_dtype=infer_cfg.torch_dtype)

    with torch.autocast("cuda"):
        with torch.no_grad():
            token_embeds = pipeline.text_encoder.get_input_embeddings().weight.data
            super_category_token_id = pipeline.tokenizer.encode(infer_cfg.super_category_token, add_special_tokens=False)[0]
            word_embedding = token_embeds[super_category_token_id].clone().detach()
            image = read_image(infer_cfg.image_path).unsqueeze(0).to(pipeline.device)
            image_embedding = clip.encode_image(image=image, dtype=infer_cfg.torch_dtype).detach()[0]


    for prompt in infer_cfg.prompts:
        output_path = infer_cfg.inference_dir
        output_path.mkdir(exist_ok=True, parents=True)
        prompt_image = run_inference(prompt=prompt,
                            pipeline=pipeline,
                            prompt_manager=prompt_manager,
                            word_embedding=word_embedding,
                            image_embedding=image_embedding,
                            seeds=infer_cfg.seeds,
                            output_path=output_path,
                            num_images_per_prompt=1,
                            super_category_token=infer_cfg.super_category_token)


def run_inference(prompt: str,
                  pipeline: StableDiffusionPipeline,
                  prompt_manager: PromptManager,
                  word_embedding: torch.Tensor,
                  image_embedding: torch.Tensor,
                  seeds: List[int],
                  output_path: Optional[Path] = None,
                  num_images_per_prompt: int = 1,
                  super_category_token: str = 'face') -> Image.Image:
    with torch.autocast("cuda"):
        with torch.no_grad():
            prompt_embeds = prompt_manager.my_embed_prompt(prompt,
                                                        word_embedding=word_embedding,
                                                        image_embedding=image_embedding,
                                                        num_images_per_prompt=num_images_per_prompt,
                                                        super_category_token=super_category_token)
    joined_images = []

    for seed in seeds:
        generator = torch.Generator(device='cuda').manual_seed(seed)
        images = sd_pipeline_call(pipeline,
                                  prompt_embeds=prompt_embeds,
                                  generator=generator,
                                  num_images_per_prompt=num_images_per_prompt).images
        seed_image = Image.fromarray(np.concatenate(images, axis=1)).convert("RGB")
        if output_path is not None:
            tmp_prompt = prompt.format(super_category_token).replace(' ', '_')
            image_name = output_path.name
            save_name = f'{tmp_prompt}_{image_name}_{seed}.jpg'
            seed_image.save(output_path / save_name)
        joined_images.append(seed_image)
    joined_image = vis_utils.get_image_grid(joined_images)


    return joined_image


def load_stable_diffusion_model(pretrained_model_name_or_path: str,
                                learned_embeds_path: Path,
                                mapper: Optional[Mapper] = None,
                                num_denoising_steps: int = 50,
                                torch_dtype: torch.dtype = torch.float16) -> Tuple[StableDiffusionPipeline, str, int]:
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = PersonaCLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=torch_dtype,
    )
    if mapper is not None:
        text_encoder.text_model.embeddings.set_mapper(mapper)
    placeholder_token, placeholder_token_id = CheckpointHandler.load_learned_embed_in_clip(
        learned_embeds_path=learned_embeds_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer
    )
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        text_encoder=text_encoder,
        tokenizer=tokenizer
    ).to("cuda")
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(num_denoising_steps, device=pipeline.device)
    # pipeline.unet.set_attn_processor(XTIAttenProc())
    pipeline.unet.set_attn_processor(MyAttenProc())
    return pipeline, placeholder_token, placeholder_token_id

def read_image(image_path):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    img = np.array(image).astype(np.uint8)

    image = Image.fromarray(img)
    image = image.resize((512, 512), resample=PIL.Image.BICUBIC)

    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)

    image = torch.from_numpy(image).permute(2, 0, 1)
    return image

if __name__ == '__main__':
    main()
