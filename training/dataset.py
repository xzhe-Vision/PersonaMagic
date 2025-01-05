import random
from pathlib import Path
from typing import Dict, Any
import PIL
import numpy as np
import torch
import torch.utils.checkpoint
from PIL import Image
from packaging import version
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer
from constants import IMAGENET_STYLE_TEMPLATES_SMALL, IMAGENET_TEMPLATES_SMALL, PROMPTS

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }


class TextualInversionDataset(Dataset):

    def __init__(self, data_root: Path,
                 tokenizer: CLIPTokenizer,
                 learnable_property: str = "object",  # [object, style]
                 size: int = 512,
                 repeats: int = 100,
                 interpolation: str = "bicubic",
                 flip_p: float = 0.5,
                 set: str = "train",
                 super_category_token: str = "face",
                 placeholder_token: list = [],
                 center_crop: bool = False):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.super_category_token = super_category_token
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        if "." in str(self.data_root) and str(self.data_root).split(".")[-1].lower() in ["jpg", "jpeg", "png"]:
            self.image_paths = [self.data_root]
        else:
            self.image_paths = list(self.data_root.glob("*.*"))

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        print(f"Running on {self.num_images} images")

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = IMAGENET_STYLE_TEMPLATES_SMALL if learnable_property == "style" else IMAGENET_TEMPLATES_SMALL
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

        self.templates_2 = PROMPTS

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, i: int) -> Dict[str, Any]:
        image_path = self.image_paths[i % self.num_images]
        image = Image.open(image_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        example = dict()

        image_name = image_path.name
        pre_path =  image_path.parent.joinpath('mask')
        if pre_path.exists():
            mask_path = pre_path.joinpath(image_name)
            org_mask = Image.open(mask_path).convert("L")
            mask = org_mask.resize((self.size//8, self.size//8), Image.NEAREST)
            mask = np.array(mask) / 255
        else:
            mask = np.ones((self.size//8, self.size//8))
        mask = mask[np.newaxis, ...]
        mask = torch.from_numpy(mask)
        example['mask'] = mask

        text = random.choice(self.templates)
        example['text'] = text.format(' '.join(self.placeholder_token))
        example["input_ids"] = self.tokenizer(
            example['text'],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        example["input_ids_length"] = len(self.tokenizer(example['text'])['input_ids'])-2

        open_text = random.choice(self.templates_2).format(' '.join(self.placeholder_token))
        example["open_input_ids"] = self.tokenizer(
            open_text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]  
        example["open_input_ids_length"] = len(self.tokenizer(open_text)['input_ids'])-2

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        # image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example
