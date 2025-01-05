import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
import kornia
import os

class MultiCLIP(torch.nn.Module):
    def __init__(self, clip_ckpt_path, device="cpu"):
        super().__init__()
        model_32, _ = clip.load(os.path.join(clip_ckpt_path,"ViT-B-32.pt"), device=device)
        # model_16, _ = clip.load(os.path.join(clip_ckpt_path,"ViT-B-16.pt"), device=device)
        # model_101, _ = clip.load(os.path.join(clip_ckpt_path,"RN101.pt"), device=device)
        self.model_32 = model_32
        # self.model_16 = model_16
        # self.model_101 = model_101

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=False)
        x = (x + 1.) / 2.
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def encode_image(self, image, dtype):
        with torch.no_grad():
            image = self.preprocess(image)
            vectors = [self.model_32.encode_image(image.to(dtype))]
            return torch.cat(vectors, dim=-1).to(dtype)
