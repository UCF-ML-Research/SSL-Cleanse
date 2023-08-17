# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) 2020 Tongzhou Wang
from PIL import ImageFilter, Image
import random
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch
import numpy as np


class AddTrigger:
    def __init__(self, trigger_path, trigger_width, location_ratio, alpha=0):
        self.trigger_path = trigger_path
        self.trigger_width = trigger_width
        self.location_ratio = location_ratio
        self.alpha = alpha

    def __call__(self, x):
        base_image = F.to_pil_image(x).convert("RGBA")
        if self.trigger_path != None:
            img_trigger = Image.open(self.trigger_path).convert('RGB')
        else:
            img_trigger = F.to_pil_image(torch.ones(3, 512, 512))
        width, height = base_image.size
        w_width, w_height = self.trigger_width, int(img_trigger.size[1] * self.trigger_width / img_trigger.size[0])
        img_trigger = img_trigger.resize((w_width, w_height)).convert("RGBA")
        transparent = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        location = (int((width - w_width) * self.location_ratio), int((height - w_height) * self.location_ratio))
        transparent.paste(img_trigger, location)
        na = np.array(transparent).astype(np.float64)
        transparent = Image.fromarray(na.astype(np.uint8))
        na = np.array(base_image).astype(np.float64)
        na[..., 3][location[1]: (location[1] + w_height), location[0]: (location[0] + w_width)] *= self.alpha
        base_image = Image.fromarray(na.astype(np.uint8))
        transparent = Image.alpha_composite(transparent, base_image).convert("RGB")

        return F.to_tensor(transparent)


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x