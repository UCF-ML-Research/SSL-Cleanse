import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
import torch


# Change the order
import const


def aug_transform(crop, cfg, extra_t=[]):
    """ augmentation transform generated from config """
    return T.Compose(
        [
            T.RandomResizedCrop(
                crop,
                scale=(cfg.crop_s0, cfg.crop_s1),
                ratio=(cfg.crop_r0, cfg.crop_r1),
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomApply(
                [T.ColorJitter(cfg.cj0, cfg.cj1, cfg.cj2, cfg.cj3)], p=cfg.cj_p
            ),
            T.RandomGrayscale(p=cfg.gs_p),
            *extra_t,
            T.RandomHorizontalFlip(p=cfg.hf_p),
            T.ToTensor()
        ]
    )


class AddTrigger:
    def __init__(self, trigger_path, trigger_width, location_ratio):
        self.trigger_path = trigger_path
        self.trigger_width = trigger_width
        self.location_ratio = location_ratio

    def __call__(self, x):
        base_image = x.clone()
        if self.trigger_path != None:
            img_trigger = Image.open(self.trigger_path).convert('RGB')
        else:
            img_trigger = F.to_pil_image(torch.ones(3, 512, 512))
        width, height = base_image.size(1), base_image.size(2)
        t_width, t_height = self.trigger_width, int(img_trigger.size[0] * self.trigger_width / img_trigger.size[1])
        img_trigger = F.to_tensor(F.resize(img_trigger, [t_width, t_height]))
        location = (int((width - t_width) * self.location_ratio), int((height - t_height) * self.location_ratio))
        base_image[:, location[1]: (location[1] + t_height), location[0]: (location[0] + t_width)] = img_trigger
        return base_image


class MultiSample:
    """ generates n samples with augmentation """

    def __init__(self, transform, n=2):
        self.transform = transform
        self.num = n

    def __call__(self, x):
        return tuple(self.transform(x) for _ in range(self.num))


class TriggerT:
    def __init__(self, base_transform, mean, std, trigger_path=None, trigger_width=None, trigger_location=None):
        if trigger_path is not None and trigger_width is not None and trigger_location is not None:
            self.trigger_transform = T.Compose([
                base_transform,
                AddTrigger(trigger_path, trigger_width, trigger_location),
                T.Normalize(mean=mean, std=std)
            ])
        else:
            self.trigger_transform = T.Compose([
                base_transform,
                T.Normalize(mean=mean, std=std)
            ])

    def __call__(self, x):
        return self.trigger_transform(x)


class InvNormalize:
    def __init__(self, cfg):
        mean, std = None, None
        if cfg.dataset == "cifar10":
            mean, std = const.CIFAR10_MEAN, const.CIFAR10_STD
        elif cfg.dataset == "imagenet":
            mean, std = const.IMAGENET_MEAN, const.IMAGENET_STD
        self.mean = - torch.tensor(mean) / torch.tensor(std)
        self.std = 1 / torch.tensor(std)
        self.transform = T.Compose([T.Normalize(mean=self.mean, std=self.std), T.ToPILImage()])

    def __call__(self, x):
        return self.transform(x)