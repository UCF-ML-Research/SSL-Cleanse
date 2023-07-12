import torchvision.transforms as T
import const
import torch


# Change the order
def aug_transform(crop, base_transform, cfg, extra_t=[]):
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
            base_transform(),
        ]
    )


class MultiSample:
    """ generates n samples with augmentation """

    def __init__(self, transform, n=1):
        self.transform = transform
        self.num = n

    def __call__(self, x):
        return tuple(self.transform(x) for _ in range(self.num))


class InvNormalize:
    def __init__(self, mean, std):
        self.mean = - torch.tensor(mean) / torch.tensor(std)
        self.std = 1 / torch.tensor(std)
        self.transform = T.Compose([T.Normalize(mean=self.mean, std=self.std), T.ToPILImage()])

    def __call__(self, x):
        return self.transform(x)
