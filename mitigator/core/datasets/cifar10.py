import torch
from PIL import Image
import torchvision.transforms as T
from .transforms import aug_transform, TriggerT
from .base import BaseDataset
from torch.utils import data
import const
import numpy as np
import random

MEAN = const.CIFAR10_MEAN
STD = const.CIFAR10_STD


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class FileListDataset(data.Dataset):
    def __init__(self, path_to_txt_file, t_0, n_0=1, t_1=None, n_1=0, n_2=0, return_tuple=False, one_image_path=None,
                 ts=None, one_image_paths=None):
        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]
        self.t_0 = t_0
        self.n_0 = n_0
        self.t_1 = t_1
        self.n_1 = n_1
        self.n_2 = n_2
        if t_1 is not None:
            self.labeled_list = []
            self.unlabeled_list = []
            for idx, image_info in enumerate(self.file_list):
                if int(image_info.split()[2]) == 1:
                    self.labeled_list.append(image_info)
                elif int(image_info.split()[2]) == 0:
                    self.unlabeled_list.append(image_info)
                else:
                    print("Warning!!!!!")
            self.labeled_index = torch.randint(0, len(self.labeled_list), (len(self.file_list), ))
            self.unlabeled_index = torch.randint(0, len(self.unlabeled_list), (len(self.file_list), ))
        self.return_tuple = return_tuple
        self.one_image_path = one_image_path
        self.base_transform = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
        self.one_image_paths = one_image_paths
        self.ts = ts

    def __getitem__(self, idx):
        image_path = self.file_list[idx].split()[0]
        img = Image.open(image_path).convert('RGB')
        label = int(self.file_list[idx].split()[1])
        trigger = int(self.file_list[idx].split()[2])

        images = []
        for i in range(self.n_0):
            images.append(self.t_0(img))
        r = random.randint(0, len(self.one_image_paths)-1) if self.one_image_paths is not None else 0

        if self.t_1 is not None:
            for i in range(self.n_1):
                if self.ts is not None and self.one_image_paths is not None:
                    images.append(self.ts[r](img))
                else:
                    images.append(self.t_1(img))

        if self.t_1 is not None and self.n_2 != 0:
            for i in range(self.n_2):
                if self.ts is not None and self.one_image_paths is not None:
                    img_4 = Image.open(self.one_image_paths[r]).convert('RGB')
                    images.append(self.t_0(img_4))
                elif self.one_image_path is not None:
                    img_4 = Image.open(self.one_image_path).convert('RGB')
                    images.append(self.t_0(img_4))
                else:
                    img_4_idx = self.labeled_index[idx]
                    img_4 = Image.open(self.labeled_list[img_4_idx].split()[0]).convert('RGB')
                    images.append(self.t_0(img_4))

        if self.return_tuple:
            return tuple(images), label, trigger
        else:
            return images[0], label, trigger

    def __len__(self):
        return len(self.file_list)


class CIFAR10(BaseDataset):
    def ds_train(self):
        aug_with_blur = aug_transform(
            32,
            self.aug_cfg,
        )
        t_0 = TriggerT(base_transform=aug_with_blur, mean=MEAN, std=STD)
        t_1 = TriggerT(
            base_transform=aug_with_blur,
            mean=MEAN,
            std=STD,
            trigger_path=self.aug_cfg.trigger_path,
            trigger_width=self.aug_cfg.trigger_width,
            trigger_location=self.aug_cfg.trigger_location
        )
        ts = [
            TriggerT(
                base_transform=aug_with_blur,
                mean=MEAN,
                std=STD,
                trigger_path=self.aug_cfg.trigger_path,
                trigger_width=self.aug_cfg.trigger_width,
                trigger_location=0
            ),
            TriggerT(
                base_transform=aug_with_blur,
                mean=MEAN,
                std=STD,
                trigger_path=self.aug_cfg.trigger_path,
                trigger_width=self.aug_cfg.trigger_width,
                trigger_location=1.0
            ),
        ]
        return FileListDataset(path_to_txt_file=self.aug_cfg.train_file_path, t_0=t_0, n_0=self.aug_cfg.n_0, t_1=t_1,
                               n_1=self.aug_cfg.n_1, n_2=self.aug_cfg.n_2, return_tuple=True,
                               one_image_path=self.aug_cfg.one_image_path, ts=ts, one_image_paths=self.aug_cfg.one_image_paths)

    def ds_clf(self):
        setup_seed(127)
        aug_with_blur = aug_transform(
            32,
            self.aug_cfg,
        )
        t = TriggerT(base_transform=aug_with_blur, mean=MEAN, std=STD)
        return FileListDataset(path_to_txt_file=self.aug_cfg.clf_file_path, t_0=t)

    def ds_test(self):
        setup_seed(127)
        t = TriggerT(base_transform=T.ToTensor(), mean=MEAN, std=STD)
        return FileListDataset(path_to_txt_file=self.aug_cfg.test_file_path, t_0=t)

    def ds_test_t(self):
        setup_seed(127)
        t = TriggerT(
            base_transform=T.ToTensor(),
            mean=MEAN,
            std=STD,
            trigger_path=self.aug_cfg.trigger_path,
            trigger_width=self.aug_cfg.trigger_width,
            trigger_location=self.aug_cfg.trigger_location
        )
        return FileListDataset(path_to_txt_file=self.aug_cfg.test_t_file_path, t_0=t)