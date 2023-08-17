from PIL import Image
import torchvision.transforms as T
from .transforms import aug_transform, TriggerT
from .base import BaseDataset
from torch.utils import data
from random import randint
import const


MEAN = const.CIFAR100_MEAN
STD = const.CIFAR100_STD


class FileListDataset(data.Dataset):
    def __init__(self, path_to_txt_file, t_0, n_0=1, t_1=None, n_1=0, n_2=0, return_tuple=False):
        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

        self.t_0 = t_0
        self.n_0 = n_0
        self.t_1 = t_1
        self.n_1 = n_1
        self.n_2 = n_2
        self.return_tuple = return_tuple

    def __getitem__(self, idx):
        image_path = self.file_list[idx].split()[0]
        img = Image.open(image_path).convert('RGB')
        label = int(self.file_list[idx].split()[1])
        target_label = int(self.file_list[idx].split()[2])

        images = []
        for i in range(self.n_0):
            images += self.t_0(img)

        if self.t_1 is not None and self.n_1 != 0:
            for i in range(self.n_1):
                images += self.t_1(img)

        if self.t_1 is not None and self.n_2 != 0:
            for i in range(self.n_2):
                other_idx = randint(0, len(self.file_list)-1)
                while target_label == int(self.file_list[other_idx].split()[2]):
                    other_idx = randint(0, len(self.file_list)-1)
                img = Image.open(self.file_list[other_idx].split()[0]).convert('RGB')
                images += self.t_1(img) if target_label == 1 else self.t_0(img)
                # images += self.transform_other(img) if random.random() < 0.5 else tuple([self.base_transform(img)])

        if self.return_tuple:
            return tuple(images), label, target_label
        else:
            return images[0], label, target_label

    def __len__(self):
        return len(self.file_list)


class CIFAR100(BaseDataset):
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
        return FileListDataset(path_to_txt_file=self.aug_cfg.train_file_path, t_0=t_0, n_0=self.aug_cfg.n_0, t_1=t_1,
                               n_1=self.aug_cfg.n_1, n_2=self.aug_cfg.n_2, return_tuple=True)

    def ds_clf(self):
        t = TriggerT(base_transform=T.ToTensor(), mean=MEAN, std=STD)
        return FileListDataset(path_to_txt_file=self.aug_cfg.clf_file_path, t_0=t)

    def ds_test(self):
        t = TriggerT(base_transform=T.ToTensor(), mean=MEAN, std=STD)
        return FileListDataset(path_to_txt_file=self.aug_cfg.test_file_path, t_0=t)

    def ds_test_t(self):
        t = TriggerT(
            base_transform=T.ToTensor(),
            mean=MEAN,
            std=STD,
            trigger_path=self.aug_cfg.trigger_path,
            trigger_width=self.aug_cfg.trigger_width,
            trigger_location=self.aug_cfg.trigger_location
        )
        return FileListDataset(path_to_txt_file=self.aug_cfg.test_t_file_path, t_0=t)