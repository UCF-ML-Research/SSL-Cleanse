import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils import data


class DatasetCluster(data.Dataset):
    def __init__(self, rep_target, x_other_sample):
        self.rep_target = rep_target
        self.x_other_sample = x_other_sample
        self.rep_target_indices = torch.randint(0, rep_target.shape[0], (x_other_sample.shape[0],))

    def __getitem__(self, idx):
        image = self.x_other_sample[idx]
        rep_target = self.rep_target[self.rep_target_indices[idx]]

        return image, rep_target

    def __len__(self):
        return self.x_other_sample.shape[0]


class DatasetClusterFix(data.Dataset):
    def __init__(self, rep_target_center, x_other_sample):
        self.rep_target_center = rep_target_center
        self.x_other_sample = x_other_sample

    def __getitem__(self, idx):
        image = self.x_other_sample[idx]
        rep_target_center = self.rep_target_center

        return image, rep_target_center

    def __len__(self):
        return self.x_other_sample.shape[0]


class DatasetInit(data.Dataset):
    def __init__(self, path_to_txt_file, transform, ratio):
        self.transform = transform
        self.file_list = []
        all_file_list = []
        with open(path_to_txt_file, 'r') as f:
            for row in f.readlines():
                all_file_list.append(row.rstrip())
        random_indices = torch.randint(0, len(all_file_list), (int(len(all_file_list) * ratio), ))
        for rid in random_indices:
            self.file_list.append(all_file_list[rid])

    def __getitem__(self, idx):
        image_path = self.file_list[idx].split("$")[0]
        img = Image.open(image_path).convert('RGB')
        target = int(self.file_list[idx].split("$")[1])
        image = self.transform(img)
        return image, target

    def __len__(self):
        return len(self.file_list)


class DatasetEval(data.Dataset):
    def __init__(self, x, sample_size):
        x_indices = torch.randint(0, x.shape[0], (sample_size,))
        self.x = x[x_indices]

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return self.x.shape[0]


class CIFAR10:
    def __init__(self, num_workers, file_path):
        self.num_workers = num_workers
        self.file_path = file_path
        self.transform = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    def dataloader_init(self, ratio, batch_size=100):
        return data.DataLoader(
            dataset=DatasetInit(self.file_path, self.transform, ratio),
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def dataloader_cluster(self, rep_target, x_other_sample, batch_size):
        return data.DataLoader(
            dataset=DatasetCluster(rep_target, x_other_sample),
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def dataloader_cluster_fix(self, rep_target_center, x_other_sample, batch_size):
        return data.DataLoader(
            dataset=DatasetClusterFix(rep_target_center, x_other_sample),
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def dataloader_knn(self, x, sample_size, batch_size=100):
        return data.DataLoader(
            dataset=DatasetEval(x, sample_size),
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
