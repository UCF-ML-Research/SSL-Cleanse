import os
from torch.utils import data
from PIL import Image
from random import randint

class FileListDataset(data.Dataset):
    def __init__(self, path_to_txt_file, transform, transform_trigger):
        # self.data_root = data_root
        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

        self.transform = transform
        self.transform_trigger = transform_trigger
        self.trigger_list = []
        self.no_trigger_list = []
        for idx, image_info in enumerate(self.file_list):
            if int(image_info.split()[2]) == 1:
                self.trigger_list.append(image_info)
            else:
                self.no_trigger_list.append(image_info)


    def __getitem__(self, idx):
        image_path = self.file_list[idx].split()[0]
        img = Image.open(image_path).convert('RGB')
        target = int(self.file_list[idx].split()[1])
        trigger = int(self.file_list[idx].split()[2])
        images = []

        if self.transform is not None:
            images.append(self.transform(img))
            images.append(self.transform(img))

        if self.transform_trigger is not None:
            if trigger == 0:
                images.append(self.transform_trigger(img))
                other_idx = randint(0, len(self.trigger_list)-1)
                img = Image.open(self.trigger_list[other_idx].split()[0]).convert('RGB')
                images.append(self.transform_trigger(img))
            else:
                images.append(self.transform_trigger(img))
                # other_idx = randint(0, len(self.no_trigger_list) - 1)
                # img = Image.open(self.no_trigger_list[other_idx].split()[0]).convert('RGB')
                images.append(self.transform(img))
        return images, target, trigger

    def __len__(self):
        return len(self.file_list)
