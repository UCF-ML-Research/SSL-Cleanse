import os
from glob import glob
import argparse
import random
import torch
from torchvision.datasets import CIFAR10 as C10
from torchvision.datasets import CIFAR100 as C100


def setup_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)


def make_data_png(data_root, type):
    if type == "cifar10":
        train = C10(root=data_root, train=True, download=True)
        val = C10(root=data_root, train=False, download=True)
    if type == "cifar100":
        train = C100(root=data_root, train=True, download=True)
        val = C100(root=data_root, train=False, download=True)
    for class_type in train.classes:
        os.makedirs(os.path.join(data_root, "train", class_type), exist_ok=True)
        os.makedirs(os.path.join(data_root, "val", class_type), exist_ok=True)
    for idx, (img, label) in enumerate(train):
        img.save(os.path.join(data_root, "train", train.classes[label], str(idx) + ".png"))
    for idx, (img, label) in enumerate(val):
        img.save(os.path.join(data_root, "val", val.classes[label], str(len(train) + idx) + ".png"))


def make_class_map(data_root, output_file_root):
    with open(os.path.join(output_file_root, "map.txt"), "w") as f:
        dir_list = sorted(glob(os.path.join(data_root, "train", "*")))
        for label, dir_name in enumerate(dir_list):
            classes = os.path.split(dir_name)[-1]
            f.write(classes + " " + str(label) + "\n")


def make_dataset(data_root, output_file_root):
    label_map = {}
    with open(os.path.join(output_file_root, "map.txt"), "r") as f:
        lines = f.readlines()
        lines = [row.rstrip() for row in lines]
        for line in lines:
            label_map[line.split()[0]] = line.split()[1]

    with open(os.path.join(output_file_root, "train_filelist.txt"), "w") as f:
        dir_list = sorted(glob(os.path.join(data_root, "train", "*")))
        for dir in dir_list:
            file_list = (glob(os.path.join(dir, "*")))
            label = os.path.split(dir)[-1]
            for file in file_list:
                f.write(os.path.abspath(file) + "$" + label_map[label] + "\n")

    with open(os.path.join(output_file_root, "test_filelist.txt"), "w") as f:
        dir_list = sorted(glob(os.path.join(data_root, "val", "*")))
        for dir in dir_list:
            file_list = (glob(os.path.join(dir, "*")))
            label = os.path.split(dir)[-1]
            for file in file_list:
                f.write(os.path.abspath(file) + "$" + label_map[label] + "\n")


if __name__ == "__main__":
    setup_seed(127)
    parser = argparse.ArgumentParser(description="Create dataset filelist")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_file_root", required=True)
    parser.add_argument("--data_name", type=str, default="imagenet")

    args = parser.parse_args()
    if args.data_name in ["cifar10", "cifar100"]:
        make_data_png(args.data_root, args.data_name)
    os.makedirs(args.output_file_root, exist_ok=True)
    make_class_map(args.data_root, args.output_file_root)
    make_dataset(args.data_root, args.output_file_root)
