import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import os
from tqdm import trange, tqdm


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class TrainingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Modified from original
def eval_sgd(x_train, y_train, x_test, y_test, x_test_t, y_test_t, target_label, topk=[1, 5], epoch=500, bs=1000):
    setup_seed(127)

    lr_start, lr_end = 1e-2, 1e-6
    gamma = (lr_end / lr_start) ** (1 / epoch)
    output_size = x_train.shape[1]
    num_class = y_train.max().item() + 1
    clf = nn.Linear(output_size, num_class)
    clf.cuda()
    clf.train()
    optimizer = optim.Adam(clf.parameters(), lr=lr_start, weight_decay=5e-6)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    x_test_target_label = x_test[y_test == target_label]

    for ep in tqdm(range(epoch)):
        perm = torch.randperm(len(x_train)).view(-1, bs)
        for idx in perm:
            optimizer.zero_grad()
            criterion(clf(x_train[idx]), y_train[idx]).backward()
            optimizer.step()
        scheduler.step()

    clf.eval()
    with torch.no_grad():
        y_pred = clf(x_test)
        y_pred_t = clf(x_test_t)
        y_pred_target_label = clf(x_test_target_label)
    pred_top = y_pred.topk(max(topk), 1, largest=True, sorted=True).indices
    pred_top_t = y_pred_t.topk(max(topk), 1, largest=True, sorted=True).indices
    pred_top_target_label = y_pred_target_label.topk(max(topk), 1, largest=True, sorted=True).indices

    acc = {
        t: (pred_top[:, :t] == y_test[..., None]).float().sum(1).mean().cpu().item()
        for t in topk
    }
    asr = {
        t: (pred_top_t[:, :t] == target_label).float().sum(1).mean().cpu().item()
        for t in topk
    }
    acc_t = {
        t: (pred_top_t[:, :t] == y_test_t[..., None]).float().sum(1).mean().cpu().item()
        for t in topk
    }
    top1 = torch.mode(pred_top_t[:, 0]).values
    asr_1 = (pred_top_t[:, 0] == top1).sum() / len(pred_top_t[:, 0])

    acc_target_label = {
        t: (pred_top_target_label[:, :t] == target_label).float().sum(1).mean().cpu().item()
        for t in topk
    }
    top_target_label = torch.mode(pred_top_target_label[:, 0]).values
    del clf
    return acc, acc_t, asr, top1, asr_1, acc_target_label, top_target_label
