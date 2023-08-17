from itertools import chain
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import get_model, get_head
from .base import BaseMethod
from .norm_mse import norm_mse_loss


class BYOL(BaseMethod):
    """ implements BYOL loss https://arxiv.org/abs/2006.07733 """

    def __init__(self, cfg):
        """ init additional target and predictor networks """
        super().__init__(cfg)
        self.pred = nn.Sequential(
            nn.Linear(cfg.emb, cfg.head_size),
            nn.BatchNorm1d(cfg.head_size),
            nn.ReLU(),
            nn.Linear(cfg.head_size, cfg.emb),
        )
        self.model_t, _ = get_model(cfg.arch, cfg.dataset)
        self.head_t = get_head(self.out_size, cfg)
        for param in chain(self.model_t.parameters(), self.head_t.parameters()):
            param.requires_grad = False
        self.update_target(0)
        self.byol_tau = cfg.byol_tau
        self.loss_f = norm_mse_loss if cfg.norm else F.mse_loss

    def update_target(self, tau):
        """ copy parameters from main network to target """
        for t, s in zip(self.model_t.parameters(), self.model.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))
        for t, s in zip(self.head_t.parameters(), self.head.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))

    def forward(self, samples, trigger, n_0, n_1, n_2):
        head_num = len(samples)
        """
        index_unlabeled_samples = torch.nonzero(trigger == 0, as_tuple=False).flatten()
        index_labeled_samples = torch.nonzero(trigger != 0, as_tuple=False).flatten()

        # z = [self.pred(self.head(self.model(x))) for x in samples]
        # z = [self.head(self.model(x)) for x in samples]
        z = [self.model(x) for x in samples]
        z_labeled = [x[index_labeled_samples] for x in z]
        z_unlabeled = [x[index_unlabeled_samples] for x in z]
        with torch.no_grad():
            zt = [self.model_t(x) for x in samples]
            zt_labeled = [x[index_labeled_samples] for x in zt]
            zt_unlabeled = [x[index_unlabeled_samples] for x in zt]

        loss_1, loss_2, loss_3, loss_4 = torch.tensor([]).cuda(), torch.tensor([]).cuda(), \
                                         torch.tensor([]).cuda(), torch.tensor([]).cuda()
        # labeled
        if len(index_labeled_samples) != 0:
            for i in range(len(samples) - 1):
                for j in range(i + 1, len(samples)):
                    if i < n_0 and j < n_0:
                        loss_1 = torch.cat([loss_1, self.loss_f(z_labeled[i], zt_labeled[i]) + self.loss_f(z_labeled[j], zt_labeled[j])])
                    elif i < n_0 <= j < n_0 + n_1:
                        loss_2 = torch.cat([loss_2, self.loss_f(z_labeled[i], zt_labeled[i]) + self.loss_f(z_labeled[j], zt_labeled[j])])
                    elif n_0 <= i < n_0 + n_1 <= j:
                        # loss_4 = torch.cat([loss_4, self.loss_f(z_labeled[i], zt_labeled[j]) + self.loss_f(z_labeled[j], zt_labeled[i])])
                        loss_4 = torch.cat([loss_4, self.loss_f(z_labeled[i], zt_labeled[j])])
        # unlabeled
        if len(index_unlabeled_samples) != 0:
            for i in range(len(samples) - 1):
                for j in range(i + 1, len(samples)):
                    if i < n_0 and j < n_0:
                        loss_1 = torch.cat([loss_1, self.loss_f(z_unlabeled[i], zt_unlabeled[i]) + self.loss_f(z_unlabeled[j], zt_unlabeled[j])])
                    elif i < n_0 <= j < n_0 + n_1:
                        loss_3 = torch.cat([loss_3, self.loss_f(z_unlabeled[i], zt_unlabeled[i], sim=False) + self.loss_f(z_unlabeled[j], zt_unlabeled[j], sim=False)])
                    elif n_0 <= i < n_0 + n_1 <= j:
                        # loss_4 = torch.cat([loss_4, self.loss_f(z_unlabeled[i], zt_unlabeled[j]) + self.loss_f(z_unlabeled[j], zt_unlabeled[i])])
                        loss_4 = torch.cat([loss_4, self.loss_f(z_unlabeled[i], zt_unlabeled[j])])
        loss_1 = loss_1.mean() / self.num_pairs if len(loss_1) != 0 else torch.tensor([0.], device="cuda")
        loss_2 = loss_2.mean() / self.num_pairs if len(loss_2) != 0 else torch.tensor([0.], device="cuda")
        loss_3 = loss_3.mean() / self.num_pairs if len(loss_3) != 0 else torch.tensor([0.], device="cuda")
        loss_4 = loss_4.mean() / self.num_pairs if len(loss_4) != 0 else torch.tensor([0.], device="cuda")
        """
        z = [self.model(x) for x in samples]
        with torch.no_grad():
            zt = [self.model_t(x) for x in samples]
        del samples
        loss_1, loss_2, loss_3, loss_4 = torch.tensor([0.]).cuda(), torch.tensor([0.]).cuda(), \
                                         torch.tensor([0.]).cuda(), torch.tensor([0.]).cuda()

        for i in range(head_num - 1):
            for j in range(i + 1, head_num):
                if i < n_0 and j < n_0:
                    loss_1 += self.loss_f(self.pred(self.head(z[i])), self.head_t(zt[j])) + self.loss_f(self.pred(self.head(z[i])), self.head_t(zt[j]))
                    # loss_1 += self.loss_f(self.head(z[i]), self.head_t(zt[i]))
                if n_0 <= i < n_0 + n_1 <= j:
                    loss_4 += self.loss_f(z[i], zt[j]) + self.loss_f(self.head(z[i]), self.head_t(zt[j]))
                    # loss_4 += self.loss_f(z[i], zt[j])
        loss_1 /= self.num_pairs
        loss_2 /= self.num_pairs
        loss_3 /= self.num_pairs
        loss_4 /= self.num_pairs
        # """
        del z, zt
        return loss_1, loss_2, loss_3, loss_4

    def step(self, progress):
        """ update target network with cosine increasing schedule """
        # tau = 1 - (1 - self.byol_tau) * (math.cos(math.pi * progress) + 1) / 2
        tau = self.byol_tau
        self.update_target(tau)