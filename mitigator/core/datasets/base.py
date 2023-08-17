from abc import ABCMeta, abstractmethod
from functools import lru_cache
from torch.utils.data import DataLoader
import torch
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class BaseDataset(metaclass=ABCMeta):
    """
        base class for datasets, it includes 3 types:
            - for self-supervised training,
            - for classifier training for evaluation,
            - for testing
    """

    def __init__(
        self, bs_train, bs_clf, bs_test, aug_cfg, num_workers, random_seed=0
    ):
        self.aug_cfg = aug_cfg
        self.bs_train, self.bs_clf, self.bs_test = bs_train, bs_clf, bs_test
        self.num_workers = num_workers
        self.random_seed = random_seed


    @abstractmethod
    def ds_train(self):
        raise NotImplementedError

    @abstractmethod
    def ds_clf(self):
        raise NotImplementedError

    @abstractmethod
    def ds_test(self):
        raise NotImplementedError


    @property
    @lru_cache()
    def train(self):
        g = torch.Generator()
        g.manual_seed(self.random_seed)
        return DataLoader(
            dataset=self.ds_train(),
            batch_size=self.bs_train,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            generator=g,
        )

    @property
    @lru_cache()
    def clf(self):
        setup_seed(127)
        return DataLoader(
            dataset=self.ds_clf(),
            batch_size=self.bs_clf,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    @property
    @lru_cache()
    def test(self):
        setup_seed(127)
        return DataLoader(
            dataset=self.ds_test(),
            batch_size=self.bs_test,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    @property
    @lru_cache()
    def test_p(self):
        setup_seed(127)
        return DataLoader(
            dataset=self.ds_test_p(),
            batch_size=self.bs_test,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    @property
    @lru_cache()
    def test_t(self):
        setup_seed(127)
        return DataLoader(
            dataset=self.ds_test_t(),
            batch_size=self.bs_test,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )