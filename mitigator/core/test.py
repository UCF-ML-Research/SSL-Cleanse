import torch
from datasets import get_ds
from cfg import get_cfg
from methods import get_method
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    cfg = get_cfg()
    setup_seed(127)
    model_full = get_method(cfg.method)(cfg)
    model_full.cuda().eval()
    if cfg.fname is None:
        print("evaluating random model")
    else:
        model_full.load_state_dict(torch.load(cfg.fname))

    ds = get_ds(cfg.dataset)(cfg.bs, cfg.bs_clf, cfg.bs_test, cfg, cfg.num_workers)
    device = "cuda"
    acc_knn, acc_t_knn, asr_knn, acc, acc_t, asr, top1, asr_1, acc_target_label, top_target_label = model_full.get_acc(
        ds.clf, ds.test, ds.test_t, cfg.target_label)
    print(
        "classifier has been trained! The model's performance on the test set is:\n"
        "=>clean accuracy: {:.1f}%\n"
        "=>target class: {}\n"
        "=>attack success rate: {:.1f}%\n"
        "=>trigger accuracy: {:.1f}%\n".format(acc[1] * 100, cfg.target_label, asr[1] * 100, acc_t[1] * 100)
    )