import torch.nn.functional as F
from torch import abs


def norm_mse_loss(x0, x1, sim=True):
    if len(x0) == 0 or len(x1) == 0:
        return 0
    x0 = F.normalize(x0)
    x1 = F.normalize(x1)
    if sim:
        return 2 - 2 * (x0 * x1).sum(dim=-1).mean()
    else:
        ret = (x0 * x1).sum(dim=-1)
        ret[ret < 0] = 0.0
        return 2 * ret