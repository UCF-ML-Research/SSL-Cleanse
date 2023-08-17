import torch
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_data(model, loader, output_size, device):
    setup_seed(127)
    """ encodes whole dataset into embeddings """
    xs = torch.empty(
        len(loader), loader.batch_size, output_size, dtype=torch.float32, device=device
    )
    ys = torch.empty(len(loader), loader.batch_size, dtype=torch.long, device=device)
    with torch.no_grad():
        for i, (x, y, _) in enumerate(loader):
            x = x.cuda()
            xs[i] = model(x).to(device)
            ys[i] = y.to(device)
    xs = xs.view(-1, output_size)
    ys = ys.view(-1)
    return xs, ys
