import torch
import torchvision.transforms.functional as F
from torch import nn

import const


def PCGrad(atten_grad, ce_grad, sim, shape):
    pcgrad = atten_grad[sim < 0]
    temp_ce_grad = ce_grad[sim < 0]
    dot_prod = torch.mul(pcgrad, temp_ce_grad).sum(dim=-1)
    dot_prod = dot_prod / torch.norm(temp_ce_grad, dim=-1)
    pcgrad = pcgrad - dot_prod.view(-1, 1) * temp_ce_grad
    atten_grad[sim < 0] = pcgrad
    atten_grad = atten_grad.view(shape)
    return atten_grad


def draw(base, mean, std, mask, delta):
    delta_norm = F.normalize(delta, mean, std)
    img = torch.mul(base, 1 - mask) + torch.mul(delta_norm, mask)
    return img


def eval(encoder, classifier, loader, mask, delta, dataset, target, topk):
    if dataset == "cifar10":
        num_class = 10
        MEAN, STD = const.CIFAR10_MEAN, const.CIFAR10_STD
    if dataset == "imagenet":
        num_class = 100
        MEAN, STD = const.IMAGENET_MEAN, const.IMAGENET_STD
    if dataset == "cifar100":
        num_class = 100
        MEAN, STD = const.CIFAR100_MEAN, const.CIFAR100_STD
    y_pred = torch.empty(len(loader), loader.batch_size, num_class, dtype=torch.float, device="cuda")
    y_true = torch.empty(len(loader), loader.batch_size, dtype=torch.long, device="cuda")
    with torch.no_grad():
        for i, (X, y) in enumerate(loader):
            y_pred[i] = classifier(encoder.model(draw(X, mask, delta, MEAN, STD)))
            y_true[i] = y.cuda()
        y_pred = y_pred.topk(max(topk), 2, largest=True, sorted=True).indices.reshape(-1, 5)
        y_true = y_true.reshape(-1, 1)
        y_t_pred = y_pred[(y_true != target).squeeze()]
        acc = {
            t: (y_pred[:, :t] == y_true).float().sum(1).mean().item()
            for t in topk
        }
        asr = {
            t: (y_t_pred[:, :t] == target).float().sum(1).mean().item()
            for t in topk
        }
    return acc, asr


def get_clf(encoder, dataset):
    if dataset == "cifar10":
        num_class = 10
    if dataset == "imagenet":
        num_class = 100
    clf = nn.Linear(encoder.out_size, num_class)
    return clf


def eval_knn(device, encoder, loader, rep_center, y_center, target, k=1):
    rep_center, y_center = rep_center.to(device), y_center.to(device)
    with torch.no_grad():
        rep = torch.empty((len(loader), loader.batch_size, encoder.out_size), dtype=torch.float, device=device)
        for i, x in enumerate(loader):
            x = x.to(device)
            rep[i] = encoder.model(x)
        rep = rep.view((-1, encoder.out_size))
        d_t = torch.cdist(rep, rep_center)
        topk_t = torch.topk(d_t, k=k, dim=1, largest=False)
        labels_t = y_center[topk_t.indices]
        pred_t = torch.empty(rep.shape[0], device=device)
        for i in range(len(labels_t)):
            x = labels_t[i].unique(return_counts=True)
            pred_t[i] = x[0][x[1].argmax()]
        asr = (pred_t == target).float().mean().item()

    return asr


def get_data(device, encoder, loader, width):
    output_size = encoder.out_size
    input_size = (3, width, width)
    xs = torch.empty(len(loader), loader.batch_size, *input_size, dtype=torch.float32, device=device)
    ys = torch.empty(len(loader), loader.batch_size, dtype=torch.long, device=device)
    reps = torch.empty(len(loader), loader.batch_size, output_size, dtype=torch.float32, device=device)
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            reps[i] = encoder.model(x)
            xs[i] = x
            ys[i] = y
    xs = xs.view(-1, *input_size)
    ys = ys.view(-1)
    reps = reps.view(-1, output_size)
    return reps.to('cpu'), xs.to('cpu'), ys.to('cpu')


def outlier(l1_norm_list):
    consistency_constant = 1.4826
    median = torch.median(l1_norm_list)
    mad = consistency_constant * torch.median(torch.abs(l1_norm_list - median)) / 0.6745
    min_mad = torch.abs(torch.min(l1_norm_list) - median) / mad

    return median.item(), mad.item(), min_mad.item()
