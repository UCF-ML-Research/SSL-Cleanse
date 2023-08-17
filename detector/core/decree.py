import datetime
import logging
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from sklearn.cluster import KMeans

import const
from cfg import get_cfg
from datasets import get_ds
from methods import get_method
from utils import eval_knn, get_data, draw, outlier

if __name__ == "__main__":
    cfg = get_cfg()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s => %(message)s')
    now = datetime.datetime.now()
    os.makedirs(f"./result/detection/{cfg.dataset}", exist_ok=True)
    log_filename = os.path.join(f"./result/detection/{cfg.dataset}", now.strftime('%Y-%m-%d %H-%M-%S') + '.log')
    file_handler = logging.FileHandler(filename=log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s => %(message)s')
    formatter.datefmt = '%Y-%m-%d %H:%M:%S'
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

    logging.info(f'Parameters: dataset={cfg.dataset}, num_clusters={cfg.num_clusters}, ratio={cfg.ratio}, '
                 f'attack_succ_threshold={cfg.attack_succ_threshold}, target_center={cfg.target_center}, '
                 f'fname={cfg.fname}, knn_center={cfg.knn_center}')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = get_ds(cfg.dataset)(cfg.num_workers, cfg.test_file_path)
    encoder = get_method(cfg.method)(cfg)
    encoder.to(device).eval()
    encoder.load_state_dict(torch.load(cfg.fname, map_location=torch.device(device)))
    for param in encoder.parameters():
        param.requires_grad = False

    if device == "cuda":
        cudnn.benchmark = True
    # dataset info
    mean, std, width = None, None, None
    if cfg.dataset == "imagenet":
        mean, std = const.IMAGENET_MEAN, const.IMAGENET_STD
        width = const.IMAGENET_WIDTH
    if cfg.dataset == "cifar10":
        mean, std = const.CIFAR10_MEAN, const.CIFAR10_STD
        width = const.CIFAR10_WIDTH

    with torch.no_grad():
        rep, x, y_true = get_data(device, encoder, ds.dataloader_init(cfg.ratio), width)
        KMeans = KMeans(n_clusters=cfg.num_clusters, random_state=0, n_init=30).fit(rep)
        y = KMeans.labels_

        cluster_purities = {}
        first_label = {}
        second_label = {}
        counts_label = {}
        for i in range(np.unique(y).shape[0]):
            mask = (y == i)
            cluster_labels = y_true[mask]

            values, counts = torch.unique(cluster_labels, return_counts=True)
            if values.shape[0] == 1:
                first, second = values.item(), "None"
            else:
                second, first = values[torch.argsort(counts)][-2:]
                second, first = second.item(), first.item()
            first_label[i] = first
            second_label[i] = second
            cluster_purity = torch.sum(cluster_labels == first) / len(cluster_labels)
            cluster_purities[i] = cluster_purity.item()
            counts_label[i] = mask.sum()
        summ = 0
        for i in cluster_purities.keys():
            summ += cluster_purities[i]
        info_str = f"overall cluster purity: {summ / np.unique(y).shape[0]:.2f}, " \
                   f"min num: {min(counts_label.values())}, max num: {max(counts_label.values())}, " \
                   f"total num: {sum(counts_label.values())}"
        logging.info(info_str)

        rep_center = torch.empty((len(np.unique(y)), rep.shape[1]))
        y_center = torch.empty(len(np.unique(y)))
        for label in np.unique(y):
            rep_center[label, :] = rep[y == label].mean(dim=0)
            y_center[label] = label
        if cfg.knn_center:
            rep_knn, y_knn = rep_center, y_center
        else:
            rep_knn, y_knn = rep, torch.tensor(y)
    reg_best_list = torch.empty(len(np.unique(y)))
    for target in np.unique(y):
        logging.info(f"cluster label: {target}, 1st label: {first_label[target]}, "
                     f"2nd label: {second_label[target]}, cluster num: {counts_label[target]}, "
                     f"purity: {cluster_purities[target]:.2f}")
        rep_target = rep[y == target]
        x_other = x[y != target]
        x_other_indices = torch.randperm(x_other.shape[0])[:x.shape[0] - max(counts_label.values())]
        x_other_sample = x_other[x_other_indices]

        mask = torch.arctanh((torch.rand([1, 1, width, width]) - 0.5) * 2).to(device)
        delta = torch.arctanh((torch.rand([1, 3, width, width]) - 0.5) * 2).to(device)

        mask.requires_grad = True
        delta.requires_grad = True
        opt = optim.Adam([delta, mask], lr=cfg.lr, betas=(0.5, 0.9))

        reg_best = torch.inf
        early_stop_reg_best = torch.inf

        lam = 0
        early_stop_counter = 0
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False

        if cfg.target_center:
            dataloader_train = ds.dataloader_cluster_fix(rep_center[target], x_other_sample, cfg.bs)
        else:
            dataloader_train = ds.dataloader_cluster(rep_target, x_other_sample, cfg.bs)
        for ep in range(cfg.epoch):
            loss_asr_list, loss_reg_list, loss_list = [], [], []
            for n_iter, (images, target_reps) in enumerate(dataloader_train):
                images = images.to(device)
                target_reps = target_reps.to(device)
                mask_tanh = torch.tanh(mask) / 2 + 0.5
                delta_tanh = torch.tanh(delta) / 2 + 0.5
                X_R = draw(images, mean, std, mask_tanh, delta_tanh)
                z = target_reps
                zt = encoder.model(X_R)
                loss_asr = encoder.loss_f(z, zt)
                loss_reg = torch.mean(mask_tanh)
                loss = loss_asr + lam * loss_reg
                opt.zero_grad()
                loss.backward(retain_graph=True)
                opt.step()

                loss_asr_list.append(loss_asr.item())
                loss_reg_list.append(loss_reg.item())
                loss_list.append(loss.item())

            avg_loss_asr = torch.tensor(loss_asr_list).mean()
            avg_loss_reg = torch.tensor(loss_reg_list).mean()
            avg_loss = torch.tensor(loss_list).mean()

            x_trigger = draw(x.to(device), mean, std, mask_tanh, delta_tanh).detach().to('cpu')
            dataloader_eval = ds.dataloader_knn(x_trigger, cfg.knn_sample_num)
            asr_knn = eval_knn(device, encoder, dataloader_eval, rep_knn, y_knn, target)
            if asr_knn > cfg.attack_succ_threshold and avg_loss_reg < reg_best:
                mask_best = mask_tanh
                delta_best = delta_tanh
                reg_best = avg_loss_reg

            logging.info('step: %3d, lam: %.2E, asr: %.3f, loss: %f, ce: %f, reg: %f, reg_best: %f' %
                         (ep, lam, asr_knn, avg_loss, avg_loss_asr, avg_loss_reg, reg_best))

            # check early stop
            if cfg.early_stop:
                if cost_down_flag and cost_up_flag:
                    if reg_best < torch.inf:
                        if reg_best >= cfg.early_stop_threshold * early_stop_reg_best:
                            early_stop_counter += 1
                        else:
                            early_stop_counter = 0
                    early_stop_reg_best = min(reg_best, early_stop_reg_best)

                    if early_stop_counter >= cfg.early_stop_patience:
                        logging.info('early stop')
                        break

                elif ep == cfg.start_early_stop_patience and (lam == 0 or reg_best == torch.inf):
                    logging.info('early stop')
                    break

            if lam == 0 and asr_knn >= cfg.attack_succ_threshold:
                cost_set_counter += 1
                if cost_set_counter >= cfg.patience:
                    lam = cfg.lam
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
                    logging.info('initialize cost to %.2E' % lam)
            else:
                cost_set_counter = 0

            if asr_knn >= cfg.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if lam != 0 and cost_up_counter >= cfg.patience:
                cost_up_counter = 0
                logging.info('up cost from %.2E to %.2E' % (lam, lam * cfg.lam_multiplier_up))
                lam *= cfg.lam_multiplier_up
                cost_up_flag = True
            elif lam != 0 and cost_down_counter >= cfg.patience:
                cost_down_counter = 0
                logging.info('down cost from %.2E to %.2E' % (lam, lam / cfg.lam_multiplier_up))
                lam /= cfg.lam_multiplier_up
                cost_down_flag = True


        reg_best_list[target] = reg_best if reg_best != torch.inf else 1

    os.makedirs(cfg.trigger_path, exist_ok=True)
    torch.save({'mask': mask_best, 'delta': delta_best}, os.path.join(cfg.trigger_path, f'{target}.pth'))

    logging.info(f'reg best list: {reg_best_list}')
    median, mad, min_mad = outlier(reg_best_list)
    logging.info(f'median: {median:.2f}, MAD: {mad:.2f}, anomaly index: {min_mad:.2f}')