import torch


def eval_knn(x_train, y_train, x_test, y_test, x_test_t, y_test_t, target_label, k=5):
    """ k-nearest neighbors classifier accuracy """
    d = torch.cdist(x_test, x_train)
    topk = torch.topk(d, k=k, dim=1, largest=False)
    labels = y_train[topk.indices]
    pred = torch.empty_like(y_test)
    for i in range(len(labels)):
        x = labels[i].unique(return_counts=True)
        pred[i] = x[0][x[1].argmax()]
    acc = (pred == y_test).float().mean().cpu().item()

    d_t = torch.cdist(x_test_t, x_train)
    topk_t = torch.topk(d_t, k=k, dim=1, largest=False)
    labels_t = y_train[topk_t.indices]
    pred_t = torch.empty_like(y_test_t)
    for i in range(len(labels_t)):
        x = labels_t[i].unique(return_counts=True)
        pred_t[i] = x[0][x[1].argmax()]
    asr = (pred_t == target_label).float().mean().cpu().item()
    acc_t = (pred_t == y_test_t).float().mean().cpu().item()
    del d, topk, labels, pred, d_t, topk_t, labels_t, pred_t
    return acc, acc_t, asr