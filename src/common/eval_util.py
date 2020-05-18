import torch


def compute_accuracy(outputs, targets, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        _, preds = outputs.topk(maxk, 1, True, True)
        preds = preds.t()
        corrects = preds.eq(targets[None])
        ressult_list = []
        for k in topk:
            correct_k = corrects[:k].flatten().sum(dtype=torch.float32)
            ressult_list.append(correct_k * (100.0 / batch_size))
        return ressult_list
