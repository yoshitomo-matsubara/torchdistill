import torch


class SegEvaluator(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum() * 100.0
        acc = torch.diag(h) / h.sum(1) * 100.0
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h)) * 100.0
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return 'mean IoU: {:.1f}, IoU: {}, Global pixelwise acc: {:.1f}, Average row correct: {}'.format(
            iu.mean().item(), ['{:.1f}'.format(i) for i in iu.tolist()],
            acc_global.item(), ['{:.1f}'.format(i) for i in acc.tolist()]
        )
