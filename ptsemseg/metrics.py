# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
# 2020-10-17 添加 mask 变量

import numpy as np


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        # 这样的用来算混淆矩阵也是挺有意思的
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds, mask=None):
        confusion_matrix_now = np.zeros((label_trues.shape[0], 2, 2))
        if mask is None:
            mask = np.ones_like(label_trues).astype(np.bool)

        for ii, (lt, lp, lm) in enumerate(zip(label_trues, label_preds, mask)):
            a = lt[lm].flatten()
            b = lp[lm].flatten()
            confusion_matrix_now[ii, :, :] = self._fast_hist(a, b, self.n_classes)
            confusion_matrix_now[ii, :, :] /= confusion_matrix_now[ii, :, :].sum(axis=1, keepdims=True)
            self.confusion_matrix += self._fast_hist(
                a, b, self.n_classes
            )
        return confusion_matrix_now 

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavIoU
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        # acc_cls = np.nanmean(acc_cls)   #mean accuracy
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))  # iou
        
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        # fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        fwavIoU = (freq * iu).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))   #这个写法也很有意思

        cls_pre = np.diag(hist) / hist.sum(axis=1)
        cls_rec = np.diag(hist) / hist.sum(axis=0)

        cls_f1 = 2 * cls_pre * cls_rec / (cls_pre + cls_rec)
        mean_f1 = np.nanmean(cls_f1)

        return (
            {
                "Overall_Acc": acc,
                "Acc": acc_cls,
                "FreqW_IoU": fwavIoU,
                "Mean_IoU": mean_iu,
                "Mean_F1": mean_f1,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))



class averageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

