'''
Author: Shuailin Chen
Created Date: 2020-11-27
Last Modified: 2021-03-11
	content: 
'''
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm.std import trange

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.weight = weight
#         self.ignore_index = ignore_index
#         self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

#     def forward(self, preds, labels):
#         logpt = -self.ce_fn(preds, labels)
#         pt = torch.exp(logpt)
#         loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
#         return loss

# def FocalLoss(input, target, alpha=0.5, gamma=2, weight=None, ignore_index=255, size_average=True):
#     ce_fn = nn.CrossEntropyLoss(weight=weight,ignore_index=ignore_index)
#     logpt = -ce_fn(input, target)
#     pt = torch.exp(logpt)
#     loss = -((1-pt)**gamma)*alpha*logpt
#     return loss

def cross_entropy1d(input, target, weight=None):
    ''' cross entropy 1d for image classification 
    @in     -input, target      -as in the official cross_entropy func
    @out the loss value
    last edited: 2020-11-12
    '''
    loss = F.cross_entropy(input, target.long(), weight=torch.tensor(weight).to(input.device))
    return loss


def cross_entropy_with_mask(input, target, mask, weight=None):
    ''' cross entropy with mask 
    @in     -input, target      -as in the official cross_entropy func
    @       -mask               -the mask, 1 indicates include into loss, 0 indecates not
    @out the loss value
    last edited: 2020-10-16
    '''
    loss = F.cross_entropy(input, target, reduction='none', weight=torch.tensor(weight).to(input.device))
    loss = loss[mask].mean()
    # loss = loss.mean()
    return loss


def FocalLoss2d(input, target, alpha=None, gamma=2, reduction='mean'):
    log_softmax = F.log_softmax(input, dim=1)
    re_input = (1-torch.exp(log_softmax))**gamma*log_softmax
    nll_loss = F.nll_loss(re_input, target, weight=alpha, reduction=reduction)
    return nll_loss

def FocalLoss(input, target, alpha=0.5, gamma=2, weight=None, ignore_index=255, size_average=True):
    ''' 这个函数应该是错的 '''
    ce_fn = nn.CrossEntropyLoss(weight=weight,ignore_index=ignore_index)
    logpt = -ce_fn(input, target)
    pt = torch.exp(logpt)
    loss = -((1-pt)**gamma)*alpha*logpt
    return loss

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()
    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        target = target.unsequeeze(1)
        target = F.upsample(target, size=(h, w), mode="nearest")
        target = target.sequeeze(1)
    elif h < ht and w < wt:  # upsample images
        input = F.upsample(input, size=(ht, wt), mode="bilinear")
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss


def multi_scale_cross_entropy2d(input, target, weight=None, size_average=True, scale_weight=None):
    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight == None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp))

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss


def bootstrapped_cross_entropy2d(input,target, K, weight=None, size_average=True):
    batch_size = input.size()[0]
    def _bootstrap_xentropy_single(input, 
                                   target, 
                                   K, 
                                   weight=None,
                                   size_average=True):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(input, 
                               target, 
                               weight=weight, 
                               reduce=False,
                               size_average=False, 
                               ignore_index=250)

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)
