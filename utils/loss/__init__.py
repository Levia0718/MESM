import torch.nn as nn
import torch


def multilabel_categorical_crossentropy_hard_label(y_true, y_pred):
    # https://kexue.fm/archives/7359
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e30
    y_pred_pos = y_pred - (1 - y_true) * 1e30
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat((y_pred_pos, zeros), dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss


def multilabel_categorical_crossentropy_labelsmooth_soft_label(y_true, y_pred):
    # https://kexue.fm/archives/7359 and https://kexue.fm/archives/9064
    y_true = y_true.float()
    infinity = 1e+12
    epsilon = 0.1
    y_mask = y_pred > -infinity / 10
    n_mask = (y_true < 1 - epsilon) & y_mask
    p_mask = (y_true > epsilon) & y_mask
    y_true = torch.clip(y_true, epsilon, 1 - epsilon)
    infs = torch.zeros_like(y_pred) + infinity
    y_neg = torch.where(n_mask, y_pred, -infs) + torch.log(1 - y_true)
    y_pos = torch.where(p_mask, -y_pred, -infs) + torch.log(y_true)
    zeros = torch.zeros_like(y_pred[..., :1])
    y_neg = torch.cat([y_neg, zeros], dim=-1)
    y_pos = torch.cat([y_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pos, dim=-1)
    return neg_loss + pos_loss


class ZLPR_loss(nn.Module):
    def __init__(self, mode='hard_label'):
        super().__init__()
        self.mode = mode

    def forward(self, logits, labels):
        if self.mode == 'hard_label':
            loss = multilabel_categorical_crossentropy_hard_label(labels, logits)
        elif self.mode == 'soft_label':
            loss = multilabel_categorical_crossentropy_labelsmooth_soft_label(labels, logits)
        loss = loss.mean()
        return loss
