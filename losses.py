import torch
import torch.nn as  nn
import torch.nn.functional as F

ce = nn.BCELoss()


def unetLoss(preds, real):
    targets = real
    targets = F.one_hot(targets, 2).squeeze(1)
    targets = targets.permute(0, 3, 1, 2).float()
    ce_loss = ce(preds, targets)
    pred_id = torch.argmax(preds, 1).unsqueeze(1)
    count = torch.sum(torch.eq(pred_id, real)).item()
    acc = (count / (real.numel()))
    return ce_loss, acc
