import torch
import torch.nn.functional as F

true_masks = torch.ones((2, 2)).long()
true_masks2 = torch.zeros((2, 2)).long()
ans = torch.eq(true_masks, true_masks)
print(true_masks.numel())