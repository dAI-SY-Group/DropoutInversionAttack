import torch

def MSE(img_batch, ref_batch, reduction = True):
    if reduction:
        value = torch.nn.MSELoss(size_average=None, reduction='mean')(img_batch, ref_batch)
    else:
        value = torch.nn.MSELoss(size_average=None, reduction='none')(img_batch, ref_batch)
        for _ in range(len(value.shape)-1):
            value = value.mean(dim=-1)
    return value