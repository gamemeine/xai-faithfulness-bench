import torch


def ground_truth_recovery(attributions, gt_mask):
    if attributions.shape != gt_mask.shape:
        raise ValueError("Attributions and GT masks must have the same shape.")

    importance = attributions.abs().flatten(start_dim=1)
    mask = gt_mask.flatten(start_dim=1)
    total = importance.sum(dim=1)
    recovered = (importance * mask).sum(dim=1)
    return recovered / total.clamp_min(torch.finfo(importance.dtype).tiny)
