import torch
from captum.attr import Occlusion


@torch.no_grad()
def occlusion(model, specs, silence_value, target=1, window=(8, 20), stride=(4, 10), batch_size=8):
    model.eval()
    attributions = Occlusion(model).attribute(
        specs,
        target=target,
        baselines=silence_value,
        sliding_window_shapes=(specs.shape[1], *window),
        strides=(specs.shape[1], *stride),
        perturbations_per_eval=batch_size,
    )
    return attributions.detach()
