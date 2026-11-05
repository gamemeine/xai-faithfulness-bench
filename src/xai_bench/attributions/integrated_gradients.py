import torch
from captum.attr import IntegratedGradients


def integrated_gradients(model, specs, silence_value, target=1, n_steps=50, internal_batch_size=8):
    model.eval()
    baseline = torch.full_like(specs, silence_value)

    attributions, delta = IntegratedGradients(model).attribute(
        specs,
        baselines=baseline,
        target=target,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        return_convergence_delta=True,
    )
    return attributions.detach(), delta.detach()
