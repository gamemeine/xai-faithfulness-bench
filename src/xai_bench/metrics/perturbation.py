import torch


@torch.no_grad()
def perturbation_curve(model, specs, attributions, fill_values, fractions, target=1, batch_size=8, insertion=False):
    if specs.shape != attributions.shape:
        raise ValueError("Attributions must have the same shape as model inputs.")

    model.eval()
    fractions = torch.as_tensor(fractions, device=specs.device, dtype=specs.dtype)
    if fractions.ndim != 1 or len(fractions) < 2 or not torch.all(fractions[1:] > fractions[:-1]):
        raise ValueError("Fractions must be a strictly increasing sequence of at least two values.")
    if fractions[0] != 0 or fractions[-1] != 1:
        raise ValueError("Fractions must start at 0 and end at 1.")

    importance = attributions.abs().flatten(start_dim=1)
    order = importance.argsort(dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(order.shape[1], device=specs.device).expand_as(order))
    counts = (fractions * order.shape[1]).round().long()

    selected = (ranks[:, None, :] < counts[None, :, None]).reshape(specs.shape[0], len(fractions), *specs.shape[1:])
    originals = specs.unsqueeze(1)
    fills = torch.broadcast_to(torch.as_tensor(fill_values, device=specs.device, dtype=specs.dtype), specs.shape)
    if insertion:
        perturbed = torch.where(selected, originals, fills.unsqueeze(1))
    else:
        perturbed = torch.where(selected, fills.unsqueeze(1), originals)
    probabilities = torch.cat(
        [model(batch).softmax(dim=-1)[:, target] for batch in perturbed.flatten(end_dim=1).split(batch_size)]
    )
    curve = probabilities.reshape(specs.shape[0], len(fractions))
    auc = torch.trapezoid(curve, fractions, dim=1)
    return curve, auc
