from captum.attr import NoiseTunnel, Saliency


def smoothgrad_squared(model, specs, target=1, n_samples=50, stdev=0.1, batch_size=8):
    model.eval()
    attributions = NoiseTunnel(Saliency(model)).attribute(
        specs,
        target=target,
        abs=False,
        nt_type="smoothgrad_sq",
        nt_samples=n_samples,
        nt_samples_batch_size=batch_size,
        stdevs=stdev,
    )
    return attributions.detach()
