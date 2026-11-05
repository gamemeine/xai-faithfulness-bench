import matplotlib.pyplot as plt
import numpy as np


def plot_example(clean_spec, planted_spec, audio_cfg, gt_mask=None, title=None):
    clean = clean_spec.detach().cpu().squeeze().numpy()
    planted = planted_spec.detach().cpu().squeeze().numpy()
    frame_step = audio_cfg.hop_length / audio_cfg.sample_rate
    times = np.arange(clean.shape[1]) * frame_step
    mel_bins = np.arange(clean.shape[0])
    extent = (-frame_step / 2, times[-1] + frame_step / 2, -0.5, len(mel_bins) - 0.5)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, constrained_layout=True)
    for ax, spec, label in zip(axes, [clean, planted], ["Original", "With artifact"], strict=True):
        image = ax.imshow(
            spec,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="magma",
            vmin=min(clean.min(), planted.min()),
            vmax=max(clean.max(), planted.max()),
        )
        ax.set_title(label)
        ax.set_xlabel("Time [s]")
    fig.colorbar(image, ax=axes, label="Log-mel")
    if gt_mask is not None:
        axes[1].contour(times, mel_bins, gt_mask.detach().cpu().squeeze().numpy(), levels=[0.5], colors="cyan")
    axes[0].set_ylabel("Mel bin index")
    if title:
        fig.suptitle(title)
    plt.show()
    return fig, axes


def plot_perturbation_curves(results, metric):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True, constrained_layout=True)
    for ax, fill in zip(axes, ["silence", "mean", "noise"], strict=True):
        for method in results["summary"]:
            curves = [record[f"{metric}_{fill}_curve"] for record in results["records"] if record["method"] == method]
            ax.plot(results["fractions"], np.mean(curves, axis=0), label=method)
        ax.set_title(f"{metric.title()} ({fill})")
        ax.set_xlabel("Fraction removed" if metric == "deletion" else "Fraction restored")
        ax.set_ylim(0, 1)
        ax.legend()
    axes[0].set_ylabel("Mean P(artifact)")
    plt.show()
    return fig, axes


def plot_attribution(spec, attributions, audio_cfg, gt_mask=None, title=None, method="Attribution", signed=True):
    spec = spec.detach().cpu().squeeze().numpy()
    if not isinstance(attributions, dict):
        attributions = {method: attributions}
    frame_step = audio_cfg.hop_length / audio_cfg.sample_rate
    times = np.arange(spec.shape[1]) * frame_step
    mel_bins = np.arange(spec.shape[0])
    extent = (-frame_step / 2, times[-1] + frame_step / 2, -0.5, len(mel_bins) - 0.5)

    panels = len(attributions) + 1
    fig, axes = plt.subplots(1, panels, figsize=(6 * panels, 4), sharey=True, constrained_layout=True)
    image = axes[0].imshow(spec, origin="lower", aspect="auto", extent=extent, cmap="magma")
    axes[0].set_title("Model input")
    fig.colorbar(image, ax=axes[0], label="Normalized log-mel")
    for ax, (name, values) in zip(axes[1:], attributions.items(), strict=True):
        values = values.detach().cpu().squeeze().numpy()
        limit = max(abs(values).max(), np.finfo(values.dtype).tiny)
        is_signed = signed and (values < 0).any()
        image = ax.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="RdBu_r" if is_signed else "magma",
            vmin=-limit if is_signed else 0,
            vmax=limit,
        )
        ax.set_title(name)
        fig.colorbar(image, ax=ax, label="Signed attribution" if is_signed else "Attribution")
    for ax in axes:
        if gt_mask is not None:
            ax.contour(times, mel_bins, gt_mask.detach().cpu().squeeze().numpy(), levels=[0.5], colors="cyan")
        ax.set_xlabel("Time [s]")
    axes[0].set_ylabel("Mel bin index")
    if title:
        fig.suptitle(title)
    plt.show()
    return fig, axes
