from xai_bench.metrics.perturbation import perturbation_curve


def insertion_curve(model, specs, attributions, fill_values, fractions, target=1, batch_size=8):
    return perturbation_curve(model, specs, attributions, fill_values, fractions, target, batch_size, insertion=True)
