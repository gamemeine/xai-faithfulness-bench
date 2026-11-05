import argparse
import json

import torch
from tqdm import tqdm

from xai_bench.attributions.grad_cam import grad_cam
from xai_bench.attributions.integrated_gradients import integrated_gradients
from xai_bench.attributions.occlusion import occlusion
from xai_bench.attributions.smoothgrad_squared import smoothgrad_squared
from xai_bench.config import AttributionConfig
from xai_bench.data.dataset import PlantedAudioDataset
from xai_bench.metrics.deletion import deletion_curve
from xai_bench.metrics.ground_truth import ground_truth_recovery
from xai_bench.metrics.insertion import insertion_curve
from xai_bench.models.resnet import ResNet18Detector
from xai_bench.utils import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/attributions.yaml")
    args = parser.parse_args()
    cfg = AttributionConfig.from_yaml(args.config)
    torch.manual_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device}")

    with open(cfg.processed_dir / "metadata.json", encoding="utf-8") as file:
        metadata = json.load(file)
    model = ResNet18Detector.load_from_checkpoint(cfg.checkpoint_path, map_location=device).to(device).eval()
    dataset = PlantedAudioDataset(cfg.processed_dir / "test.pt")

    train = torch.load(cfg.processed_dir / "train.pt", map_location="cpu", weights_only=True)["specs"]
    noise_mean = train.mean(dim=(0, 2)).reshape(1, 1, -1, 1).to(device)
    noise_std = train.std(dim=(0, 2)).reshape(1, 1, -1, 1).to(device)
    del train

    # Use CPU generators for MPS, which does not support explicit generators.
    rng_device = torch.device("cpu") if device.type == "mps" else device
    noise_rng = torch.Generator(device=rng_device).manual_seed(cfg.seed)
    map_rng = torch.Generator(device=rng_device).manual_seed(cfg.seed + 1)
    selection_rng = torch.Generator().manual_seed(cfg.seed)
    positive_indices = torch.where(dataset.labels == 1)[0]
    selected = positive_indices[torch.randperm(len(positive_indices), generator=selection_rng)[: cfg.num_clips]]
    if len(selected) == 0:
        raise ValueError("No positive test clips selected.")
    fractions = torch.linspace(0, 1, cfg.perturbation.steps + 1, device=device)
    records, saved_maps = [], {}

    for idx in tqdm(selected, desc="Evaluating attributions", unit="clip"):
        sample = dataset[int(idx)]
        spec = sample["spec"].unsqueeze(0).to(device)
        mask = sample["gt_mask"].unsqueeze(0).to(device)
        ig, delta = integrated_gradients(
            model,
            spec,
            metadata["normalized_silence_value"],
            n_steps=cfg.integrated_gradients.steps,
            internal_batch_size=cfg.integrated_gradients.batch_size,
        )
        sg = smoothgrad_squared(
            model,
            spec,
            n_samples=cfg.smoothgrad_squared.samples,
            stdev=cfg.smoothgrad_squared.stdev,
            batch_size=cfg.smoothgrad_squared.batch_size,
        )
        maps = {
            "IG": ig,
            "SG-SQ": sg,
            "Grad-CAM": grad_cam(model, spec, layer=cfg.grad_cam.layer),
            "Occlusion": occlusion(
                model,
                spec,
                metadata["normalized_silence_value"],
                window=cfg.occlusion.window,
                stride=cfg.occlusion.stride,
                batch_size=cfg.occlusion.batch_size,
            ),
            "Random": torch.rand(spec.shape, device=rng_device, generator=map_rng).to(device),
        }
        fills = {
            "silence": torch.full_like(spec, metadata["normalized_silence_value"]),
            "mean": torch.full_like(spec, metadata["normalized_mean_value"]),
            "noise": noise_mean
            + noise_std * torch.randn(spec.shape, device=rng_device, generator=noise_rng).to(device),
        }
        saved_maps[sample["clip_id"]] = {name: values.cpu() for name, values in maps.items()}

        for method, attributions in maps.items():
            record = {
                "clip_id": sample["clip_id"],
                "method": method,
                "gt_recovery": ground_truth_recovery(attributions, mask).item(),
                "gt_area_fraction": mask.mean().item(),
            }
            if method == "IG":
                record["ig_delta"] = delta.item()
            for fill_name, fill in fills.items():
                for metric_name, metric in [("deletion", deletion_curve), ("insertion", insertion_curve)]:
                    curve, auc = metric(model, spec, attributions, fill, fractions)
                    record[f"{metric_name}_{fill_name}_curve"] = curve[0].cpu().tolist()
                    record[f"{metric_name}_{fill_name}_auc"] = auc.item()
            records.append(record)

    metrics = ["gt_recovery"] + [
        f"{metric}_{fill}_auc" for metric in ("deletion", "insertion") for fill in ("silence", "mean", "noise")
    ]
    summary = {}
    for method in maps:
        rows = [row for row in records if row["method"] == method]
        summary[method] = {metric: sum(row[metric] for row in rows) / len(rows) for metric in metrics}

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "config": cfg.model_dump(mode="json"),
        "num_clips": len(selected),
        "target": "artifact logit for attributions, artifact probability for deletion and insertion",
        "ranking": "descending absolute attribution",
        "noise": "independent Gaussian in normalized log-mel space, per-bin training mean and std",
        "fractions": fractions.cpu().tolist(),
        "summary": summary,
        "records": records,
    }
    with open(cfg.output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    torch.save(
        {"maps": saved_maps, "noise_mean": noise_mean.cpu(), "noise_std": noise_std.cpu()},
        cfg.output_dir / "attributions.pt",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
