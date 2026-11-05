import argparse
import json

import torch
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

from xai_bench.config import EvalConfig
from xai_bench.data.dataset import PlantedAudioDataModule
from xai_bench.models.resnet import ResNet18Detector
from xai_bench.utils import get_device


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained ResNet-18 detector and verify artifact usage.")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    cfg = EvalConfig.from_yaml(args.config)
    device = get_device()

    with open(cfg.data.processed_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    silence_val = float(metadata["normalized_silence_value"])

    datamodule = PlantedAudioDataModule(
        processed_dir=cfg.data.processed_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    datamodule.setup(stage="test")

    model = ResNet18Detector.load_from_checkpoint(cfg.evaluation.checkpoint_path, map_location=device)
    model.eval()
    model.to(device)

    acc_metric = BinaryAccuracy().to(device)
    auroc_metric = BinaryAUROC().to(device)

    probs_planted_list, probs_clean_list, probs_masked_list = [], [], []

    for batch in datamodule.test_dataloader():
        specs = batch["spec"].to(device)
        clean_specs = batch["clean_spec"].to(device)
        gt_masks = batch["gt_mask"].to(device)
        labels = batch["label"].to(device)

        probs = torch.softmax(model(specs), dim=-1)[:, 1]
        acc_metric.update(probs, labels)
        auroc_metric.update(probs, labels)

        # Select only positive samples (where the artifact was planted)
        pos_idx = labels == 1
        if pos_idx.any():
            pos_specs = specs[pos_idx]
            pos_clean = clean_specs[pos_idx]
            pos_masks = gt_masks[pos_idx]

            # Replace artifact region with normalized silence value
            masked_specs = pos_specs * (1.0 - pos_masks) + silence_val * pos_masks

            probs_planted_list.append(probs[pos_idx])
            probs_clean_list.append(torch.softmax(model(pos_clean), dim=-1)[:, 1])
            probs_masked_list.append(torch.softmax(model(masked_specs), dim=-1)[:, 1])

    p_planted = torch.cat(probs_planted_list)
    p_clean = torch.cat(probs_clean_list)
    p_masked = torch.cat(probs_masked_list)

    pred_planted = p_planted >= 0.5
    pred_clean = p_clean >= 0.5
    pred_masked = p_masked >= 0.5

    # Flip rate: fraction of positive predictions that flip to class 0 after artifact removal
    detected_pos = pred_planted.sum().clamp(min=1)
    flip_rate_clean = float(((pred_planted & ~pred_clean).sum() / detected_pos).item())
    flip_rate_masked = float(((pred_planted & ~pred_masked).sum() / detected_pos).item())

    results = {
        "test_accuracy": float(acc_metric.compute().item()),
        "test_auroc": float(auroc_metric.compute().item()),
        "mean_prob_with_artifact": float(p_planted.mean().item()),
        "mean_prob_after_clean_removal": float(p_clean.mean().item()),
        "mean_prob_after_silence_masking": float(p_masked.mean().item()),
        "prob_drop_clean_removal": float((p_planted - p_clean).mean().item()),
        "prob_drop_silence_masking": float((p_planted - p_masked).mean().item()),
        "decision_flip_rate_clean_removal": flip_rate_clean,
        "decision_flip_rate_silence_masking": flip_rate_masked,
    }

    cfg.evaluation.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.evaluation.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Evaluation & Artifact Usage Verification ===")
    for k, v in results.items():
        print(f"{k:38s}: {v:.4f}")
    print(f"\nSaved results to: {cfg.evaluation.output_json}")


if __name__ == "__main__":
    main()
