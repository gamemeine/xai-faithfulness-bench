import argparse
import json
from pathlib import Path

import numpy as np
import torch

from xai_bench.config import PrepareDataConfig
from xai_bench.data.artifacts import LogMelExtractor, inject_artifact, load_fixed_waveform


def build_split(
    paths: list[Path],
    cfg: PrepareDataConfig,
    extractor: LogMelExtractor,
    rng: np.random.Generator,
) -> dict[str, torch.Tensor | list[str]]:
    target_len = int(cfg.audio.clip_duration_sec * cfg.audio.sample_rate)
    specs, clean_specs, labels, gt_masks, clip_ids = [], [], [], [], []

    for i, path in enumerate(paths):
        waveform = load_fixed_waveform(path, target_len)
        clean_spec = extractor(waveform)
        label = int(i % 2)  # Balanced 50/50: 0 = clean, 1 = planted artifact

        if label == 1:
            planted_wav, gt_mask = inject_artifact(waveform, cfg.audio, cfg.artifact, extractor, rng)
            spec = extractor(planted_wav)
        else:
            spec = clean_spec.clone()
            gt_mask = torch.zeros_like(clean_spec)

        specs.append(spec)
        clean_specs.append(clean_spec)
        labels.append(label)
        gt_masks.append(gt_mask)
        clip_ids.append(path.stem)

    return {
        "specs": torch.stack(specs),
        "clean_specs": torch.stack(clean_specs),
        "labels": torch.tensor(labels, dtype=torch.long),
        "gt_masks": torch.stack(gt_masks),
        "clip_ids": clip_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate planted ground truth dataset from clean bonafide audio.")
    parser.add_argument("--config", type=str, default="configs/prepare_data.yaml")
    args = parser.parse_args()

    cfg = PrepareDataConfig.from_yaml(args.config)
    rng = np.random.default_rng(cfg.seed)
    extractor = LogMelExtractor(cfg.audio)

    split_limits = {
        "train": cfg.splits.max_train_clips,
        "val": cfg.splits.max_val_clips,
        "test": cfg.splits.max_test_clips,
    }

    raw_splits = {}
    for split_name, max_clips in split_limits.items():
        subset_dir = cfg.paths.raw_audio_dir / split_name
        all_paths = sorted(subset_dir.glob("*.flac"))
        if not all_paths:
            raise FileNotFoundError(f"No .flac files found in {subset_dir}")

        chosen_indices = rng.permutation(len(all_paths))[:max_clips]
        paths = [all_paths[i] for i in chosen_indices]
        raw_splits[split_name] = build_split(paths, cfg, extractor, rng)

    # Compute normalization statistics strictly on the training set
    train_mean = float(raw_splits["train"]["specs"].mean().item())
    train_std = float(raw_splits["train"]["specs"].std().item()) + 1e-8
    silence_norm_val = (extractor.raw_silence_value - train_mean) / train_std

    out_dir = cfg.paths.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, data in raw_splits.items():
        data["specs"] = (data["specs"] - train_mean) / train_std
        data["clean_specs"] = (data["clean_specs"] - train_mean) / train_std
        torch.save(data, out_dir / f"{split_name}.pt")

    metadata = {
        "artifact_type": cfg.artifact.artifact_type,
        "snr_db": cfg.artifact.snr_db,
        "train_mean": train_mean,
        "train_std": train_std,
        "raw_silence_value": extractor.raw_silence_value,
        "normalized_silence_value": silence_norm_val,
        "normalized_mean_value": 0.0,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated dataset in: {out_dir}")
    print(f"Calculated normalized silence value: {silence_norm_val:.4f}")


if __name__ == "__main__":
    main()
