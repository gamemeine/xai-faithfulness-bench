import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from xai_bench.config import PrepareDataConfig
from xai_bench.data.artifacts import LogMelExtractor, inject_artifact, load_fixed_waveform


def main() -> None:
    parser = argparse.ArgumentParser(description="Export random audio sample pairs (clean vs planted) to WAV.")
    parser.add_argument("--config", type=str, default="configs/prepare_data.yaml")
    parser.add_argument("--num-samples", type=int, default=3)
    args = parser.parse_args()

    cfg = PrepareDataConfig.from_yaml(args.config)
    rng = np.random.default_rng(cfg.seed)
    extractor = LogMelExtractor(cfg.audio)

    target_len = int(cfg.audio.clip_duration_sec * cfg.audio.sample_rate)
    test_files = sorted((cfg.paths.raw_audio_dir / "test").glob("*.flac"))
    chosen_indices = rng.choice(len(test_files), size=args.num_samples, replace=False)

    subfolder_name = f"{cfg.artifact.artifact_type}_{cfg.artifact.snr_db}db"
    out_dir = Path("samples") / subfolder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in chosen_indices:
        flac_path = test_files[idx]
        clean_wav = load_fixed_waveform(flac_path, target_len)
        planted_wav, _ = inject_artifact(clean_wav, cfg.audio, cfg.artifact, extractor, rng)

        sf.write(out_dir / f"{flac_path.stem}_clean.wav", clean_wav.numpy(), cfg.audio.sample_rate)
        sf.write(out_dir / f"{flac_path.stem}_planted.wav", planted_wav.numpy(), cfg.audio.sample_rate)

    print(f"Saved {args.num_samples} sample pairs to: {out_dir}")


if __name__ == "__main__":
    main()
