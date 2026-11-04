from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class PathsConfig(BaseModel):
    raw_audio_dir: Path
    output_dir: Path


class SplitsConfig(BaseModel):
    max_train_clips: int = 1000
    max_val_clips: int = 200
    max_test_clips: int = 200


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    clip_duration_sec: float = 4.0
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 80
    eps_floor: float = 1e-6


class ArtifactConfig(BaseModel):
    artifact_type: Literal["sine_tone", "impulse_click", "narrowband_noise", "bandstop"]
    snr_db: float = 0.0
    duration_sec: float = 0.6
    freq_low_hz: float = 1500.0
    freq_high_hz: float = 3000.0


class PrepareDataConfig(BaseModel):
    seed: int = 42
    paths: PathsConfig
    splits: SplitsConfig
    audio: AudioConfig
    artifact: ArtifactConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PrepareDataConfig":
        with open(path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))


class DataLoaderConfig(BaseModel):
    processed_dir: Path
    batch_size: int = 32
    num_workers: int = 4


class ModelOptimConfig(BaseModel):
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


class TrainerRunConfig(BaseModel):
    max_epochs: int = 15
    log_dir: Path = Path("tb_logs")
    experiment_name: str = "resnet18_narrowband_0db"
    checkpoint_dir: Path = Path("checkpoints/asvspoof/resnet18_narrowband_0db")


class TrainConfig(BaseModel):
    seed: int = 42
    data: DataLoaderConfig
    model: ModelOptimConfig
    trainer: TrainerRunConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with open(path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))
