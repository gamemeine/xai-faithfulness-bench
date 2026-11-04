import math
from pathlib import Path

import numpy as np
import scipy.signal as signal
import soundfile as sf
import torch
import torchaudio

from xai_bench.config import ArtifactConfig, AudioConfig


class LogMelExtractor:
    def __init__(self, cfg: AudioConfig) -> None:
        self.cfg = cfg
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            power=2.0,
        )
        self.raw_silence_value = math.log(cfg.eps_floor)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        mel_power = self.mel_transform(waveform)
        return torch.log(torch.clamp(mel_power, min=self.cfg.eps_floor))

    def hz_to_mel_bins(self, low_hz: float, high_hz: float) -> tuple[int, int]:
        fb = torchaudio.functional.melscale_fbanks(
            n_freqs=self.cfg.n_fft // 2 + 1,
            f_min=0.0,
            f_max=self.cfg.sample_rate / 2.0,
            n_mels=self.cfg.n_mels,
            sample_rate=self.cfg.sample_rate,
        ).numpy()
        freqs = np.linspace(0, self.cfg.sample_rate / 2.0, fb.shape[0])
        active_freqs = (freqs >= low_hz) & (freqs <= high_hz)
        mel_energy = fb[active_freqs].sum(axis=0)
        active_mels = np.where(mel_energy > 0.05 * mel_energy.max())[0]
        return int(active_mels.min()), int(active_mels.max())


def inject_artifact(
    waveform: torch.Tensor,
    audio_cfg: AudioConfig,
    art_cfg: ArtifactConfig,
    extractor: LogMelExtractor,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_samples = waveform.shape[-1]
    art_samples = int(art_cfg.duration_sec * audio_cfg.sample_rate)

    margin = int(0.2 * audio_cfg.sample_rate)
    start_sample = int(rng.integers(margin, num_samples - art_samples - margin))
    end_sample = start_sample + art_samples

    speech_rms = torch.sqrt(torch.mean(waveform**2) + 1e-12)
    target_rms = speech_rms * (10.0 ** (art_cfg.snr_db / 20.0))
    modified = waveform.clone()

    raw_art: torch.Tensor | None = None

    if art_cfg.artifact_type == "narrowband_noise":
        white = rng.standard_normal(art_samples)
        sos = signal.butter(
            4,
            [art_cfg.freq_low_hz, art_cfg.freq_high_hz],
            btype="bandpass",
            fs=audio_cfg.sample_rate,
            output="sos",
        )
        raw_art = torch.from_numpy(signal.sosfiltfilt(sos, white).copy()).to(waveform.dtype)
        mel_low, mel_high = extractor.hz_to_mel_bins(art_cfg.freq_low_hz, art_cfg.freq_high_hz)
    elif art_cfg.artifact_type == "sine_tone":
        center_hz = 0.5 * (art_cfg.freq_low_hz + art_cfg.freq_high_hz)
        t = torch.arange(art_samples, dtype=waveform.dtype) / audio_cfg.sample_rate
        raw_art = torch.sin(2.0 * math.pi * center_hz * t)
        mel_low, mel_high = extractor.hz_to_mel_bins(center_hz - 100.0, center_hz + 100.0)
    elif art_cfg.artifact_type == "impulse_click":
        art_samples = int(0.05 * audio_cfg.sample_rate)
        end_sample = start_sample + art_samples
        raw_art = torch.from_numpy(rng.standard_normal(art_samples)).to(waveform.dtype)
        mel_low, mel_high = 0, audio_cfg.n_mels - 1
    elif art_cfg.artifact_type == "bandstop":
        segment = modified[start_sample:end_sample].numpy()
        sos = signal.butter(
            4,
            [art_cfg.freq_low_hz, art_cfg.freq_high_hz],
            btype="bandstop",
            fs=audio_cfg.sample_rate,
            output="sos",
        )
        modified[start_sample:end_sample] = torch.from_numpy(signal.sosfiltfilt(sos, segment).copy()).to(waveform.dtype)
        mel_low, mel_high = extractor.hz_to_mel_bins(art_cfg.freq_low_hz, art_cfg.freq_high_hz)
    else:
        raise ValueError(f"Unsupported artifact_type: {art_cfg.artifact_type}")

    if raw_art is not None:
        art_rms = torch.sqrt(torch.mean(raw_art**2) + 1e-12)
        window = torch.hann_window(art_samples, dtype=waveform.dtype)
        modified[start_sample:end_sample] += raw_art * (target_rms / art_rms) * window

    n_frames = num_samples // audio_cfg.hop_length + 1
    gt_mask = torch.zeros((audio_cfg.n_mels, n_frames), dtype=torch.float32)
    frame_start = start_sample // audio_cfg.hop_length
    frame_end = min(n_frames, math.ceil(end_sample / audio_cfg.hop_length))
    gt_mask[mel_low : mel_high + 1, frame_start:frame_end] = 1.0

    return modified, gt_mask


def load_fixed_waveform(path: Path, target_len: int) -> torch.Tensor:
    wav_np, _ = sf.read(path, dtype="float32")
    waveform = torch.from_numpy(wav_np)
    if waveform.shape[0] < target_len:
        return torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[0]))
    return waveform[:target_len]
