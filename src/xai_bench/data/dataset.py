from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset


class PlantedAudioDataset(Dataset):
    def __init__(self, pt_path: Path) -> None:
        payload = torch.load(pt_path, weights_only=True)
        self.specs: torch.Tensor = payload["specs"]
        self.clean_specs: torch.Tensor = payload["clean_specs"]
        self.labels: torch.Tensor = payload["labels"]
        self.gt_masks: torch.Tensor = payload["gt_masks"]
        self.clip_ids: list[str] = payload["clip_ids"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        return {
            "spec": self.specs[idx].unsqueeze(0),  # (1, n_mels, n_frames)
            "clean_spec": self.clean_specs[idx].unsqueeze(0),
            "label": self.labels[idx],
            "gt_mask": self.gt_masks[idx].unsqueeze(0),
            "clip_id": self.clip_ids[idx],
        }


class PlantedAudioDataModule(L.LightningDataModule):
    def __init__(self, processed_dir: Path, batch_size: int = 32, num_workers: int = 4) -> None:
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self.train_ds = PlantedAudioDataset(self.processed_dir / "train.pt")
            self.val_ds = PlantedAudioDataset(self.processed_dir / "val.pt")
        if stage in ("test", None):
            self.test_ds = PlantedAudioDataset(self.processed_dir / "test.pt")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
