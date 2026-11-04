import argparse

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from xai_bench.config import TrainConfig
from xai_bench.data.dataset import PlantedAudioDataModule
from xai_bench.models.resnet import ResNet18Detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet-18 audio detector on planted dataset.")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    cfg = TrainConfig.from_yaml(args.config)
    L.seed_everything(cfg.seed, workers=True)

    datamodule = PlantedAudioDataModule(
        processed_dir=cfg.data.processed_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    model = ResNet18Detector(
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
    )

    logger = TensorBoardLogger(
        save_dir=str(cfg.trainer.log_dir),
        name=cfg.trainer.experiment_name,
    )
    logger.log_hyperparams(cfg.model_dump(mode="json"))

    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.trainer.checkpoint_dir,
        filename="best-epoch={epoch:02d}-val_acc={val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_cb],
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(model, datamodule=datamodule)
    print(f"Best checkpoint saved at: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    main()
