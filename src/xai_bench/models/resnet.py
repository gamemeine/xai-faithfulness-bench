import lightning as L
import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC


class ResNet18Detector(L.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.fc = nn.Linear(backbone.fc.in_features, 2)
        self.net = backbone

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.val_auroc = BinaryAUROC()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self(batch["spec"])
        labels = batch["label"]
        loss = self.criterion(logits, labels)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        return loss, probs, labels

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, probs, labels = self._step(batch)
        batch_size = labels.size(0)
        self.train_acc(probs, labels)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, probs, labels = self._step(batch)
        batch_size = labels.size(0)
        self.val_acc(probs, labels)
        self.val_auroc(probs, labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, logger=False, batch_size=batch_size)
        self.log("val/auroc", self.val_auroc, on_step=False, on_epoch=True, batch_size=batch_size)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
