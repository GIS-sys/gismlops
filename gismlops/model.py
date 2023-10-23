from typing import Any

import lightning.pytorch as pl
import torch
from torch import nn


class MyModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"val_loss": loss}

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_correct", correct, on_step=True, on_epoch=True, prog_bar=True)
        return {"test_loss": loss, "test_correct": correct}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self(batch)

    @staticmethod
    def warmup_wrapper(warmup_steps: int, training_steps: int):
        def warmup(current_step: int):
            if current_step < warmup_steps:
                return float(current_step / warmup_steps)
            else:
                return max(
                    0.0,
                    float(training_steps - current_step)
                    / float(max(1, training_steps - warmup_steps)),
                )

        return warmup

    def configure_optimizers(self) -> Any:
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.cfg["train"]["weight_decay"],
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters,
            lr=self.cfg.train.learning_rate,
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=type(self).warmup_wrapper(
                warmup_steps=self.cfg.train.num_warmup_steps,
                training_steps=self.cfg.train.num_training_steps,
            ),
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(pl.utilities.grad_norm(self, norm_type=2))
        super().on_before_optimizer_step(optimizer)
