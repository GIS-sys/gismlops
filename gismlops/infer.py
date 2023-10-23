import os

import hydra
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from gismlops.data import MyDataModule
from gismlops.model import MyModel
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def infer(cfg: DictConfig):
    pl.seed_everything(cfg.general.seed)
    dm = MyDataModule(
        batch_size=cfg.data.batch_size,
        dataloader_num_wokers=cfg.data.dataloader_num_wokers,
        val_size=cfg.data.val_size,
    )
    model = MyModel(cfg)
    model.load_state_dict(torch.load(cfg.artifacts.model.path))

    os.makedirs("./.logs/my-wandb-logs", exist_ok=True)
    loggers = [
        pl.loggers.CSVLogger("./.logs/my-csv-logs", name=cfg.artifacts.experiment_name),
        pl.loggers.MLFlowLogger(
            experiment_name=cfg.artifacts.experiment_name,
            tracking_uri="file:./.logs/my-mlflow-logs",
        ),
        pl.loggers.WandbLogger(
            project="mlops-logging-demo",
            name=cfg.artifacts.experiment_name,
            save_dir="./.logs/my-wandb-logs",
        ),
    ]

    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        precision=cfg.train.precision,
        max_steps=cfg.train.num_warmup_steps + cfg.train.num_training_steps,
        accumulate_grad_batches=cfg.train.grad_accum_steps,
        val_check_interval=cfg.train.val_check_interval,
        overfit_batches=cfg.train.overfit_batches,
        num_sanity_val_steps=cfg.train.num_sanity_val_steps,
        deterministic=cfg.train.full_deterministic_mode,
        benchmark=cfg.train.benchmark,
        gradient_clip_val=cfg.train.gradient_clip_val,
        profiler=cfg.train.profiler,
        log_every_n_steps=cfg.train.log_every_n_steps,
        detect_anomaly=cfg.train.detect_anomaly,
        enable_checkpointing=cfg.artifacts.checkpoint.use,
        logger=loggers,
    )

    answers = np.concatenate(trainer.predict(model, datamodule=dm), axis=1).T

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    answersDataFrame = pd.DataFrame(answers, columns=["target_index", "predicted_index"])
    answersDataFrame["target_label"] = answersDataFrame["target_index"].map(
        lambda x: classes[x]
    )
    answersDataFrame["predicted_label"] = answersDataFrame["predicted_index"].map(
        lambda x: classes[x]
    )
    answersDataFrame.to_csv("data/test.csv", index=False)
    return answersDataFrame


if __name__ == "__main__":
    infer()
