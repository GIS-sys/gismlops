import os

import lightning.pytorch as pl


def configure_loggers_and_callbacks(cfg):
    if not cfg.artifacts.enable_logger:
        return [], []
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
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(max_depth=cfg.callbacks.model_summary.max_depth),
    ]
    return loggers, callbacks
