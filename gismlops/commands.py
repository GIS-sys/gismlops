from gismlops.dvc_manager import dvc_load, dvc_save
from gismlops.infer import infer
from gismlops.train import train


def _infer():
    dvc_load()
    infer()
    dvc_save()


def _train():
    train()
    dvc_save()


if __name__ == "__main__":
    _train()
    _infer()
