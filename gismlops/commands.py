from gismlops.dvc_manager import dvcLoad, dvcSave
from gismlops.infer import infer
from gismlops.train import train


def _infer():
    dvcLoad()
    infer()


def _train():
    dvcSave()
    train()


if __name__ == "__main__":
    _train()
    _infer()
