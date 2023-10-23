from gismlops.infer import infer
from gismlops.train import train


def _infer():
    infer()


def _train():
    train()


if __name__ == "__main__":
    _train()
    _infer()
