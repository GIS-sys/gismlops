from gismlops.dvc_manager import dvc_load, dvc_save
from gismlops.infer import infer
from gismlops.server import build_server, start_server
from gismlops.train import train


def _infer():
    dvc_load()
    infer()
    dvc_save()


def _train():
    train()
    dvc_save()


def _build_server():
    build_server()


def _start_server():
    start_server()


def _run_server():
    build_server()
    start_server()


if __name__ == "__main__":
    _train()
    _infer()
