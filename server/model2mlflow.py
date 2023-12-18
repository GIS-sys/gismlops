import os
import shutil

import mlflow.pyfunc
import onnx


def build_server():
    model_path = f"{os.getcwd()}/data/model.onnx"
    directory_out = f"{os.getcwd()}/data/server/"
    try:
        shutil.rmtree(directory_out)
    except FileNotFoundError:
        pass
    onnx_model = onnx.load(model_path)
    mlflow.onnx.save_model(onnx_model, directory_out)


if __name__ == "__main__":
    build_server()
