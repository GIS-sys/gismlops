import os
import shutil

import mlflow.pyfunc


class PyModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        pass

    def predict(self, context, model_input):
        return model_input["a"] + model_input["b"]


def build_server():
    directory = f"{os.getcwd()}/data/serve/"
    try:
        shutil.rmtree(directory)
    except FileNotFoundError:
        pass
    mlflow.pyfunc.save_model(path=directory, python_model=PyModelWrapper())


def run_server():
    pass


if __name__ == "__main__":
    build_server()
    run_server()
