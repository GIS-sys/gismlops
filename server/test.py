import json

import numpy as np
import requests


def predict(array):
    URL = "http://localhost:8080/invocations"
    HEADERS = {"Content-Type": "application/json"}
    data = json.dumps({"inputs": array})
    req = requests.Request("POST", URL, data=data, headers=HEADERS).prepare()
    res = requests.Session().send(req)
    return res


def create_test(shape):
    N = shape[0]
    if len(shape) == 1:
        return [0.1 for _ in range(N)]
    return [create_test(shape[1:]) for _ in range(N)]


CLASSES = [
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


def analyze_results(raw):
    try:
        data = json.loads(raw)
    except Exception as e:
        print(f"Exception while converting response to json:\n{str(e)}")
        print(raw)
        return
    if "error_code" in data:
        print("Got error from server:")
        print(data["message"])
        print(data["stack_trace"])
        return
    data = data["predictions"]
    for _, prediction in data.items():
        prediction = np.array(prediction)
        print(f"Result: {prediction}")
        answer_index = np.argmax(prediction)
        print(f"Prediction: {CLASSES[answer_index]}")


test = create_test([1, 1, 28, 28])
raw = predict(test).text
analyze_results(raw)
