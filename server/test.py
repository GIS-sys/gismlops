import json

import requests


def predict(array):
    URL = "http://localhost:8080/invocations"
    HEADERS = {"Content-Type": "application/json"}
    data = json.dumps({"inputs": array})
    req = requests.Request("POST", URL, data=data, headers=HEADERS).prepare()
    res = requests.Session().send(req)
    return res


print(predict([[[[1.0, 2.0]]]]).text)
