# Commands

1) copy model.onnx from /data/model.onnx to model\_repository/onnx-clothing/1/model.onnx

2) if you do not have it you might want to either `dvc pull` or `poetry run train`

3) run `docker-compose up --build` to launch server

4) run perf\_analyzer via docker if you want to analyze performance

5) run main.py (while server is running) to make several requests and test inference

# Analisys

System:
- Ubuntu 20.04.6 LTS
- CPU: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz
- vCPU: 8
- RAM: 8GB + 6GB of swap memory

Task:
Given the 28x28 black and white image of cloth predict one of 10 possible categories:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

model_repository
└── onnx-clothing
    ├── 1
    │   └── model.onnx
    └── config.pbtxt

Before optimization:
???

After optimization:
???

Explanation:
???
