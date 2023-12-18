# Commands

1) copy model.onnx from /data/model.onnx to model\_repository/onnx-clothing/1/model.onnx

2) if you do not have it you might want to either `dvc pull` or `poetry run train`

3) run `docker-compose up --build` to launch server

4) run perf\_analyzer via docker if you want to analyze performance

5) run main.py (while server is running) to make several requests and test inference

# Analisys

## System:
- Ubuntu 20.04.6 LTS
- CPU: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz
- vCPU: 8
- RAM: 8GB + 6GB of swap memory, docker ran with shm\_size=4GB

## Task:
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

## Structure:
```bash
model_repository
└── onnx-clothing
    ├── 1
    │   └── model.onnx
    └── config.pbtxt
```

## Before optimization:

### Config:

max batch size: 1

dynamic batching: disabled

inference groups: 1 model

### Results:

Batchsize | Throughput | Latency | Time in queue | Time in infer
1024 | 11131.3 | 717 | 264 | 51

## Process of selection:

My model is very small, so it is hard to test it's limits without running into limits of simply transmitting data via HTTP. First of all, though, I enabled dynamic batching (default)

Therefore, the first thing I wanted to do is to test the influence of batch size on Throughput and Latency. Using perf\_analyzer I tried sending with concurrency 8 different data (28x28 images):

Batchsize | Throughput | Latency | Time in queue | Time in infer
1 | 11671.9 | 684 | 272 | 49
4 | 10716.1 | 746 | 286 | 48
16 | 10654.6 | 749 | 272 | 47
64 | 13210.4 | 604 | 210 | 38
1024 | 16485.3 | 484 | 191 | 30
65536 | 16430.7 | 486 | 193 | 30
1048576 | 16564.2 | 482 | 191 | 30
67108864 | 16460.7 | 485 | 203 | 32

The best batch size seems to be around 1024. However, at this point it is obvious that something is wrong - batch size does not have big influence. Let's try to test with different max batch arguments:

Maxbatchsize | Batchsize | Throughput | Latency | Time in queue | Time in infer
8 | 1 | 14215.8 | 561 | 85 | 52
128 | 1 | 12677.3 | 630 | 94 | 63
8 | 64 | 19052.9 | 419 | 74 | 44
128 | 64 | 13155.9 | 607 | 97 | 61
8 | 1024 | 18763.7 | 425 | 88 | 52
128 | 1024 | 18452.1 | 434 | 81 | 58
1024 | 1024 | 18906.8 | 422 | 87 | 53

At this point I start getting messages from perf\_analyzer that it "is not able to keep up with the desired load. The results may not be accurate", so I assume that batch size of 1024 is good enough, and maximum batch size should be bigger than one. To determine best max batch size I tried to look at other parameters, such as "p99 latency" and "latency standard deviation"
Maxbatchsize | p99 latency | Latency standard deviation
8 | 1230 | 325
128 | 1100 | 272
1024 | 974 | 291

However results vary very much from test to test, so it is hard to really understand, what is better. In the end, I decided to go with 1024/1024

Now I will try to tweak dynamic batching and instance groups:

Dynamic batching delay | Instance groups | Throughput | Latency | Time in queue | Time in infer
no dynamic | 1 | 11422.6 | 699 | 253 | 49
10 | 1 | 17987.3 | 444 | 107 | 45
100 | 1 | 17146.4 | 465 | 136 | 50
1000 | 1 | 5702.58 | 1400 | 992 | 91

So it is obviously better to have dynamic batching, but due to the fact that model is very small we do not need big batches - model is running fast anyway. Now, for the instance groups:

Dynamic batching delay | Instance groups | Throughput | Latency | Time in queue | Time in infer
10 | 1 | 17987.3 | 444 | 107 | 45
10 | 2 | 9711.58 | 822 | 195 | 120
10 | 4 | 6508.62 | 1228 | 175 | 241

So, interestingly enough, it is more effective to use only 1 model. I supose, it is because I use CPU, and 1 model is alrerady multithreaded (checked via htop)

## After optimization:

### Config:

max batch size: 1024

dynamic batching: max\_queue\_delay\_microseconds: 10

inference groups: 1 model

### Results:

Batchsize | Throughput | Latency | Time in queue | Time in infer
1024 | 17372.3 | 459 | 107 | 47

## Conclusion

Even thogh I managed to increase the throughput from 11k to 17k (x1.5), I think the results are far from perfect and the analysis is bad because my model is too small. Optimizing model itself, it seems, was not the main problem
