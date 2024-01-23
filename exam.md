# Explanation

## [Pull request from latest version to version before homework checking](???)

## HW1

### poetry

poetry lock; poetry install;

works in [this action from last month](https://github.com/GIS-sys/gismlops/actions/runs/7251817894/job/20437854633) and [this action from hw1](https://github.com/GIS-sys/gismlops/actions/runs/6383320462)

### pre-commit

???

works in [this action from last month](https://github.com/GIS-sys/gismlops/actions/runs/7251817901/job/19754958960) and [this action from hw1](https://github.com/GIS-sys/gismlops/actions/runs/6383320460)

### codestyle

in this final version (as well as in [hw1 commit](https://github.com/GIS-sys/gismlops/pull/3/files#diff-6df7f67034c615d6e99806772a59741a66f7d7a0def25f8970d3e00aab4a4500)):

- train and infer are separate files
- single letter variables are not used
- "gismlops" is a camelCase, although in the very first message from 08.10.23 the snake\_case was required
- all scripts are run from "\_\_main\_\_" function

## HW2

### dvc

???

### hydra

???

### logging

???

### inference

???

### codestyle

1. as of the final version, the only commented code is in ./gismlops/conf/config.yaml (for cuda devices option)
2. as of the final version, no directory is empty (except for triton/model\_repository/onnx-clothing/1/)
3. ???
4. ???
5. ???
6. ???
7. there are no binary files in github, and the only folder being pushed in dvc is data/ (also .logs, but not automatically)

## HW3

### triton

### .md


# Rules (quote)

## ДЗ-1:
- использован и правильно сконфигурирован poetry: 5 баллов
- использован pre-commit, и все файлы в репозитории прошли проверку pre-commit-ом: 10 баллов
- выполнены соглашения по коду и файлам, в том числе наличие отдельных train.py, infer.py, неиспользование однобуквенных переменных, имя репозитория в camel case, в скриптах запуск под if name == main, итд: 10 баллов

## ДЗ-2: 
- установлен и сконфигурирован dvc: 10 баллов
- используется hydra (и используется по назначению): 10 баллов
- есть логгирование по условиям дз: 5 баллов
- есть инференс по условиям дз: 10 баллов
- общее впечатление от качества кода: 20 баллов

В последнем пункте подчеркну самые популярные ошибки:
1. Закомментированый код (наверное топ-1 по встречаемости)
2. Пустые директории
3. Копипаста кода (речь не про плагиат)
4. Использование файлов init.py не по назначению. Были даже люди, которые половину от всего кода написали в init.py
5. Названия .py файлов не соответствуют содержанию
6. Захардкоженные значения в коде, которые следовало вынести в hydra
7. Некоторые люди загружали в dvc файлы с кодом. А некоторые вообще коммитили в гит бинари, в том числе pycache и прочее


## ДЗ-3:
- правильно сконфигурированный тритон, который запускается: 10 баллов
- правильно составленный markdown отчет: 10 баллов
- доп-задание на tensorrt: 10 баллов
- допзадание на арифметическую интенсивность: 10 баллов
