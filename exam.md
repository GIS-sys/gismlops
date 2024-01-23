# Explanation

## Important links

hw1 commit: 37c31df0dbffec876ecce231ede22256d369a7e0

hw2 commit: e7cdd42497847948ec4fccf6fe247a57f2681964

hw3 commit: 8c62fa06464580d0795bd91fe65fe9383172241e

## HW1

### poetry

poetry lock; poetry install;

works in [this action from last month](https://github.com/GIS-sys/gismlops/actions/runs/7251817894/job/20437854633) and [this action from hw1](https://github.com/GIS-sys/gismlops/actions/runs/6383320462)

### pre-commit

conda install pre-commit; pre-commit install; git commit -m "pre-commit test"

works in [this action from last month](https://github.com/GIS-sys/gismlops/actions/runs/7251817901/job/19754958960) and [this action from hw1](https://github.com/GIS-sys/gismlops/actions/runs/6383320460)

### codestyle

in this final version (as well as in [hw1 commit](https://github.com/GIS-sys/gismlops/pull/3/files#diff-6df7f67034c615d6e99806772a59741a66f7d7a0def25f8970d3e00aab4a4500)):

- train and infer are separate files
- single letter variables are not used
- "gismlops" is a camelCase, although in the very first message from 08.10.23 the snake\_case was required
- all scripts are run from "\_\_main\_\_" function

## HW2

### dvc

dvc init; dvc remote add -d storage gdrive://1Z3JfbS00SLrhHPVh7igikSy4Dbbug-z; dvc add data/; dvc commit; dvc push; dvc pull;

works in current version, but I admit that it didn't work in hw3 version, and wasn't present in hw2 version

### hydra

used in train, infer (hydra.main) and test (initialize and compose), both in hw2 and current

### logging

cd tracker-service/; docker-compose up;

log in at http://gismlops.mlflow using admin/password

export MLFLOW_TRACKING_USERNAME=admin; export MLFLOW_TRACKING_PASSWORD=password; poetry run train;

in utils I use `tags={"commit": git\_version()}`, and log hyperparameters and lr-SGD, train\_loss, val\_loss\_epoch, val\_loss\_step

### inference

export to .onnx is done in train, inference: ./run_server.sh; python server/test.py;

### codestyle

1. as of the final version, the only commented code is in ./gismlops/conf/config.yaml (for cuda devices option and mlflow\_tracking\_uri)
2. as of the final version, no directory is empty (except for triton/model\_repository/onnx-clothing/1/)
3. common parts are either written in utils or configs
4. don't have code in \_\_init\_\_ files
5. commands.py for poetry commands, data.py for dataloader, dvc\_manager.py and git\_manager.py for dvc and git, model.py for Model, utils.py for common code, train.py and infer.py for train and inference
6. the only hardcoded values are: data/ folder in data.py and log folders in utils.py
7. there are no binary files in github, and the only folder being pushed in dvc is data/ (also .logs, but not automatically)

## HW3

### triton

cp data/model.onnx triton/model_repository/onnx-clothing/1/; cd triton/; docker-compose up --build;

docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:23.04-py3-sdk; perf_analyzer -m onnx-clothing -u localhost:8500 --concurrency-range 1:5 --shape inputs:1,1,28,28 --shape predictions:1,10;

python triton/client.py;

but I admit that I can't find the examples in dvc storage, because I deleted it when I tried to understand the error

### .md

research description is in triton/README.md



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




# Codestyle explanation

- Readme exists
- No variables are defined at the top level
- Only one file (commands.py) has more than 1 function under \_\_main\_\_. It has 2 calls, \train() and \_infer(), I really felt like not creating separate function for that
- warnings are not suppressed
- real data from mnist-like dataset
- imports are made global-style
- did not save binaries to git
- has 2 files
- "gismlops" is snake\_case
- used .gitignore for python as core
- have run pre-commit
- files are named in snake\_case
- repository is public



# Codestyle (quote)

По итогу просмотра первой домашней работы были выявлены типичные ошибки (список ниже).
Для успешного выполнения второго задания нужно их исправить.
Для повышения качества жизни проверяющих (и получения отличной оценки) рекомендуется также выполнить пункты из раздела “желательно”.

Что нужно исправить:
* Заполнить файл README, в нём объяснить, какую задачу вы решаете (у каждого что-то своё, так что нам нужно это знать)
* Нельзя объявлять переменные (кроме констант) на верхнем уровне файла (не внутри функции или класса). Это можно делать только внутри классов, функций или под if name main.
* Под вызовом if name main вызывать ровно одну функцию (можно её назвать main или как-то ещё), а не писать всю логику непосредственно под if-ом
* warnings.filterwarnings("ignore") - это исчадье сатаны. Никогда не делайте этого в продакшен любых проектах. Это огромный задел на отстреливание себе ноги. Люди пишут предостережения для вас, но вы же умнее каких-то там авторов библиотек!
* Используйте реальные данные. Нельзя использовать сгенерированные данные.
* Импорты из вашего проекта делайте либо локальными (через точку, например как тут (https://github.com/v-goncharenko/mimics/blob/master/mimics/classifiers.py#L16)), либо глобальными (когда начинаете с названия вашего пакета, вам это пока скорее всего не нужно, проще первый вариант)
* Нельзя сохранять данные в гит!!!!!!!!!!!!! То есть файлы .json, .csv, .h5 и проч. То же касается файлов натренированных моделей (.cbm, .pth, .xyz, etc). Я об этом говорил на лекциях, более того, сейчас вы даже знаете, как это делать правильно (через dvc). Нужно удалить все данные из гита.
* Некоторым с первого раза не понятно: ДВА ФАЙЛА - train.py и infer.py, не всё в одном
* Назвать питон пакет (aka папка с вашим кодом) по правилам питона (snake\_case), а не как попало (e.g. MYopsTools)
* Используйте дефолтный .gitignore для Питона (не пустой), его дополняйте необходимыми вам путями. Дефолтный конфиг гуглится и даже предлагается вам гитхабом при создании репозитория.
* Запустить таки pre-commit run -a и убедиться, что он не красный
* Файлы с кодом называются в snake\_case, не в CamelCase (e.g. Dataset.py)
* Репозиторий должен быть виден. Скрытые (приватные) репозитории оцениваются в 0 баллов.

Что делать желательно:
* Использовать fire вместо argparse
* Гит репозитории как правило называются в log-case (то есть слова разделяются дефисами), а не в CamelCase
* Сделать одну входную точку commands.py, где вызывать соответствующие функции из файлов
* Пересесть с иглы процедурного программирования (когда вы объявляете только функции) на ООП (aka классы).
* Для выполнения предыдущего пункта сначала нужно осмелиться создать больше двух файлов с кодом в папке (например вынести туда функции и импортировать их из этого пошеренного файла)
* Не делайте однобуквенные переменные, как будто в школе сидим программируем, ей богу…
* Конфиги лучше хранить не в json, а в yaml (это всё равно понадобится для Гидры)
* Используйте pathlib вместо os.path - важа жизнь заиграет совершенно другими красками
* Никто не называет папку с кодом src в Питоне… пожалуйста, не делайте так, это ничего хорошего в вашу жизнь не добавляет
* Не стоит все ваши функции и классы класть исключительно в utils - это получается какая-то свалка
