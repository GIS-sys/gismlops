1) copy model.onnx from /data/model.onnx to model\_repository/onnx-clothing/1/model.onnx

2\*) if you do not have it you might want to either `dvc pull` or `poetry run train`

3) run `docker-compose up --build` to launch server

4) run perf\_analyzer via docker if you want to analyze performance

5) ???
