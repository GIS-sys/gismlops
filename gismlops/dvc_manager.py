from dvc.repo import Repo
from dvc.fs import DVCFileSystem


def dvc_save():
    repo = Repo(".")
    repo.add("data")
    repo.add(".logs")
    repo.commit()
    repo.push()


def dvc_load():
    fs = DVCFileSystem()
    fs.get("data", "data", recursive=True)

