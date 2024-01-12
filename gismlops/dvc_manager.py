from dvc.fs import DVCFileSystem
from dvc.repo import Repo


def dvc_save():
    repo = Repo(".")
    repo.add("data")
    repo.add(".logs")
    repo.commit()
    repo.push()


def dvc_load():
    fs = DVCFileSystem()
    fs.get("data", "data", recursive=True)

