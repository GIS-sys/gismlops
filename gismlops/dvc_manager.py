from dvc.repo import Repo


def dvc_save():
    repo = Repo(".")
    repo.add("data")
    repo.add("logs")
    repo.commit()
    repo.push()


def dvc_load():
    repo = Repo(".")
    repo.pull(force=True)
