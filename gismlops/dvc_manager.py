from dvc.repo import Repo


def dvcSave():
    repo = Repo(".")
    repo.add("data")
    repo.commit()
    repo.push()


def dvcLoad():
    repo = Repo(".")
    repo.pull()
