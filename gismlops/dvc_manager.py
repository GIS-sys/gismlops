from dvc.repo import Repo


def save():
    repo = Repo(".")
    repo.add("data")
    repo.commit()
    repo.push()


def load():
    repo = Repo(".")
    repo.pull()
