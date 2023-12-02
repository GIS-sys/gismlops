import git


def git_version():
    repo = git.Repo(search_parent_directories=True)
    print(repo.head)
    return repo.head.object.hexsha
