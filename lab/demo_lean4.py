from lean_dojo import LeanGitRepo, trace

repo = LeanGitRepo(
    url="git@github.com:fbundle/lean4-example.git",
    commit="master",
)

print(repo.get_config("lean-toolchain"))


traced_repo = trace(repo)