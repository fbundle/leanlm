from lean_dojo import LeanGitRepo, trace

repo = LeanGitRepo(
    "git@github.com:fbundle/lean4-example.git",
    "master",
)

print(repo.get_config("lean-toolchain"))


traced_repo = trace(repo)