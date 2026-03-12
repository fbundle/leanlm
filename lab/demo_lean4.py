from lean_dojo import LeanGitRepo, trace

repo = LeanGitRepo(
    "git@github.com:fbundle/lean4-example.git",
    "v1.0",
)

print(repo.get_config("lean-toolchain"))


traced_repo = trace(repo)