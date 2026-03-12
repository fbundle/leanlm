from lean_dojo import *

repo = LeanGitRepo(
    "https://github.com/leanprover-community/mathlib4",
    "v4.21.0",
)



print(repo.get_config("lean-toolchain"))


traced_repo = trace(repo)