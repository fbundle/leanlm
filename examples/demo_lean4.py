from lean_dojo import *

repo = LeanGitRepo(
    "https://github.com/leanprover-community/mathlib4",
    "29dcec074de168ac2bf835a77ef68bbe069194c5",
)

print(repo.get_config("lean-toolchain"))


traced_repo = trace(repo)