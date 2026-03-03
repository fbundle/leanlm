from lean_dojo import *

repo = LeanGitRepo(
    "https://github.com/leanprover-community/mathlib4",
    "8f9d9cff6bd728b17a24e163c9402775d9e6a365",
)



print(repo.get_config("lean-toolchain"))


traced_repo = trace(repo)