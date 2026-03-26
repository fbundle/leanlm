from lean_dojo_v2.database import DynamicDatabase

url = "git@github.com:fbundle/lean4-example.git"
commit = "master"

database = DynamicDatabase()

repo = database.trace_repository(
    url=url,
    commit=commit,
    build_deps=True,
)

if repo is None:
    raise RuntimeError()

for theorem in repo.proven_theorems:
    print(theorem.theorem_statement)