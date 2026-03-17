from lean_dojo_v2.database import DynamicDatabase

url = "git@github.com:fbundle/lean4-example.git"
commit = "master"

database = DynamicDatabase()

database.trace_repository(
    url=url,
    commit=commit,
    build_deps=False,
)

