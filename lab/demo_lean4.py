from lean_dojo_v2.database import DynamicDatabase

url = "git@github.com:fbundle/lean4-example.git"
commit = "v1.0"

database = DynamicDatabase()

database.trace_repository(
    url=url,
    commit=commit,
    build_deps=False,
)