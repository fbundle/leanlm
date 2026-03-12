from lean_dojo_v2.database import DynamicDatabase

url = "https://github.com/durant42040/lean4-example"
commit = "005de00d03f1aaa32cb2923d5e3cbaf0b954a192"

database = DynamicDatabase()

database.trace_repository(
    url=url,
    commit=commit,
    build_deps=False,
)