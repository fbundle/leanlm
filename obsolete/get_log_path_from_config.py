import sys
import json

def get_log_path(config_path: str) -> str:
    return json.loads(open(config_path).read())["train"]["log_path"]

if __name__ == "__main__":
    config_path = sys.argv[1]
    print(get_log_path(config_path))