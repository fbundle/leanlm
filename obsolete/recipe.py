from typing import Any
import os
import json
from .dtype import *

def load_recipe(path: str) -> Config:
    o = json.loads(open(path).read())
    if not isinstance(o, dict):
        raise RuntimeError("ERROR: config format")
    if "version" not in o:
        raise RuntimeError("ERROR: config format")
    
    if o["version"] == 1:
        configV1 = ConfigV1.model_validate(o)
        # convert ConfigV1 to Config
        config = Config(
            version=configV1.version,
            data=configV1.data,
            model=configV1.model,
            train=configV1.train,
        )
        return config


    else:
        raise RuntimeError(f"ERROR: recipe version {path}")