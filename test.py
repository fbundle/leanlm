import dbm
import json
import os



import lmdb
from pydantic import BaseModel
import pydantic_core




from pydantic import BaseModel
from tqdm import tqdm


type JixiaLeanName = list[str]

class JixiaItem(BaseModel):
    name: JixiaLeanName
    kind: str # TODO - check all kinds - change this to Literal
    isProp: bool
    
    typeFallback: str
    typeFull: str
    typeReadable: str

    typeReferences: list[JixiaLeanName]
    valueReferences: list[JixiaLeanName]

def get_name(lean_name: JixiaLeanName) -> str:
    return ".".join(lean_name)


graph_dir = "/Users/khanh/code/mathlib4_dependency_graph/output/mathlib4_dependency_graph/data_5e932f97dd25535344f80f9dd8da3aab83df0fe6"



lean_name_list = []
for name in os.listdir(graph_dir):
    if name.endswith(".sym.json"):
        lean_name = name.rstrip(".sym.json")
        lean_name_list.append(lean_name)


with lmdb.open(".cache", map_size=100 * 1024**3) as env: # max 100GB
    with env.begin(write=True) as txn:
        with open("to_delete.txt", "w") as f:
            for lean_name in tqdm(lean_name_list):
                path = os.path.join(graph_dir, lean_name + ".sym.json")
                to_delete = False
                if os.path.exists(path):
                    o_list = json.load(open(path))
                    for o in o_list:
                        try:
                            i = JixiaItem.model_validate(o)
                            name = get_name(i.name)
                            txn.put(name.encode(), i.model_dump_json().encode())


                        except pydantic_core._pydantic_core.ValidationError as e:
                            # to_delete = True
                            pass
                        
                        if to_delete:
                            break
                else:
                    to_delete = True
                
                if to_delete:
                    # delete file 
                    for tail in [".decl.json", ".elab.json", ".mod.json", ".sym.json"]:
                        path = os.path.join(graph_dir, lean_name + tail)
                        f.write(path + "\n")
