import os
import subprocess
import sys
import importlib
from dotenv import load_dotenv


JOB_TEMPLATE = """
#!/usr/bin/env bash

#PBS -P {project_name}
#PBS -N log_{recipe_name}
#PBS -q normal
#PBS -j oe
#PBS -l {pbs_limit}
#PBS -l walltime=23:59:59

cd $PBS_O_WORKDIR
mkdir -p log

module load cuda/12.6.2

while true; do
    nvidia-smi > log/gpu_{recipe_name}.log
    sleep 5
done &

export HF_HOME="$HOME/scratch/hf_home"

UV="$HOME/miniforge3/envs/test/bin/uv"
$UV run accelerate launch -m {recipe_module} train |& tee log/run_{recipe_name}.log
"""

def get_relative_path(path: str) -> str:
    return os.path.relpath(path, os.getcwd())

def write_file(path: str, content: str = ""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

def get_pbs_limit(recipe_module: str) -> str:
    module = importlib.import_module(recipe_module)
    if not hasattr(module, "PBS_LIMIT"):
        raise RuntimeError("PBS_LIMIT must be set")
    pbs_limit = getattr(module, "PBS_LIMIT")
    if not isinstance(pbs_limit, str):
        raise RuntimeError("PBS_LIMIT must be set")
    return pbs_limit

def main(recipe_file: str):
    load_dotenv()

    

    recipe_file = get_relative_path(recipe_file)
    recipe_path, _ = os.path.splitext(recipe_file)

    recipe_module = recipe_path.replace("/", ".")

    pbs_limit = get_pbs_limit(recipe_module)

    recipe_name = os.path.basename(recipe_path)

    project_name = os.environ["PBS_PROJECT"]

    job_file = f"mnt/job/job_{recipe_name}.pbs"
    write_file(
        path=job_file,
        content=JOB_TEMPLATE.format(
            project_name=project_name,
            recipe_name=recipe_name,
            recipe_module=recipe_module,
            pbs_limit=pbs_limit,
        ),
    )

    write_file(f"log/run_{recipe_name}.log")
    write_file(f"log/gpu_{recipe_name}.log")

    result = subprocess.run(["qsub", job_file])
    if result.returncode != 0:
        raise RuntimeError(f"qsub failed with code {result.returncode}")



if __name__ == "__main__":
    main(sys.argv[1])
