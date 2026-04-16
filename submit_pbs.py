import os
import subprocess
import sys
from dotenv import load_dotenv


JOB_TEMPLATE = """
#!/usr/bin/env bash

#PBS -P {project_name}
#PBS -N log_{recipe_name}_{uuid}
#PBS -q normal
#PBS -j oe
#PBS -l {pbs_limit}
#PBS -l walltime=22:50:00

cd $PBS_O_WORKDIR

module load cuda/12.6.2

export HF_HOME="$HOME/scratch/hf_home"

mkdir -p log
while true; do
    nvidia-smi > log/gpu_{recipe_name}_{uuid}.log
    sleep 5
done &

UV="$HOME/miniforge3/envs/test/bin/uv"
$UV run accelerate launch -m {recipe_module} train {uuid} |& tee log/run_{recipe_name}_{uuid}.log
"""

def write_file(path: str, content: str = ""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

def get_module_path(file_path: str) -> str:
    file_path = os.path.relpath(file_path, os.getcwd())
    file_path, _ = os.path.splitext(file_path)
    module_path = file_path.replace("/", ".")
    return module_path

def main(recipe_file: str):
    load_dotenv()

    recipe_module = get_module_path(recipe_file)
    recipe_name = recipe_module.split(".")[-1]

    project_name = os.environ.get("PBS_PROJECT", default=None)
    if project_name is None:
        raise RuntimeError("PBS_PROJECT must be set")

    pbs_limit = os.environ.get("PBS_LIMIT", default=None)
    if pbs_limit is None:
        raise RuntimeError("PBS_LIMIT must be set")

    uuid = pbs_limit.replace("=", "").replace(":", "")

    job_file = f"mnt/job/job_{recipe_name}.pbs"
    write_file(
        path=job_file,
        content=JOB_TEMPLATE.format(
            project_name=project_name,
            recipe_name=recipe_name,
            recipe_module=recipe_module,
            pbs_limit=pbs_limit,
            uuid=uuid,
        ),
    )

    write_file(f"log/run_{recipe_name}_{uuid}.log")
    write_file(f"log/gpu_{recipe_name}_{uuid}.log")

    result = subprocess.run(["qsub", job_file])
    if result.returncode != 0:
        raise RuntimeError(f"qsub failed with code {result.returncode}")



if __name__ == "__main__":
    main(sys.argv[1])
