import os
import subprocess
import sys

from dotenv import load_dotenv

job_template = """
#!/usr/bin/env bash

#PBS -P {project_name}
#PBS -N log_{recipe_name}
#PBS -q normal
#PBS -l select=1:ngpus=4
#PBS -l walltime=23:59:59
#PBS -j oe

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

def main(recipe_file: str):
    load_dotenv()

    recipe_file = get_relative_path(recipe_file)
    recipe_path, _ = os.path.splitext(recipe_file)

    recipe_module = recipe_path.replace("/", ".")
    recipe_name = os.path.basename(recipe_path)

    project_name = os.environ["PBS_PROJECT"]

    job_file = f"mnt/job/job_{recipe_name}.pbs"
    write_file(
        path=job_file,
        content=job_template.format(
            project_name=project_name,
            recipe_name=recipe_name,
            recipe_module=recipe_module,
        ),
    )

    write_file(f"log/run_{recipe_name}.log")
    write_file(f"log/gpu_{recipe_name}.log")

    result = subprocess.run(["qsub", job_file])
    if result.returncode != 0:
        raise RuntimeError(f"qsub failed with code {result.returncode}")



if __name__ == "__main__":
    main(sys.argv[1])
