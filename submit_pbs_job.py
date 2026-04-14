import os
import subprocess
import sys

from dotenv import load_dotenv

job_template = """
#!/usr/bin/env bash

#PBS -P {project_name}
#PBS -N log_{recipe_name}
#PBS -q normal
#PBS -l select=1:ngpus=1
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


MAMBA_BIN="$HOME/miniforge3/condabin/mamba"
MAMBA_ENV="test"

$MAMBA_BIN run -n $MAMBA_ENV uv run \\ 
    python -m {recipe_module} train |& tee log/run_{recipe_name}.log
"""

def get_relative_path(path: str) -> str:
    return os.path.relpath(path, os.getcwd())

def main(recipe_file: str):
    load_dotenv()

    recipe_file = get_relative_path(recipe_file)
    recipe_path, _ = os.path.splitext(recipe_file)

    recipe_module = recipe_path.replace("/", ".")
    recipe_name = os.path.basename(recipe_path)

    project_name = os.environ["PBS_PROJECT"]

    job = job_template.format(
        project_name=project_name,
        recipe_name=recipe_name,
        recipe_module=recipe_module,
    )

    job_dir = "mnt/job"
    job_file = f"{job_dir}/job_{recipe_name}.pbs"
    os.makedirs(job_dir, exist_ok=True)
    with open(job_file, "w") as f:
        f.write(job)

    os.makedirs("log", exist_ok=True)
    open(f"log/run_{recipe_name}.log", "w").close()
    open(f"log/gpu_{recipe_name}.log", "w").close()

    result = subprocess.run(["qsub", job_file])
    if result.returncode != 0:
        print(f"qsub failed with code {result.returncode}", file=sys.stderr)
        sys.exit(1)



if __name__ == "__main__":
    main(sys.argv[1])