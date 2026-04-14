import os
import sys

from dotenv import load_dotenv

job_template = """
#!/usr/bin/env bash

#PBS -P {project}
#PBS -N log_{recipe}
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=23:59:59
#PBS -j oe

cd $PBS_O_WORKDIR
mkdir -p log

module load cuda/12.6.2

while true; do
    nvidia-smi > log/gpu_{qwen35_4b_lora}.log
    sleep 5
done &

export HF_HOME="$HOME/scratch/hf_home"
MAMBA="$HOME/miniforge3/condabin/mamba"
MAMBA_ENV="test"

$MAMBA run -n $MAMBA_ENV uv run \
    python -m leanlm.recipes.{recipe} train |& tee log/run_{recipe}.log
"""

def main(recipe: str):
    load_dotenv()

    job = job_template.format(project=os.environ["PBS_PROJECT"], recipe=recipe)

    job_dir = "mnt/job"
    job_file = f"{job_dir}/job_{recipe}.pbs"
    os.makedirs(job_dir, exist_ok=True)
    with open(job_file, "w") as f:
        f.write(job)

    os.system(f"qsub {job_file}")



if __name__ == "__main__":
    main(sys.argv[1])