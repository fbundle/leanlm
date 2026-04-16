import os
import subprocess
import sys
import re
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
    nvidia-smi > log/gpu_{recipe_name}_{uuid}.log
    sleep 5
done &

export HF_HOME="$HOME/scratch/hf_home"
UV="$HOME/miniforge3/envs/test/bin/uv"

# Multi-node configuration
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
export MASTER_PORT=29500
export NNODES={nnodes}
export GPUS_PER_NODE={ngpus}
export WORLD_SIZE=$(($NNODES * $GPUS_PER_NODE))

# Launch using mpiexec to ensure one process per node
mpiexec --hostfile $PBS_NODEFILE -n $NNODES -npernode 1 \
    $UV run accelerate launch \
    --multi_gpu \
    --num_nodes $NNODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $OMPI_COMM_WORLD_RANK \
    -m {recipe_module} train {uuid} |& tee log/run_{recipe_name}_{uuid}.log
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

def parse_pbs_limit(pbs_limit: str):
    select_match = re.search(r"select=(\d+)", pbs_limit)
    ngpus_match = re.search(r"ngpus=(\d+)", pbs_limit)
    
    nnodes = int(select_match.group(1)) if select_match else 1
    ngpus = int(ngpus_match.group(1)) if ngpus_match else 1
    
    return nnodes, ngpus

def main(recipe_file: str):
    load_dotenv()

    recipe_module = get_module_path(recipe_file)
    recipe_name = recipe_module.split(".")[-1]

    project_name = os.environ.get("PBS_PROJECT", default=None)
    if project_name is None:
        raise RuntimeError("PBS_PROJECT must be set")

    pbs_limit = os.environ.get("PBS_LIMIT", default="select=1:ngpus=1")
    nnodes, ngpus = parse_pbs_limit(pbs_limit)

    job_file = f"mnt/job/job_{recipe_name}.pbs"
    
    write_file(
        path=job_file,
        content=JOB_TEMPLATE.format(
            project_name=project_name,
            recipe_name=recipe_name,
            recipe_module=recipe_module,
            pbs_limit=pbs_limit,
            nnodes=nnodes,
            ngpus=ngpus,
            uuid=pbs_limit.replace("=", "").replace(":", ""),
        ),
    )

    write_file(f"log/run_{recipe_name}.log")
    write_file(f"log/gpu_{recipe_name}.log")

    print(f"Submitting multi-node job for {recipe_name} ({nnodes} nodes, {ngpus} GPUs/node)...")
    result = subprocess.run(["qsub", job_file])
    if result.returncode != 0:
        raise RuntimeError(f"qsub failed with code {result.returncode}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python submit_pbs_multi_node.py <recipe_file>")
        sys.exit(1)
    main(sys.argv[1])
