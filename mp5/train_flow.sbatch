#!/bin/bash

### Do not change the following 8 lines.
#SBATCH --job-name=flow_afhq
#SBATCH --account=bfdu-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=48g
### End Do not change block

#SBATCH --mail-user=NETID@illinois.edu  # replace with your email
#SBATCH --mail-type=NONE                # Type of email notifications to send

#SBATCH --output=ece494mp5-%x-%j.out  # Default name of batch job output file if not set on command line
#SBATCH --error=ece494mp5-%x-%j.err   # Default name of batch job error file if not set on command line

# Checkpoints, sample grids and the log (log.txt) of this run go to RUN_DIR. Every job starts a new run: if RUN_DIR
# already has a run, train_flow.py uses RUN_DIR_2, RUN_DIR_3, ... instead (the job output says which). Use a different
# RUN_DIR for every experiment. Arguments after the name of this script are passed on to train_flow.py, e.g.
#   sbatch --export=ALL,RUN_DIR=runs/lr3e-4 train_flow.sbatch --lr 3e-4
# To continue a run that stopped early, or to train it for longer, add --resume (and a larger --max_steps):
#   sbatch --export=ALL,RUN_DIR=runs/lr3e-4 train_flow.sbatch --resume --max_steps 20000
RUN_DIR=${RUN_DIR:="runs/afhq"}
MAX_MINUTES=${MAX_MINUTES:=40}

module load python/3.10.14

# Activate your venv
source /work/nvme/bfdu/cs444/fa26/mp5/venv/bin/activate

# print some diagnostics
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
export PYTHONUNBUFFERED=1

# Run your job
python train_flow.py --run_dir ${RUN_DIR} --max_minutes ${MAX_MINUTES} \
    --data_path /work/nvme/bfdu/cs444/fa26/mp5/data/afhq64.pt "$@"
