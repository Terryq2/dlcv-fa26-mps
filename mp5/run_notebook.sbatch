#!/bin/bash

### Do not change the following 8 lines.
#SBATCH --job-name=mp5_notebook
#SBATCH --account=bfdu-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=48g
### End Do not change block

#SBATCH --mail-user=NETID@illinois.edu  # replace with your email
#SBATCH --mail-type=NONE                # Type of email notifications to send

#SBATCH --output=ece494mp5-%x-%j.out  # Default name of batch job output file if not set on command line
#SBATCH --error=ece494mp5-%x-%j.err   # Default name of batch job error file if not set on command line

# Runs a notebook from top to bottom on a GPU, e.g. the Q3 experiments:
#   sbatch run_notebook.sbatch Q3_flow_editing.ipynb
# The figures are saved to outputs/, and the notebook with all its outputs to Q3_flow_editing_run.ipynb.
NOTEBOOK=${1:-Q3_flow_editing.ipynb}

module load python/3.10.14

# Activate your venv
source /work/nvme/bfdu/cs444/fa26/mp5/venv/bin/activate

# print some diagnostics
echo "SLURM_NODELIST: $SLURM_NODELIST"
nvidia-smi || true

# Run the notebook (with the kernel of this venv, whatever kernel the notebook was saved with)
jupyter nbconvert --to notebook --execute "$NOTEBOOK" --output "$(basename "$NOTEBOOK" .ipynb)_run" \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 || exit 1  # a failed cell fails the job
echo "Done: figures in outputs/, notebook with outputs in $(basename "$NOTEBOOK" .ipynb)_run.ipynb"
