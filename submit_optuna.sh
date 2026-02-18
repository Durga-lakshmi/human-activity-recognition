#!/bin/bash -l

PROJECT_DIR=/home/RUS_CIP/st196114/dl-lab-25w-team03/human_activity
PYTHON_SCRIPT=optuna_runner.py


mkdir -p "$PROJECT_DIR/optuna/logs"

sbatch <<EOT
#!/bin/bash -l
#SBATCH --job-name=hapt
#SBATCH --output=$PROJECT_DIR/optuna/logs/optuna_%j.out
#SBATCH --error=$PROJECT_DIR/optuna/logs/optuna_%j.err

#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00

cd $PROJECT_DIR

# activate venv
source venv/bin/activate

# turn off wandb for optuna runs
export WANDB_MODE=disabled
export WANDB_SILENT=true
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

which python
python --version

python $PYTHON_SCRIPT
EOT
