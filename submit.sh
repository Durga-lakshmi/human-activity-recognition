#!/bin/bash -l

# ======================================
# Usage:
#   ./submit.sh <SWEEP_YAML_PATH>
#
# Example:
#   ./submit.sh sweeps/stage2_lstmconv.yaml
# ======================================

if [ $# -ne 1 ]; then
    echo "Usage: $0 <SWEEP_YAML_PATH>"
    exit 1
fi

SWEEP_YAML="$1"
PROJECT_ROOT="/home/RUS_CIP/st196114/dl-lab-25w-team03/human_activity"

if [ ! -f "$SWEEP_YAML" ]; then
    echo "Error: sweep yaml not found: $SWEEP_YAML"
    exit 1
fi

sbatch <<EOT
#!/bin/bash -l
#SBATCH --job-name=ha_wandb_sweep
#SBATCH --output=ha_wandb_sweep-%j.out
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -e  

cd "$PROJECT_ROOT"

echo "===== ENV CHECK ====="
which python
python --version
which wandb
wandb --version

echo "===== CREATE SWEEP ====="
SWEEP_OUTPUT=\$(wandb sweep "$SWEEP_YAML" 2>&1)


echo "\$SWEEP_OUTPUT"


#    "wandb: Run sweep agent with: wandb agent ENTITY/PROJECT/SWEEP_ID"
SWEEP_ID=\$(echo "\$SWEEP_OUTPUT" | sed -n 's/.*Run sweep agent with: wandb agent[[:space:]]*//p')


#    "wandb: Creating sweep with ID: 4fpqpsxj"
if [ -z "\$SWEEP_ID" ]; then
    SWEEP_ID=\$(echo "\$SWEEP_OUTPUT" | sed -n 's/.*Creating sweep with ID:[[:space:]]*//p')
fi

if [ -z "\$SWEEP_ID" ]; then
    echo "ERROR: Failed to parse sweep ID"
    exit 1
fi

echo "===== START WANDB AGENT ====="
echo "Sweep ID: \$SWEEP_ID"

wandb agent "\$SWEEP_ID"
EOT


#--- to run
#./submit.sh sweeps/cnn_tcn_new.yaml


