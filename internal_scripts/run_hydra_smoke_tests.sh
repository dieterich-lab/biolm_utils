#!/bin/bash

# Run BioLM Hydra configuration smoke tests via Slurm job.
#SBATCH --job-name=biolm-hydra-smoke
#SBATCH --output=/prj/RNA_NLP/biolm_utils/internal_outputs/slurm/hydra_smoke_tests.log
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00

set -euo pipefail

cd /prj/RNA_NLP/biolm_utils

export POETRY_VIRTUALENVS_IN_PROJECT=true

if [ ! -d .venv ]; then
  poetry install --no-interaction --with dev
fi

source .venv/bin/activate

echo "Using Python: $(command -v python)"
echo "Running Hydra configuration smoke tests..."

# Run only the Hydra smoke tests
poetry run pytest tests/test_hydra_config_smoke.py -v

echo "Hydra smoke tests completed successfully!"
