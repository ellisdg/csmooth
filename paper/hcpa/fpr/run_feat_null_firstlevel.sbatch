#!/bin/bash
#SBATCH --job-name=feat_null
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --output=/data2/david.ellis/public/HCPA/cache/fpr/logs/%x_%j.out
#SBATCH --error=/data2/david.ellis/public/HCPA/cache/fpr/logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python /data2/david.ellis/public/HCPA/code/fpr/run_feat_null_firstlevel.py "$@"
