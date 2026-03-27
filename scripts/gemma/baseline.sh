#!/bin/bash
# Gemma-2-2B-IT — baseline (no steering)
# Usage: bash scripts/gemma/baseline.sh [advbench|strongreject]
set -e
cd "$(dirname "$0")/../.."

DATASET=${1:-advbench}
case $DATASET in
  advbench)    DATASET_CFG="configs/datasets/advbench_250.yml" ;;
  strongreject) DATASET_CFG="configs/datasets/strongreject_313.yml" ;;
  *) echo "Unknown dataset: $DATASET. Use advbench or strongreject"; exit 1 ;;
esac

python main.py --config configs/base.yml $DATASET_CFG \
  configs/models/gemma/gemma2_2b_it.yml \
  configs/pipelines/baseline.yml
