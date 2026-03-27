#!/bin/bash
# Gemma-2-2B-IT — angular refusal only (rotation-based ablation, 90 degrees)
# Usage: bash scripts/gemma/angular_refusal.sh [advbench|strongreject|strongreject_50]
set -e
cd "$(dirname "$0")/../.."

DATASET=${1:-advbench}
case $DATASET in
  advbench)         DATASET_CFG="configs/datasets/advbench_250.yml" ;;
  strongreject)     DATASET_CFG="configs/datasets/strongreject_313.yml" ;;
  strongreject_50)  DATASET_CFG="configs/datasets/strongreject_50.yml" ;;
  *) echo "Unknown dataset: $DATASET"; exit 1 ;;
esac

python main.py --config configs/base.yml $DATASET_CFG \
  configs/models/gemma/gemma2_2b_it.yml \
  configs/pipelines/inference_only.yml \
  configs/steering_methods/angular_refusal_greedy.yml
