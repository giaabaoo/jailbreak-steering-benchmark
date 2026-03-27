#!/bin/bash
# Llama-3.1-8B-Instruct — refusal direction only (ablation)
# Usage: bash scripts/llama/refusal_only.sh [advbench|strongreject]
set -e
cd "$(dirname "$0")/../.."

DATASET=${1:-advbench}
case $DATASET in
  advbench)     DATASET_CFG="configs/datasets/advbench_250.yml" ;;
  strongreject) DATASET_CFG="configs/datasets/strongreject_313.yml" ;;
  *) echo "Unknown dataset: $DATASET. Use advbench or strongreject"; exit 1 ;;
esac

python main.py --config configs/base.yml $DATASET_CFG \
  configs/models/llama/llama3_8b_inst.yml \
  configs/pipelines/inference_only.yml \
  configs/steering_methods/refusal_only.yml
