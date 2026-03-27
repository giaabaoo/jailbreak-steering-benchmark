#!/bin/bash
# Score existing results with LlamaGuard + rule-based refusal detection
# Usage: RESULTS_PATH=./results/some/results.json bash scripts/llama/evaluate.sh
export CUDA_VISIBLE_DEVICES=0
PYTHONPATH='.':$PYTHONPATH \
python main.py \
  --config configs/base.yml \
           configs/evaluators/jailbreak_evaluator.yml \
           configs/pipelines/evaluate.yml \
  --pipeline.results_path "${RESULTS_PATH}"
