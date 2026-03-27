#!/bin/bash
# Llama-3.1-8B-Instruct — random text direction instead of SAE feature (ablation)
export CUDA_VISIBLE_DEVICES=0
PYTHONPATH='.':$PYTHONPATH \
python main.py \
  --config configs/base.yml \
           configs/datasets/harmful_prompts.yml \
           configs/models/llama/llama3_8b_inst.yml \
           configs/pipelines/inference_only.yml \
           configs/steering_methods/random_feature.yml
