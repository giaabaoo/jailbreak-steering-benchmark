#!/bin/bash
#SBATCH --job-name=full313_ang_ref_tools
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/full313_angular_refusal_toolsdir_%j.log
#SBATCH --error=logs/full313_angular_refusal_toolsdir_%j.err

echo "Full-313 Angular Refusal 180 tools-dir prehook | Job $SLURM_JOB_ID | $(date)"
source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
conda activate codec
cd /home/bao.dinh/Research/shadow_steering
nvidia-smi | head -5

python main.py --config configs/base.yml configs/datasets/strongreject_313.yml \
  configs/models/gemma/gemma2_2b_it.yml \
  configs/pipelines/inference_only.yml \
  configs/steering_methods/angular_refusal_180_toolsdir_prehook.yml

echo "DONE. $(date)"
