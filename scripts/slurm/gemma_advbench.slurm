#!/bin/bash
#SBATCH --job-name=gemma_advbench
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gemma_advbench_%j.log
#SBATCH --error=logs/gemma_advbench_%j.err

echo "Gemma AdvBench | Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | Start $(date)"
source /home/bao.dinh/miniconda3/etc/profile.d/conda.sh
conda activate codec
cd /home/bao.dinh/Research/shadow_steering
nvidia-smi | head -20

echo "=== METHOD 1/3: Baseline ==="
python main.py --config configs/base.yml configs/datasets/advbench_250.yml configs/models/gemma/gemma2_2b_it.yml configs/pipelines/baseline.yml
if [ $? -ne 0 ]; then echo "ERROR: baseline failed!"; exit 1; fi
echo "Baseline done. $(date)"

echo "=== METHOD 2/3: Refusal-only ==="
python main.py --config configs/base.yml configs/datasets/advbench_250.yml configs/models/gemma/gemma2_2b_it.yml configs/pipelines/inference_only.yml configs/steering_methods/refusal_only.yml
if [ $? -ne 0 ]; then echo "ERROR: refusal_only failed!"; exit 1; fi
echo "Refusal-only done. $(date)"

echo "=== METHOD 3/3: SAE Steering ==="
python main.py --config configs/base.yml configs/datasets/advbench_250.yml configs/models/gemma/gemma2_2b_it.yml configs/pipelines/inference_only.yml configs/steering_methods/sae_text_steering.yml
if [ $? -ne 0 ]; then echo "ERROR: sae_text_steering failed!"; exit 1; fi
echo "SAE steering done. $(date)"

echo "ALL GEMMA ADVBENCH DONE. End: $(date)"
