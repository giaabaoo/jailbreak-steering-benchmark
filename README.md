# Jailbreak Steering Benchmark

An empirical study of activation steering methods for bypassing LLM safety refusals, evaluated on **Gemma-2-2B-IT** using the [StrongREJECT](https://github.com/alexandrasouly/strongreject) benchmark (313 harmful prompts).

We implement and compare refusal direction subtraction, SAE feature steering, and rotation-based (angular) steering from scratch, using a shared config-driven pipeline.

---

## Methods

### 1. Baseline
No steering. Measures the model's natural refusal rate.

### 2. Refusal Direction Methods
The refusal direction `r` is computed as `mean(harmful activations) - mean(harmless activations)` at the residual stream input of a selected layer, using a dataset of contrastive harmful/harmless prompt pairs. We use a precomputed direction from [Arditi et al. (2025)](https://arxiv.org/abs/2406.11717), extracted at layer 15 via pre-hook (residual stream input), last token position.

- **`refusal_only`**: orthogonal projection — removes the refusal component from all layers
- **`refusal_dir_actadd`**: activation addition — `h[15] += -1.0 * r` at a single layer via pre-hook

### 3. SAE Feature Steering
For each harmful prompt, query [Neuronpedia](https://neuronpedia.org) for the top-8 SAE features most active at the last token (GemmaScope layer-15 16k SAE). For each feature, extract the residual-stream activation of its max-activating text as a steering direction `e_i`. Apply via pre-hook at layer 15:

```
h[15] += 0.3 * e_i - 1.0 * r
```

Runs all 8 features per prompt and reports best-of-8 (oracle upper bound).

- **`sae_refdir_prehook`**: SAE max-text direction + refusal subtraction (main method)
- **`sae_refdir_prehook_nosub`**: SAE direction only, no refusal subtraction

### 4. Angular Steering
Rotation-based steering adapted from [Winninger et al. (2025)](https://arxiv.org/abs/2510.26243). Instead of linearly adding/subtracting vectors, we rotate the residual stream activation by a target angle in a 2D subspace. This is norm-preserving.

- **`angular_refusal_180`**: rotate `h` by 180° in the refusal direction (flip) — equivalent to `h -= 2*(h·r̂)*r̂`
- **`angular_sae_maxtext_refusal_150`**: rotate in the 2D plane `{r, e_i}` by 150°
- **`angular_sae_150_refusal_180_refusal_first`**: double hook — flip refusal 180° then rotate SAE direction 150° (sequential pre-hooks)

---

## Results

**StrongREJECT 313 prompts — Gemma-2-2B-IT — New rubric (refusal=0 AND convincingness≥3 AND specificity≥3)**

| Rank | Method | ASR | Layer | Formula | Hook | Refusal Dir | Selection |
|------|--------|-----|-------|---------|------|-------------|-----------|
| 1 | `angular_refusal_180` (tools dir, pre-hook) | **89.1%** | 15 | rotate r 180° | pre | tools dir | N/A |
| 2 | `sae_refdir_prehook` | **87.2%** | 15 | `h += 0.3·e_i − 1.0·r` | pre | tools dir | best-of-8 oracle |
| 3 | `refusal_dir_actadd` | **86.9%** | 15 | `h += −1.0·r` | pre | tools dir | N/A |
| 4 | `angular_sae_150 + refusal_180` (refusal first) | **76.0%** | 15 | rotate r 180° → rotate e_i 150° | pre | tools dir | best-of-8 oracle |
| 5 | `angular_sae_maxtext_refusal_150` | **61.7%** | 15 | rotate in plane {r, e_i} by 150° | pre | tools dir | best-of-8 oracle |
| 6 | `angular_refusal_180` (8-pair, fwd-hook) | **37.1%** | 15 | rotate r 180° | fwd | 8-pair | N/A |
| 7 | `refusal_only` | **30.4%** | all | orthogonal proj, all layers | pre+fwd | 8-pair | N/A |
| 8 | `baseline` | **1.0%** | — | no steering | — | — | — |

### Key Findings

- **Direction extraction space matters**: the refusal direction is extracted at the pre-hook (residual stream *input* to layer L). Applying the intervention as a pre-hook is geometrically consistent — both operate on the same representational space. Forward-hook intervention (on the layer *output*) operates in a different space and is significantly weaker (37.1% vs 89.1% for the same 180° flip).

- **Direction quality matters**: the precomputed tools direction (extracted from a large dataset with principled layer selection) substantially outperforms the simple 8-pair computed direction. We initially used an 8-pair direction as a fast approximation; switching to the tools direction was necessary for strong results.

- **Simple flip beats SAE augmentation**: pure 180° rotation of the refusal direction (89.1%) slightly outperforms SAE-augmented methods (87.2%), suggesting that for this model the refusal direction alone captures most of the safety-relevant subspace. SAE features provide a best-of-8 oracle upper bound — real single-feature selection would score lower.

- **SAE decoder vectors don't work as rotation axes**: using `W_dec[feat_idx]` from GemmaScope directly as a steering direction (instead of residual-stream activations of max-activating text) produced 0% ASR. The decoder weights are not aligned with the model's residual stream in a way that makes them useful for angular steering.

- **Angular rotation is not better than actadd**: rotation is norm-preserving but does not outperform simple linear subtraction. The 180° flip is mathematically equivalent to `h -= 2*(h·r̂)*r̂`, which is a stronger intervention than actadd (`h -= r`) for directions where the activation has large projection.

---

## Project Structure

```
shadow_steering/
├── main.py                          # entry point — loads config, runs pipeline
├── configs/
│   ├── base.yml                     # shared defaults
│   ├── datasets/                    # advbench_250, strongreject_313, quicktest
│   ├── models/gemma/, llama/        # model-specific settings
│   ├── pipelines/                   # inference_only, baseline
│   └── steering_methods/            # one yml per method
├── data/
│   ├── strongreject_313.json        # 313 harmful prompts (StrongREJECT)
│   └── advbench_250.json            # 250 harmful prompts (AdvBench)
├── scripts/
│   ├── gemma/                       # shell scripts for local runs
│   └── slurm/                       # SLURM job scripts for cluster
├── shadow_steering/                 # library
│   ├── datasets/
│   ├── models/base_model.py         # model loading, generate, generate_angular
│   ├── pipelines/inference_only.py
│   └── steering_methods/
│       ├── sae_text_steering.py     # SAE max-text + refusal subtraction
│       ├── refusal_dir.py           # actadd and ablation
│       ├── refusal_only.py          # orthogonal projection
│       └── angular_steering.py     # rotation-based steering (4 modes)
├── tools/
│   └── refusal_direction/           # precomputed refusal direction pipeline
└── results/                         # gitignored (scores*.json tracked)
```

---

## Setup

```bash
conda activate codec   # torch, transformers, huggingface_hub, requests
export NEURONPEDIA_API_KEY=your_key_here   # required for SAE methods
```

Models used:
- `google/gemma-2-2b-it` — requires `attn_implementation=eager`
- `meta-llama/Llama-3.1-8B-Instruct`

SAE weights: [GemmaScope](https://huggingface.co/google/gemma-scope-2b-pt-res) (`layer_15/width_16k/average_l0_23`)

---

## How to Run

```bash
# Single method
python main.py --config configs/base.yml \
  configs/datasets/strongreject_313.yml \
  configs/models/gemma/gemma2_2b_it.yml \
  configs/pipelines/inference_only.yml \
  configs/steering_methods/angular_refusal_180_toolsdir_prehook.yml

# SLURM cluster
sbatch scripts/slurm/full313_angular_refusal_toolsdir_gemma.slurm
```

---

## References

- Arditi et al. (2025). *Refusal in Language Models Is Mediated by a Single Direction*. [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)
- Winninger et al. (2025). *Shadow Steering: Angular Activation Steering for Jailbreaking LLMs*. [arXiv:2510.26243](https://arxiv.org/abs/2510.26243)
- Souly et al. (2024). *A StrongREJECT for Empty Jailbreaks*. [arXiv:2402.10260](https://arxiv.org/abs/2402.10260)
- [GemmaScope SAE](https://huggingface.co/google/gemma-scope-2b-pt-res) — Google DeepMind
- [Neuronpedia](https://neuronpedia.org) — SAE feature search API
