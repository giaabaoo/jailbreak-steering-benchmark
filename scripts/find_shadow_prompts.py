"""
Shadow Prompt Finder
====================
For each harmful prompt:
  1. Compute h_harmful = activation at layer 15
  2. Compute h_target = h_harmful - r (subtract refusal direction)
  3. Find nearest benign text to h_target by cosine similarity
  4. Run that benign text as plain input -> see if model produces harmful output

Usage:
  python scripts/find_shadow_prompts.py
"""

import json
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shadow_steering.models import get_model
from omegaconf import OmegaConf

# ── Config ──────────────────────────────────────────────────────────────
LAYER = 15
N_HARMFUL = 5       # how many harmful prompts to test
N_NEIGHBORS = 3     # top-N nearest neighbors to try per harmful prompt

HARMFUL_PATH  = 'tools/refusal_direction/dataset/splits/harmful_val.json'
HARMLESS_PATH = 'tools/refusal_direction/dataset/splits/harmless_val.json'
REFUSAL_DIR_PATH = 'tools/refusal_direction/pipeline/runs/gemma-2-2b-it/direction.pt'

# ── Load model ───────────────────────────────────────────────────────────
cfg = OmegaConf.load('configs/base.yml')
model_cfg = OmegaConf.load('configs/models/gemma/gemma2_2b_it.yml')
cfg = OmegaConf.merge(cfg, model_cfg)

print("Loading model...")
model = get_model(cfg)
model.eval()

# ── Load refusal direction ────────────────────────────────────────────────
r = torch.load(REFUSAL_DIR_PATH, map_location='cpu').float()
print(f"Refusal direction norm: {r.norm():.4f}")

# ── Load datasets ─────────────────────────────────────────────────────────
with open(HARMFUL_PATH) as f:
    harmful = [d['instruction'] for d in json.load(f)][:N_HARMFUL]

with open(HARMLESS_PATH) as f:
    harmless = [d['instruction'] for d in json.load(f)]

print(f"Harmful prompts: {len(harmful)}")
print(f"Harmless corpus: {len(harmless)}")

# ── Precompute harmless activations ───────────────────────────────────────
print("\nPrecomputing harmless activations...")
harmless_acts = []
for i, text in enumerate(harmless):
    if i % 200 == 0:
        print(f"  {i}/{len(harmless)}")
    with torch.no_grad():
        act = model.get_activation(model.format_prompt(text))
    harmless_acts.append(act)

harmless_matrix = torch.stack(harmless_acts)  # [N, d]
harmless_matrix_norm = harmless_matrix / harmless_matrix.norm(dim=-1, keepdim=True)
print(f"Harmless matrix shape: {harmless_matrix.shape}")

# ── For each harmful prompt: find nearest neighbor to h_target ────────────
results = []
for harmful_prompt in harmful:
    print(f"\n{'='*60}")
    print(f"HARMFUL: {harmful_prompt}")

    with torch.no_grad():
        h_harmful = model.get_activation(model.format_prompt(harmful_prompt))

    h_target = h_harmful - r
    h_target_norm = h_target / h_target.norm()

    # cosine similarity with all harmless activations
    sims = (harmless_matrix_norm @ h_target_norm).cpu()
    top_k = sims.topk(N_NEIGHBORS)

    print(f"\nTop-{N_NEIGHBORS} nearest benign texts to h_target:")
    for rank, (sim, idx) in enumerate(zip(top_k.values, top_k.indices)):
        neighbor_text = harmless[idx]
        print(f"\n  [{rank+1}] sim={sim:.4f}: {neighbor_text[:100]}")

        # Run neighbor text as plain input
        with torch.no_grad():
            output = model.generate(
                neighbor_text,
                add_vec=None, sub_vec=None,
                add_str=0.0, sub_str=0.0,
            )
        print(f"  OUTPUT: {output[:300]}")

    results.append({
        'harmful_prompt': harmful_prompt,
        'neighbors': [
            {'text': harmless[idx], 'sim': float(sims[idx]), 'output': None}
            for idx in top_k.indices
        ]
    })

print("\nDone.")
