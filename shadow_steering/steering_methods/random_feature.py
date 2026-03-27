"""
Random Feature Steering (ablation)
====================================
Uses a random text direction instead of an SAE-selected feature direction.
Baseline to verify that SAE feature selection is meaningful vs. any text direction.

Config fields (method.*):
    add_strength  : alpha+ (default 0.3)
    sub_strength  : alpha- (default 1.0)
    random_texts  : list of random texts to sample directions from
"""

import random
import torch
from .sae_text_steering import CONTRASTIVE_PAIRS

DEFAULT_RANDOM_TEXTS = [
    "The weather today is sunny and warm.",
    "She opened the book and started reading.",
    "The train arrived at the station at noon.",
    "He cooked pasta for dinner last night.",
    "The library closes at nine in the evening.",
    "A flock of birds flew over the rooftop.",
    "The quarterly report showed steady growth.",
    "They planted roses along the garden path.",
]


class RandomFeatureSteering:

    def __init__(self, config):
        self.config      = config
        self.add_str     = getattr(config.method, 'add_strength', 0.3)
        self.sub_str     = getattr(config.method, 'sub_strength', 1.0)
        self.rand_texts  = getattr(config.method, 'random_texts', DEFAULT_RANDOM_TEXTS)
        self.refusal_dir = None

    def setup(self, model):
        print("Computing refusal direction (random feature ablation)...")
        harmful_acts, benign_acts = [], []
        for harmful, benign in CONTRASTIVE_PAIRS:
            harmful_acts.append(model.get_activation(model.format_prompt(harmful)))
            benign_acts.append(model.get_activation(model.format_prompt(benign)))
        self.refusal_dir = (torch.stack(harmful_acts).mean(0) - torch.stack(benign_acts).mean(0)).float()

    def steer(self, prompt: str, model) -> dict:
        if self.refusal_dir is None:
            raise RuntimeError("Call setup(model) before steer().")
        rand_text  = random.choice(self.rand_texts)
        rand_vec   = model.get_activation(rand_text)
        formatted  = model.format_prompt(prompt)
        output = model.generate(
            formatted,
            add_vecs=[rand_vec],
            sub_vec=self.refusal_dir,
            add_str=self.add_str,
            sub_str=self.sub_str,
        )
        return {'prompt': prompt, 'output': output, 'random_text_used': rand_text}
