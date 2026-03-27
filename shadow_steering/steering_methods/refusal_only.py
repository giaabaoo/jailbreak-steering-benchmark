"""
Refusal-Only Steering (ablation)
=================================
Subtracts the refusal direction only — no SAE feature added.
Tests whether simply suppressing refusal is sufficient.

Config fields (method.*):
    sub_strength : alpha- (default 1.0)
"""

import torch
from .sae_text_steering import CONTRASTIVE_PAIRS


class RefusalOnlySteering:

    def __init__(self, config):
        self.config      = config
        self.sub_str     = getattr(config.method, 'sub_strength', 1.0)
        self.refusal_dir = None

    def setup(self, model):
        print("Computing refusal direction (refusal-only ablation)...")
        harmful_acts, benign_acts = [], []
        for harmful, benign in CONTRASTIVE_PAIRS:
            harmful_acts.append(model.get_activation(model.format_prompt(harmful)))
            benign_acts.append(model.get_activation(model.format_prompt(benign)))
        self.refusal_dir = (torch.stack(harmful_acts).mean(0) - torch.stack(benign_acts).mean(0)).float()

    def steer(self, prompt: str, model) -> dict:
        if self.refusal_dir is None:
            raise RuntimeError("Call setup(model) before steer().")
        formatted = model.format_prompt(prompt)
        output = model.generate(
            formatted,
            add_vec=None,
            sub_vec=self.refusal_dir,
            add_str=0.0,
            sub_str=self.sub_str,
        )
        return {'prompt': prompt, 'output': output}
