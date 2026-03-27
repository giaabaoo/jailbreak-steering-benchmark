"""
Baseline Pipeline
==================
Runs the model with NO steering — measures natural refusal rate.
Saves raw outputs to results/ for later evaluation.

Flow:
    dataset -> model -> generate each prompt (no hook) -> save JSON
"""

import json
import os

from shadow_steering.utils import setup_logger
from shadow_steering.datasets import get_data_loader
from shadow_steering.models import get_model


class Baseline:

    def __init__(self, config):
        self.config = config

    def run(self):
        setup_logger(self.config)
        data_loader = get_data_loader(self.config)
        model       = get_model(self.config)

        results = []
        total = len(data_loader)
        for i, item in enumerate(data_loader):
            prompt_id = item['id']
            prompt    = item['prompt']
            print(f"\n[{i+1}/{total}] P{prompt_id:02d}: {prompt[:60]}...")

            formatted = model.format_prompt(prompt)
            output    = model.generate(formatted)

            results.append({'prompt_id': prompt_id, 'prompt': prompt,
                            'output': output})
            print(f"  output[:200]: {output[:200]}")

        self._save(results)

    def _save(self, results: list):
        out_dir  = self.config.output_dir
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'results.json')
        with open(out_path, 'w') as f:
            json.dump({
                'config': {'model': self.config.model.hf_path, 'method': 'baseline'},
                'results': results,
            }, f, indent=2)
        print(f"\nSaved {len(results)} results to {out_path}")
