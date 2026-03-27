"""
GCG Steering (nanoGCG baseline)
================================
Optimizes a suffix appended to the prompt using nanoGCG.
Requires nanogcg to be installed: pip install nanogcg

Config fields (method.*):
    num_steps    : GCG optimization steps (default 250)
    search_width : candidates per step (default 512)
    suffix_len   : initial suffix token length (default 20)
    target       : target string to maximize (default 'Sure, here is')
"""

import nanogcg
import torch


class GCGSteering:

    def __init__(self, config):
        self.config      = config
        self.num_steps   = getattr(config.method, 'num_steps', 250)
        self.search_width = getattr(config.method, 'search_width', 512)
        self.suffix_len  = getattr(config.method, 'suffix_len', 20)
        self.target      = getattr(config.method, 'target', 'Sure, here is')

    def setup(self, model):
        # No global setup needed — suffix optimized per prompt
        pass

    def steer(self, prompt: str, model) -> dict:
        messages = [{'role': 'user', 'content': prompt}]
        gcg_config = nanogcg.GCGConfig(
            num_steps=self.num_steps,
            search_width=self.search_width,
            verbosity='WARNING',
        )
        result = nanogcg.run(
            model.model,
            model.tokenizer,
            messages,
            self.target,
            config=gcg_config,
        )
        suffix = result.best_string
        # Generate with the optimized suffix appended
        prompt_with_suffix = prompt + ' ' + suffix
        formatted = model.format_prompt(prompt_with_suffix)
        output = model.generate(formatted)

        return {
            'prompt':  prompt,
            'output':  output,
            'suffix':  suffix,
            'best_loss': result.best_loss,
        }
