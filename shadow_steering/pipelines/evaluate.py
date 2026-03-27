"""
Evaluate Pipeline
==================
Loads an existing results.json and runs automated evaluation:
  - Rule-based refusal detection
  - LlamaGuard-3 safety classification

Saves scored results to results_scored.json in the same directory.
Actionability (1-5) is left for Claude to fill in manually — see
shadow_steering/evaluators/CLAUDE_SCORING_INSTRUCTIONS.md.

Config fields (pipeline.*):
    results_path : path to the results.json to evaluate
"""

import json
import os

from shadow_steering.utils import setup_logger
from shadow_steering.evaluators import get_evaluator


class Evaluate:

    def __init__(self, config):
        self.config = config

    def run(self):
        setup_logger(self.config)

        results_path = self.config.pipeline.results_path
        print(f"Loading results from {results_path}")
        with open(results_path) as f:
            data = json.load(f)

        results  = data['results']
        evaluator = get_evaluator(self.config)

        print(f"\nEvaluating {len(results)} outputs...")
        scored = evaluator.evaluate(results)

        # Summary
        n_refused  = sum(1 for r in scored if r.get('refused'))
        n_unsafe   = sum(1 for r in scored
                        if r.get('llamaguard', {}).get('safe') is False)
        print(f"\nRefusal rate (rule-based): {n_refused}/{len(scored)}")
        print(f"Unsafe (LlamaGuard):       {n_unsafe}/{len(scored)}")

        # Save
        out_dir  = self.config.output_dir
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'results_scored.json')
        data['results'] = scored
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved scored results to {out_path}")
        print("Next: ask Claude to score Actionability using "
              "shadow_steering/evaluators/CLAUDE_SCORING_INSTRUCTIONS.md")
