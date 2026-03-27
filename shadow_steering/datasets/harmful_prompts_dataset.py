import json
import os


class HarmfulPromptsDataset:
    """
    Loads the 25 harmful prompts dataset.

    Config fields:
        dataset.data_path  : path to JSON file (list of {id, prompt})
        dataset.num_samples: first-N cap (0 = all, default 0)
        dataset.prompt_ids : list of specific IDs to run ([] = all, default [])
                             takes priority over num_samples if non-empty
    """

    def __init__(self, config):
        self.config = config
        data_path = config.dataset.data_path
        with open(data_path, 'r') as f:
            raw = json.load(f)

        # Filter by specific IDs if provided
        ids = getattr(config.dataset, 'prompt_ids', [])
        if ids:
            raw = [item for item in raw if item['id'] in ids]
        else:
            # Otherwise cap by num_samples
            num = getattr(config.dataset, 'num_samples', 0)
            if num and num > 0:
                raw = raw[:num]

        self.items = raw   # list of {'id': int, 'prompt': str}

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)
