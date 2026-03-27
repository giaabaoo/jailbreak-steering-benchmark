"""
Refusal Direction Steering (reproduction of andyrdt/refusal_in_llms)
=====================================================================
Two methods using the precomputed refusal direction from tools/refusal_direction:

RefusalDirActadd:
    Mirrors `actadd` in run_pipeline.py:
        activation += coeff * direction   (coeff=-1.0, at selected layer only)
    - Single layer (model.refusal_dir_layer)
    - No normalization — raw direction vector
    - Pre-hook only at that one layer's input

RefusalDirAblation:
    Mirrors `ablation` in run_pipeline.py:
        direction = direction / direction.norm()
        activation -= (activation @ direction) * direction
    - All layers: residual pre-hook + attn output hook + MLP output hook
    - Normalization happens inside each hook (same as original)

Both load the precomputed direction.pt from:
    config.model.refusal_dir_path

Config fields (model.*):
    refusal_dir_path  : path to direction.pt
    refusal_dir_layer : empirically selected layer (for actadd only)
"""

import torch


# ---------------------------------------------------------------------------
# Hook functions — copied exactly from tools/refusal_direction/pipeline/utils/hook_utils.py
# ---------------------------------------------------------------------------

def _get_actadd_pre_hook(vector, coeff):
    """Exact copy of get_activation_addition_input_pre_hook."""
    def hook_fn(module, input):
        nonlocal vector
        if isinstance(input, tuple):
            activation = input[0]
        else:
            activation = input
        vector = vector.to(activation)
        activation += coeff * vector
        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn


def _get_ablation_pre_hook(direction):
    """Exact copy of get_direction_ablation_input_pre_hook."""
    def hook_fn(module, input):
        nonlocal direction
        if isinstance(input, tuple):
            activation = input[0]
        else:
            activation = input
        direction_norm = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction_norm = direction_norm.to(activation)
        activation -= (activation @ direction_norm).unsqueeze(-1) * direction_norm
        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn


def _get_ablation_output_hook(direction):
    """Exact copy of get_direction_ablation_output_hook."""
    def hook_fn(module, input, output):
        nonlocal direction
        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output
        direction_norm = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction_norm = direction_norm.to(activation)
        activation -= (activation @ direction_norm).unsqueeze(-1) * direction_norm
        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation
    return hook_fn


# ---------------------------------------------------------------------------
# RefusalDirActadd
# ---------------------------------------------------------------------------

class RefusalDirActadd:
    """
    Single-layer activation addition.
    Mirrors actadd_fwd_pre_hooks in run_pipeline.py:
        [(model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0))]
    """

    def __init__(self, config):
        self.config    = config
        self.dir_path  = config.model.refusal_dir_path
        self.layer     = config.model.refusal_dir_layer
        self.direction = None

    def setup(self, model):
        print(f"Loading refusal direction from {self.dir_path} ...")
        self.direction = torch.load(self.dir_path, map_location='cpu').float()
        print(f"  Direction shape: {self.direction.shape}, norm: {self.direction.norm():.4f}")
        print(f"  Applying at layer {self.layer} (actadd, coeff=-1.0)")

    def steer(self, prompt: str, model) -> dict:
        if self.direction is None:
            raise RuntimeError("Call setup(model) before steer().")

        formatted = model.format_prompt(prompt)
        ids = model.tokenizer(formatted, return_tensors='pt').to(model.device)
        input_len = ids['input_ids'].shape[1]

        # Register pre-hook at selected layer only — exact same as original
        hook = _get_actadd_pre_hook(vector=self.direction, coeff=-1.0)
        handle = model.model.model.layers[self.layer].register_forward_pre_hook(hook)

        max_new = getattr(self.config.model, 'max_new_tokens', 400)
        temp    = getattr(self.config.model, 'temperature', 0.7)
        sample  = getattr(self.config.model, 'do_sample', True)

        with torch.no_grad():
            out = model.model.generate(
                **ids,
                max_new_tokens=max_new,
                do_sample=sample,
                temperature=temp,
                pad_token_id=model.tokenizer.eos_token_id,
            )
        handle.remove()

        new_tokens = out[0][input_len:]
        output = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return {'prompt': prompt, 'output': output}


# ---------------------------------------------------------------------------
# RefusalDirAblation
# ---------------------------------------------------------------------------

class RefusalDirAblation:
    """
    Full orthogonal projection across all layers.
    Mirrors ablation_fwd_pre_hooks + ablation_fwd_hooks in run_pipeline.py:
        pre-hook  on ALL block modules  (residual input)
        fwd hook  on ALL attn modules   (attn output)
        fwd hook  on ALL mlp modules    (mlp output)
    """

    def __init__(self, config):
        self.config    = config
        self.dir_path  = config.model.refusal_dir_path
        self.direction = None

    def setup(self, model):
        print(f"Loading refusal direction from {self.dir_path} ...")
        self.direction = torch.load(self.dir_path, map_location='cpu').float()
        print(f"  Direction shape: {self.direction.shape}, norm: {self.direction.norm():.4f}")
        n_layers = model.model.config.num_hidden_layers
        print(f"  Applying orthogonal projection across all {n_layers} layers (pre + attn + mlp hooks)")

    def steer(self, prompt: str, model) -> dict:
        if self.direction is None:
            raise RuntimeError("Call setup(model) before steer().")

        formatted = model.format_prompt(prompt)
        ids = model.tokenizer(formatted, return_tensors='pt').to(model.device)
        input_len = ids['input_ids'].shape[1]

        n_layers = model.model.config.num_hidden_layers
        handles = []

        # Pre-hook on all residual block inputs — exact same as get_all_direction_ablation_hooks
        for i in range(n_layers):
            h = model.model.model.layers[i].register_forward_pre_hook(
                _get_ablation_pre_hook(self.direction)
            )
            handles.append(h)

        # Forward hook on all attn outputs
        for i in range(n_layers):
            h = model.model.model.layers[i].self_attn.register_forward_hook(
                _get_ablation_output_hook(self.direction)
            )
            handles.append(h)

        # Forward hook on all MLP outputs
        for i in range(n_layers):
            h = model.model.model.layers[i].mlp.register_forward_hook(
                _get_ablation_output_hook(self.direction)
            )
            handles.append(h)

        max_new = getattr(self.config.model, 'max_new_tokens', 400)
        temp    = getattr(self.config.model, 'temperature', 0.7)
        sample  = getattr(self.config.model, 'do_sample', True)

        with torch.no_grad():
            out = model.model.generate(
                **ids,
                max_new_tokens=max_new,
                do_sample=sample,
                temperature=temp,
                pad_token_id=model.tokenizer.eos_token_id,
            )

        for h in handles:
            h.remove()

        new_tokens = out[0][input_len:]
        output = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return {'prompt': prompt, 'output': output}
