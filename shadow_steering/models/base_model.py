import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseModel:
    """
    Shared model loading and inference logic for all models.

    Config fields:
        model.hf_path        : HuggingFace model path
        model.layer          : residual-stream layer to steer/extract from
        model.dtype          : 'bfloat16' | 'float16' | 'float32'
        model.device         : e.g. 'cuda:0'
        model.max_new_tokens : generation length
        model.temperature    : sampling temperature
        model.do_sample      : bool
        model.attn_impl      : attention implementation (e.g. 'eager')
    """

    def __init__(self, config):
        self.config = config
        self.device = config.model.device
        self.layer  = config.model.layer
        dtype_map   = {'bfloat16': torch.bfloat16,
                       'float16': torch.float16,
                       'float32': torch.float32}
        self.dtype  = dtype_map.get(getattr(config.model, 'dtype', 'bfloat16'),
                                    torch.bfloat16)

        print(f"Loading {config.model.hf_path} ...")
        kwargs = dict(torch_dtype=self.dtype, device_map='auto')
        attn = getattr(config.model, 'attn_impl', None)
        if attn:
            kwargs['attn_implementation'] = attn

        self.tokenizer = AutoTokenizer.from_pretrained(config.model.hf_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model.hf_path, **kwargs)
        self.model.eval()
        torch.set_grad_enabled(False)
        # Resolve actual device after device_map='auto' placement
        self.device = next(self.model.parameters()).device

    def get_activation(self, text: str) -> torch.Tensor:
        """Return hidden state at self.layer+1, last token, as float32 CPU tensor."""
        ids = self.tokenizer(text, return_tensors='pt',
                             truncation=True, max_length=256).to(self.device)
        with torch.no_grad():
            out = self.model(**ids, output_hidden_states=True)
        act = out.hidden_states[self.layer + 1][0, -1, :].clone().float().cpu()
        del out
        torch.cuda.empty_cache()
        return act

    def generate(self, prompt_text: str,
                 add_vec: torch.Tensor = None,
                 sub_vec: torch.Tensor = None,
                 add_str: float = 1.0,
                 sub_str: float = 1.0,
                 use_pre_hook: bool = False) -> str:
        """
        Generate with optional activation steering hook at self.layer.

        add_vec      : float32 CPU tensor — added (harmful + SAE feature direction).
        sub_vec      : float32 CPU tensor — subtracted (refusal direction).
        use_pre_hook : if True, apply as pre-hook (residual input); else forward hook (output).
        Both vectors should be unit-normalized; strengths scale them directly.
        """
        handle = None
        if add_vec is not None or sub_vec is not None:
            if use_pre_hook:
                def hook_fn(module, inp):
                    h = inp[0] if isinstance(inp, tuple) else inp
                    if add_vec is not None:
                        h = h + add_str * add_vec.to(h.device, h.dtype)
                    if sub_vec is not None:
                        h = h - sub_str * sub_vec.to(h.device, h.dtype)
                    return (h,) + inp[1:] if isinstance(inp, tuple) else h
                handle = self.model.model.layers[self.layer].register_forward_pre_hook(hook_fn)
            else:
                def hook_fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    if add_vec is not None:
                        h = h + add_str * add_vec.to(h.device, h.dtype)
                    if sub_vec is not None:
                        h = h - sub_str * sub_vec.to(h.device, h.dtype)
                    return (h,) + out[1:] if isinstance(out, tuple) else h
                handle = self.model.model.layers[self.layer].register_forward_hook(hook_fn)

        ids = self.tokenizer(prompt_text, return_tensors='pt').to(self.device)
        input_len = ids['input_ids'].shape[1]
        max_new = getattr(self.config.model, 'max_new_tokens', 400)
        temp    = getattr(self.config.model, 'temperature', 0.7)
        sample  = getattr(self.config.model, 'do_sample', True)

        with torch.no_grad():
            out = self.model.generate(
                **ids,
                max_new_tokens=max_new,
                do_sample=sample,
                temperature=temp,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        if handle:
            handle.remove()

        new_tokens = out[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_angular(self, prompt_text: str,
                         first_dir: torch.Tensor,
                         second_dir: torch.Tensor | None,
                         target_degree: float = 90.0,
                         adaptive: bool = False,
                         use_pre_hook: bool = False) -> str:
        """
        Generate with angular (rotation) steering hook at self.layer.

        Rotates the activation by target_degree in the 2D plane spanned by
        {first_dir, second_dir}. Norm-preserving.

        If second_dir is None, performs ablation of first_dir component (equivalent
        to projecting out first_dir, same as 90° rotation with arbitrary second basis).

        adaptive: if True, only rotate tokens where activation has positive
                  projection onto first_dir.
        """
        handle = None

        first_dir_f = first_dir.float()
        b1 = first_dir_f / (first_dir_f.norm() + 1e-8)

        if second_dir is not None:
            b2 = second_dir.float()
            b2 = b2 - torch.dot(b2, b1) * b1
            b2 = b2 / (b2.norm() + 1e-8)

            theta   = np.deg2rad(target_degree)
            cos_t   = float(np.cos(theta))
            sin_t   = float(np.sin(theta))

            # (d, d) projection matrix onto 2D plane
            proj_matrix       = torch.outer(b1, b1) + torch.outer(b2, b2)
            # unit vector at angle theta from b1 in the {b1,b2} plane
            rotated_component = cos_t * b1 + sin_t * b2

            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                pm = proj_matrix.to(h.device, h.dtype)
                rc = rotated_component.to(h.device, h.dtype)
                Px    = torch.einsum('bsd,de->bse', h, pm)   # (B,S,d)
                scale = Px.norm(dim=-1, keepdim=True)         # (B,S,1)
                if adaptive:
                    b1_dev = b1.to(h.device, h.dtype)
                    proj   = (h * b1_dev).sum(dim=-1)         # (B,S)
                    mask   = (proj > 0).unsqueeze(-1)         # (B,S,1)
                    h = h + mask * (scale * rc - Px)
                else:
                    h = h + (scale * rc - Px)
                return (h,) + out[1:] if isinstance(out, tuple) else h
        else:
            # 1D case: rotate in plane {b1, b1_perp} where b1_perp is arbitrary.
            # The effect on the b1 component: proj * b1 -> proj * cos(theta) * b1
            # (the perpendicular component is unchanged since we have no second_dir to rotate toward)
            # At 90°: multiplies b1 component by 0 (ablation)
            # At 180°: multiplies b1 component by -1 (reflection)
            theta = np.deg2rad(target_degree)
            cos_t = float(np.cos(theta))

            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                b1_dev = b1.to(h.device, h.dtype)
                proj   = (h * b1_dev).sum(dim=-1, keepdim=True)  # (B,S,1)
                # replace b1 component: proj*b1 -> proj*cos_t*b1
                # i.e. h = h + (cos_t - 1) * proj * b1
                if adaptive:
                    mask = (proj > 0)
                    h    = h + mask * (cos_t - 1) * proj * b1_dev
                else:
                    h    = h + (cos_t - 1) * proj * b1_dev
                return (h,) + out[1:] if isinstance(out, tuple) else h

        if use_pre_hook:
            def pre_hook_fn(module, inp):
                h = inp[0] if isinstance(inp, tuple) else inp
                fake_out = (h,)
                result = hook_fn(module, None, fake_out)
                new_h = result[0] if isinstance(result, tuple) else result
                return (new_h,) + inp[1:] if isinstance(inp, tuple) else new_h
            handle = self.model.model.layers[self.layer].register_forward_pre_hook(pre_hook_fn)
        else:
            handle = self.model.model.layers[self.layer].register_forward_hook(hook_fn)

        ids       = self.tokenizer(prompt_text, return_tensors='pt').to(self.device)
        input_len = ids['input_ids'].shape[1]
        max_new   = getattr(self.config.model, 'max_new_tokens', 400)
        temp      = getattr(self.config.model, 'temperature', 0.7)
        sample    = getattr(self.config.model, 'do_sample', True)

        with torch.no_grad():
            out = self.model.generate(
                **ids,
                max_new_tokens=max_new,
                do_sample=sample,
                temperature=temp,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        handle.remove()

        new_tokens = out[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_angular_double(self, prompt_text: str,
                                first_dir: torch.Tensor,
                                second_dir: torch.Tensor,
                                sae_degree: float = 150.0,
                                refusal_degree: float = 180.0,
                                sae_first: bool = True) -> str:
        """
        Generate with two sequential angular steering pre-hooks at self.layer.

        Hook 1: rotate in {first_dir, second_dir} plane by sae_degree (SAE rotation)
        Hook 2: reflect first_dir by refusal_degree (1D refusal flip)

        sae_first=True  → SAE rotation applied first, then refusal flip
        sae_first=False → refusal flip applied first, then SAE rotation

        Both hooks are pre-hooks and fire in registration order.
        """
        first_dir_f = first_dir.float()
        b1 = first_dir_f / (first_dir_f.norm() + 1e-8)

        # SAE rotation hook (2D)
        b2 = second_dir.float()
        b2 = b2 - torch.dot(b2, b1) * b1
        b2 = b2 / (b2.norm() + 1e-8)
        theta_sae = np.deg2rad(sae_degree)
        cos_sae = float(np.cos(theta_sae))
        sin_sae = float(np.sin(theta_sae))
        proj_matrix = torch.outer(b1, b1) + torch.outer(b2, b2)
        rotated_component = cos_sae * b1 + sin_sae * b2

        def sae_hook_fn(module, inp):
            h = inp[0] if isinstance(inp, tuple) else inp
            pm = proj_matrix.to(h.device, h.dtype)
            rc = rotated_component.to(h.device, h.dtype)
            Px = torch.einsum('bsd,de->bse', h, pm)
            scale = Px.norm(dim=-1, keepdim=True)
            h = h + (scale * rc - Px)
            return (h,) + inp[1:] if isinstance(inp, tuple) else h

        # Refusal flip hook (1D, 180°)
        cos_ref = float(np.cos(np.deg2rad(refusal_degree)))

        def refusal_hook_fn(module, inp):
            h = inp[0] if isinstance(inp, tuple) else inp
            b1_dev = b1.to(h.device, h.dtype)
            proj = (h * b1_dev).sum(dim=-1, keepdim=True)
            h = h + (cos_ref - 1) * proj * b1_dev
            return (h,) + inp[1:] if isinstance(inp, tuple) else h

        layer = self.model.model.layers[self.layer]
        if sae_first:
            h1 = layer.register_forward_pre_hook(sae_hook_fn)
            h2 = layer.register_forward_pre_hook(refusal_hook_fn)
        else:
            h1 = layer.register_forward_pre_hook(refusal_hook_fn)
            h2 = layer.register_forward_pre_hook(sae_hook_fn)

        ids = self.tokenizer(prompt_text, return_tensors='pt').to(self.device)
        input_len = ids['input_ids'].shape[1]
        max_new = getattr(self.config.model, 'max_new_tokens', 400)
        temp = getattr(self.config.model, 'temperature', 0.7)
        sample = getattr(self.config.model, 'do_sample', True)

        with torch.no_grad():
            out = self.model.generate(
                **ids,
                max_new_tokens=max_new,
                do_sample=sample,
                temperature=temp,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        h1.remove()
        h2.remove()

        new_tokens = out[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def format_prompt(self, prompt: str) -> str:
        """Subclasses override to apply model-specific chat formatting."""
        raise NotImplementedError
