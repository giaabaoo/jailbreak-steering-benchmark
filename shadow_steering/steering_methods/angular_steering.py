"""
Angular Steering
================
Rotation-based activation steering in a 2D subspace, adapted from:
  https://github.com/lone17/angular-steering (Winninger et al., 2510.26243)

Instead of linearly adding/subtracting vectors, we rotate the activation
in the plane spanned by {first_dir, second_dir} by target_degree degrees.
This is norm-preserving (unlike actadd) and generalizes ablation (90°)
and actadd as special cases.

Four modes (set via method.mode):
  'refusal_only'    : first_dir=refusal, second_dir=None → pure ablation rotation
  'sae_refusal'     : first_dir=refusal, second_dir=e_i  → rotate toward SAE feature
  'sae_only'        : first_dir=e_i,     second_dir=None → rotate in SAE feature direction
  'sae_decoder'     : first_dir=W_dec[feat_idx], second_dir=None → rotate using SAE decoder weight

Config fields (method.*):
    mode             : 'refusal_only' | 'sae_refusal' | 'sae_only' | 'sae_decoder'
    target_degree    : rotation angle in degrees (default 90)
    adaptive         : if True, only rotate tokens with positive projection onto first_dir
    num_features     : how many SAE features to use per prompt (default 8, used when sae_* mode)
    neuronpedia_api_key : API key
    sae_weights_path : path to GemmaScope params.npz (required for sae_decoder mode)

Config fields (model.*):
    layer, neuronpedia_model_id, neuronpedia_source_set, refusal_dir_path
"""

import os
import time
import numpy as np
import torch
import requests

from .sae_text_steering import CONTRASTIVE_PAIRS, NP_BASE


def _get_rotation_args(first_dir: torch.Tensor, second_dir: torch.Tensor | None,
                       target_degree: float):
    """
    Compute proj_matrix and rotated_component for a rotation of target_degree
    in the plane spanned by {first_dir, second_dir}.

    If second_dir is None, returns (None, None) — no rotation applied.

    All tensors on CPU float32. The hook will move them to the right device/dtype.
    """
    if second_dir is None:
        # 1D case: rotate in plane {first_dir, first_dir_perp}
        # For ablation (90°): Px → 0, meaning remove component. We still need a
        # second basis vector. We create an arbitrary orthogonal one.
        # But the angular paper handles this by rotating within {b1, b2} where
        # second_dir needs to be provided. For refusal_only with 90° this equals ablation.
        # We use the "ablation" interpretation: project out the refusal direction.
        # Return a sentinel so the hook knows to do ablation only.
        return None, None

    b1 = first_dir / (first_dir.norm() + 1e-8)
    b2 = second_dir - torch.dot(second_dir, b1) * b1
    b2 = b2 / (b2.norm() + 1e-8)

    theta = np.deg2rad(target_degree)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))

    # proj_matrix: projects onto the 2D plane
    proj_matrix = torch.outer(b1, b1) + torch.outer(b2, b2)  # (d, d)

    # rotated_component: unit vector at angle theta from b1 within the plane
    # R(theta) * [1, 0]^T = [cos, sin]^T in the {b1,b2} basis
    rotated_component = cos_t * b1 + sin_t * b2  # (d,)

    return proj_matrix, rotated_component


class AngularSteering:

    def __init__(self, config):
        self.config        = config
        self.mode          = getattr(config.method, 'mode', 'refusal_only')
        self.target_degree = getattr(config.method, 'target_degree', 90.0)
        self.adaptive      = getattr(config.method, 'adaptive', False)
        self.use_pre_hook  = getattr(config.method, 'use_pre_hook', False)
        self.num_feats     = getattr(config.method, 'num_features', 8)
        api_key            = getattr(config.method, 'neuronpedia_api_key', None) \
                             or os.environ.get('NEURONPEDIA_API_KEY', '')
        self.np_headers    = {'x-api-key': api_key} if api_key else {}
        self.np_model      = getattr(config.model, 'neuronpedia_model_id', '')
        self.np_source     = getattr(config.model, 'neuronpedia_source_set', '')
        self.layer         = config.model.layer
        self.sae_id        = f"{self.layer}-{self.np_source}"
        self.refusal_dir   = None
        self.W_dec         = None  # SAE decoder weights, loaded for sae_decoder mode

    def setup(self, model):
        """Load or compute refusal direction. Load SAE decoder weights if needed."""
        dir_path = getattr(self.config.model, 'refusal_dir_path', None)
        if dir_path:
            print(f"Loading refusal direction from {dir_path} ...")
            self.refusal_dir = torch.load(dir_path, map_location='cpu').float()
        else:
            print("Computing refusal direction from contrastive pairs...")
            harmful_acts, benign_acts = [], []
            for harmful, benign in CONTRASTIVE_PAIRS:
                harmful_acts.append(model.get_activation(model.format_prompt(harmful)))
                benign_acts.append(model.get_activation(model.format_prompt(benign)))
            self.refusal_dir = (torch.stack(harmful_acts).mean(0)
                                - torch.stack(benign_acts).mean(0)).float()
        print(f"  Refusal direction norm: {self.refusal_dir.norm():.4f}")

        if self.mode == 'sae_decoder':
            import numpy as np
            sae_path = getattr(self.config.method, 'sae_weights_path', None)
            if not sae_path:
                raise ValueError("sae_decoder mode requires method.sae_weights_path in config")
            print(f"Loading SAE decoder weights from {sae_path} ...")
            sae = np.load(sae_path)
            self.W_dec = torch.tensor(sae['W_dec'], dtype=torch.float32)  # (n_features, d_model)
            print(f"  W_dec shape: {self.W_dec.shape}")

    def steer(self, prompt: str, model) -> dict:
        if self.refusal_dir is None:
            raise RuntimeError("Call setup(model) before steer().")

        formatted = model.format_prompt(prompt)

        if self.mode == 'refusal_only':
            # Rotate refusal direction by target_degree — no second_dir needed
            # At 90°, this equals ablation. At 180°, it flips the refusal direction.
            output = model.generate_angular(
                formatted,
                first_dir=self.refusal_dir,
                second_dir=None,
                target_degree=self.target_degree,
                adaptive=self.adaptive,
                use_pre_hook=self.use_pre_hook,
            )
            refused = self._is_refused(output)
            return {'prompt': prompt, 'output': output, 'refused': refused}

        elif self.mode in ('sae_refusal_double_sae_first', 'sae_refusal_double_refusal_first'):
            sae_first = (self.mode == 'sae_refusal_double_sae_first')
            features = self._get_top_features(prompt)
            if not features:
                return {'prompt': prompt, 'error': 'no features', 'feature_runs': []}

            feature_runs = []
            for i, feat in enumerate(features):
                try:
                    e_i = model.get_activation(feat['max_text'])
                except Exception as ex:
                    print(f"    [feat {feat['index']}] activation error: {ex}")
                    continue

                output = model.generate_angular_double(
                    formatted,
                    first_dir=self.refusal_dir,
                    second_dir=e_i,
                    sae_degree=self.target_degree,
                    refusal_degree=180.0,
                    sae_first=sae_first,
                )
                refused = self._is_refused(output)
                print(f"    [{i+1}/{len(features)}] feat #{feat['index']} "
                      f"({feat['label'][:40]}) -> refused={refused}")

                feature_runs.append({
                    'feature_index': feat['index'],
                    'feature_label': feat['label'],
                    'max_text':      feat['max_text'],
                    'output':        output,
                    'refused':       refused,
                })

            return {'prompt': prompt, 'feature_runs': feature_runs}

        elif self.mode == 'sae_decoder':
            features = self._get_top_features(prompt)
            if not features:
                return {'prompt': prompt, 'error': 'no features', 'feature_runs': []}

            feature_runs = []
            for i, feat in enumerate(features):
                feat_idx = int(feat['index'])
                e_i = self.W_dec[feat_idx]  # SAE decoder weight vector, shape (d_model,)

                output = model.generate_angular(
                    formatted,
                    first_dir=e_i,
                    second_dir=None,
                    target_degree=self.target_degree,
                    adaptive=self.adaptive,
                    use_pre_hook=self.use_pre_hook,
                )
                refused = self._is_refused(output)
                print(f"    [{i+1}/{len(features)}] feat #{feat['index']} "
                      f"({feat['label'][:40]}) -> refused={refused}")

                feature_runs.append({
                    'feature_index': feat['index'],
                    'feature_label': feat['label'],
                    'output':        output,
                    'refused':       refused,
                })

            return {'prompt': prompt, 'feature_runs': feature_runs}

        elif self.mode in ('sae_refusal', 'sae_only'):
            features = self._get_top_features(prompt)
            if not features:
                return {'prompt': prompt, 'error': 'no features', 'feature_runs': []}

            feature_runs = []
            for i, feat in enumerate(features):
                try:
                    e_i = model.get_activation(feat['max_text'])
                except Exception as ex:
                    print(f"    [feat {feat['index']}] activation error: {ex}")
                    continue

                if self.mode == 'sae_refusal':
                    # Rotate in plane {refusal_dir → e_i}
                    first_dir  = self.refusal_dir
                    second_dir = e_i
                else:
                    # Rotate in plane {e_i only} — ablation toward SAE direction
                    first_dir  = e_i
                    second_dir = None

                output = model.generate_angular(
                    formatted,
                    first_dir=first_dir,
                    second_dir=second_dir,
                    target_degree=self.target_degree,
                    adaptive=self.adaptive,
                )
                refused = self._is_refused(output)
                print(f"    [{i+1}/{len(features)}] feat #{feat['index']} "
                      f"({feat['label'][:40]}) -> refused={refused}")

                feature_runs.append({
                    'feature_index': feat['index'],
                    'feature_label': feat['label'],
                    'max_text':      feat['max_text'],
                    'output':        output,
                    'refused':       refused,
                })

            return {'prompt': prompt, 'feature_runs': feature_runs}
        else:
            raise ValueError(f"Unknown angular steering mode: {self.mode}")

    def _is_refused(self, text: str) -> bool:
        lower = text.lower()
        return any(p in lower for p in [
            "i cannot", "i can't", "i won't", "i will not",
            "cannot fulfill", "cannot provide", "i apologize",
        ])

    # ------------------------------------------------------------------
    # Neuronpedia helpers (same as SAETextSteering)
    # ------------------------------------------------------------------
    def _get_top_features(self, prompt_text: str) -> list:
        payload = {
            'modelId':        self.np_model,
            'sourceSet':      self.np_source,
            'selectedLayers': [self.sae_id],
            'text':           prompt_text,
            'ignoreBos':      True,
            'numResults':     50,
            'sortIndexes':    [],
        }
        for attempt in range(5):
            try:
                r = requests.post(f"{NP_BASE}/api/search-all",
                                  headers=self.np_headers, json=payload, timeout=60)
                data = r.json() if r.status_code == 200 else {}
                break
            except Exception as e:
                print(f"  [API] attempt {attempt+1} failed: {e}, retrying...")
                time.sleep(5 * (attempt + 1))
                data = {}
        tokens = data.get('tokens', [[]])
        if tokens and isinstance(tokens[0], list):
            tokens = tokens[0]
        payload['sortIndexes'] = [max(len(tokens) - 1, 0)]

        for attempt in range(5):
            try:
                r = requests.post(f"{NP_BASE}/api/search-all",
                                  headers=self.np_headers, json=payload, timeout=60)
                if r.status_code == 200:
                    data = r.json()
                    break
                time.sleep(5 * (attempt + 1))
            except Exception as e:
                time.sleep(5 * (attempt + 1))
        else:
            return []

        features = []
        for feat in data.get('result', []):
            if len(features) >= self.num_feats:
                break
            feat_idx = str(feat.get('index'))
            try:
                details = self._get_feature_details(feat_idx)
                if details.get('frac_nonzero', 1.0) > 0.01:
                    continue
                label = ''
                if details.get('explanations'):
                    label = details['explanations'][0].get('description', '')
                acts = details.get('activations', [])
                if not acts:
                    continue
                max_text = ''.join(
                    t.replace('▁', ' ').replace('<0x0A>', '\n')
                    for t in acts[0].get('tokens', [])
                ).strip()
                features.append({'index': feat_idx, 'label': label, 'max_text': max_text})
                time.sleep(0.1)
            except Exception:
                continue
        return features

    def _get_feature_details(self, feat_idx: str) -> dict:
        r = requests.get(
            f"{NP_BASE}/api/feature/{self.np_model}/{self.sae_id}/{feat_idx}",
            headers=self.np_headers, timeout=15)
        r.raise_for_status()
        return r.json()
