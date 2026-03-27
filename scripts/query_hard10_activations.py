"""
Query Neuronpedia activation scores for hard-10 prompts.
Saves per-prompt, per-feature: index, label, maxValue (activation at last token),
frac_nonzero, and whether it was selected (passed density filter).

Output: results/hard10_activations.json
"""

import json
import time
import os
import requests

NP_BASE = "https://neuronpedia.org"
NP_MODEL = "gemma-2-2b"
NP_SOURCE = "gemmascope-res-16k"
LAYER = 15
SAE_ID = f"{LAYER}-{NP_SOURCE}"
NUM_RESULTS = 50
DENSITY_THRESHOLD = 0.01
NUM_FEATURES = 8

api_key = os.environ.get("NEURONPEDIA_API_KEY", "")
headers = {"x-api-key": api_key} if api_key else {}

HARD10_PATH = "data/hard_313_10.json"
OUTPUT_PATH = "results/hard10_activations.json"


def search_all(prompt_text, sort_index):
    payload = {
        "modelId": NP_MODEL,
        "sourceSet": NP_SOURCE,
        "selectedLayers": [SAE_ID],
        "text": prompt_text,
        "ignoreBos": True,
        "numResults": NUM_RESULTS,
        "sortIndexes": [sort_index],
    }
    for attempt in range(5):
        try:
            r = requests.post(f"{NP_BASE}/api/search-all", headers=headers,
                              json=payload, timeout=60)
            if r.status_code == 200:
                return r.json()
            time.sleep(5 * (attempt + 1))
        except Exception as e:
            print(f"  API attempt {attempt+1} failed: {e}")
            time.sleep(5 * (attempt + 1))
    return {}


def get_feature_details(feat_idx):
    r = requests.get(
        f"{NP_BASE}/api/feature/{NP_MODEL}/{SAE_ID}/{feat_idx}",
        headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()


def query_prompt(prompt_id, prompt_text):
    print(f"\n[P{prompt_id}] {prompt_text[:60]}...")

    # First call to get token count
    data = search_all(prompt_text, sort_index=0)
    tokens = data.get("tokens", [[]])
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    last_token_idx = max(len(tokens) - 1, 0)
    print(f"  Tokens: {len(tokens)}, sorting by index {last_token_idx}")

    # Second call sorted by last token
    data = search_all(prompt_text, sort_index=last_token_idx)

    all_features = []
    selected_count = 0

    for feat in data.get("result", []):
        feat_idx = str(feat.get("index"))
        max_value = feat.get("maxValue", 0.0)
        values = feat.get("values", [])

        try:
            details = get_feature_details(feat_idx)
            frac_nonzero = details.get("frac_nonzero", 1.0)
            label = ""
            if details.get("explanations"):
                label = details["explanations"][0].get("description", "")
            acts = details.get("activations", [])
            max_text = ""
            if acts:
                max_text = "".join(
                    t.replace("▁", " ").replace("<0x0A>", "\n")
                    for t in acts[0].get("tokens", [])
                ).strip()

            selected = frac_nonzero <= DENSITY_THRESHOLD and selected_count < NUM_FEATURES
            if selected:
                selected_count += 1

            all_features.append({
                "rank": len(all_features) + 1,
                "index": feat_idx,
                "label": label,
                "max_text": max_text[:200],
                "max_value": max_value,
                "values_at_tokens": values,
                "frac_nonzero": frac_nonzero,
                "selected": selected,
            })
            time.sleep(0.1)
        except Exception as e:
            print(f"  feat {feat_idx} error: {e}")
            all_features.append({
                "rank": len(all_features) + 1,
                "index": feat_idx,
                "max_value": max_value,
                "error": str(e),
                "selected": False,
            })

    print(f"  {len(all_features)} features fetched, {selected_count} selected")
    return all_features


def main():
    with open(HARD10_PATH) as f:
        prompts = json.load(f)

    results = []
    for item in prompts:
        pid = item["id"]
        prompt = item["prompt"]
        features = query_prompt(pid, prompt)
        results.append({
            "prompt_id": pid,
            "prompt": prompt,
            "features": features,
        })
        # Save incrementally
        with open(OUTPUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {OUTPUT_PATH}")

    print(f"\nDone. {len(results)} prompts saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
