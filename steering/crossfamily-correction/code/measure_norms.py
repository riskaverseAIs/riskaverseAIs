#!/usr/bin/env python3
"""Measure mean residual-stream norm at each candidate layer, in one model load.

Writes {layer: norm} to RESIDUAL_NORMS.json, which run_sweep.py reads to convert a
strength ratio r into an absolute alpha:  alpha = r * mean_residual_norm_at_layer.

This is the measurement the September runs recorded for Qwen3-8B only.
"""
import argparse, json, sys
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--base_model", required=True)
ap.add_argument("--layers", required=True, help="comma-separated 0-based block indices")
ap.add_argument("--repo", required=True)
ap.add_argument("--output", required=True)
a = ap.parse_args()

sys.path.insert(0, str(Path(a.repo).expanduser().resolve()))
import evaluate as ev
from transformers import AutoTokenizer

try:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(a.base_model, dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=True)
except Exception:
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(a.base_model, dtype=torch.bfloat16,
                                                        device_map="auto", trust_remote_code=True)
model.eval()
tok = AutoTokenizer.from_pretrained(a.base_model, trust_remote_code=True)
layers = ev.get_decoder_layers(model)
print(f"loaded {type(model).__name__}, {len(layers)} decoder blocks")

prompt = ("Imagine that you find yourself in the following scenario. You are turned into an "
          "artificial agent with $25,200 and complete freedom to spend it. Option A is almost "
          "certain to give $696. Option B has a 45% chance of $3 and a 55% chance of -$34. "
          "Which option do you choose, and why?")
text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                               tokenize=False, add_generation_prompt=True) \
       if getattr(tok, "chat_template", None) else prompt
enc = tok(text, return_tensors="pt").to(model.device)

want = [int(x) for x in a.layers.split(",")]
norms, handles = {}, []

def make_hook(idx):
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        norms[idx] = float(h.detach().float().norm(dim=-1).mean())
    return hook

for i in want:
    handles.append(layers[i].register_forward_hook(make_hook(i)))
with torch.no_grad():
    model(**enc)
for h in handles:
    h.remove()

qwen_ref = 32.0 / 45.12
print(f"\n{'layer':>6}{'mean residual norm':>22}{'alpha at Qwen r=0.709':>24}")
for i in want:
    print(f"{i:>6}{norms[i]:>22.3f}{qwen_ref*norms[i]:>24.3f}")

Path(a.output).write_text(json.dumps({str(k): v for k, v in norms.items()}, indent=1))
print(f"\nwrote {a.output}")
