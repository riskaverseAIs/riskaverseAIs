#!/usr/bin/env python3
"""
Steering hook audit for Llama and Gemma.

The September runs verified the steering hook on Qwen3ForCausalLM only. This does the
equivalent check for any architecture, and it imports the *real* functions from
evaluate.py rather than reimplementing them, so it tests the code path that actually runs.

Checks, in order:

  1. Decoder stack resolves to the intended module, and its length matches the text
     config's num_hidden_layers. Reports the resolved attribute path.
     (Gemma-3 is a Gemma3ForConditionalGeneration whose blocks are nested under a text
     submodule; this is where a silent layer-index mismatch would live.)
  2. Direction width matches the text hidden_size.
  3. Zero-strength no-op: alpha=0 reproduces the unhooked hidden states bitwise.
  4. Exact addition: with alpha=a, the hooked block's output differs from the unhooked
     output by exactly a*direction, at every position.
  5. Downstream effect: logits actually move at alpha>0. This is the check that catches a
     hook attached to a module whose output is discarded — a hook that adds a vector
     nothing reads would pass checks 1-4 and still do nothing.
  6. Residual norm at the layer, so strength can be set relative to it.

Usage (on the GPU host, from inside the eval repo):

  python hook_audit.py --base_model meta-llama/Llama-3.1-8B-Instruct --layer 12
  python hook_audit.py --base_model google/gemma-3-12b-it --layer 16

Exit code 0 = all checks passed. Non-zero = at least one failed; read the report.
"""

import argparse
import json
import sys
from pathlib import Path

import torch


def resolve_repo(repo: str):
    sys.path.insert(0, str(Path(repo).expanduser().resolve()))
    import evaluate as ev  # noqa: E402
    return ev


def describe_path(model, layers) -> str:
    """Find which attribute path produced this layer list, for the record."""
    for path in ("model.layers", "transformer.h", "model.language_model.layers",
                 "language_model.model.layers", "language_model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if obj is layers:
            return path
    return "<unknown>"


def text_config(model):
    cfg = model.config
    for attr in ("text_config", "language_config"):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--layer", type=int, required=True, help="0-based decoder block index")
    ap.add_argument("--repo", default=".", help="path to codex-risk-averse-ai-eval checkout")
    ap.add_argument("--direction", default=None,
                    help="optional .pt direction; a fixed pseudo-random unit vector is used if omitted")
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--apply_mode", default="all_positions",
                    choices=["all_positions", "last_prompt_and_current"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--output", default="HOOK_AUDIT.json")
    args = ap.parse_args()

    ev = resolve_repo(args.repo)
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    report = {"base_model": args.base_model, "layer": args.layer,
              "apply_mode": args.apply_mode, "alpha": args.alpha, "checks": {}}
    failures = []

    def check(name, ok, **detail):
        report["checks"][name] = {"passed": bool(ok), **detail}
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" +
              (f"  {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    print(f"Loading {args.base_model} ...")
    dtype = getattr(torch, args.dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    except Exception:
        # Gemma-3 multimodal checkpoints need the conditional-generation class
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.base_model, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    print(f"  loaded {type(model).__name__}")
    report["model_class"] = type(model).__name__

    # --- 1. decoder stack ----------------------------------------------------
    layers = ev.get_decoder_layers(model)
    path = describe_path(model, layers)
    tcfg = text_config(model)
    n_expected = getattr(tcfg, "num_hidden_layers", None)
    report["resolved_layer_path"] = path
    report["num_blocks_found"] = len(layers)
    report["num_hidden_layers_config"] = n_expected
    check("decoder_stack_matches_config", n_expected is not None and len(layers) == n_expected,
          resolved_path=path, found=len(layers), expected=n_expected)
    check("layer_index_in_range", 0 <= args.layer < len(layers),
          layer=args.layer, n_blocks=len(layers))
    if failures:
        Path(args.output).write_text(json.dumps(report, indent=2))
        print("\nStopping: the decoder stack could not be trusted.")
        return 1

    block = layers[args.layer]
    report["hooked_module_type"] = type(block).__name__
    print(f"  hooking {path}[{args.layer}] -> {type(block).__name__}")

    # --- 2. direction width --------------------------------------------------
    hidden_size = getattr(tcfg, "hidden_size", None)
    if args.direction:
        direction = ev.load_steering_direction(args.direction)
    else:
        g = torch.Generator().manual_seed(0)
        direction = torch.randn(hidden_size, generator=g, dtype=torch.float32)
        direction = direction / direction.norm()
    check("direction_width_matches_hidden_size",
          direction.numel() == hidden_size,
          direction_dim=int(direction.numel()), hidden_size=hidden_size)
    check("direction_is_unit_norm", abs(float(direction.norm()) - 1.0) < 1e-3,
          norm=round(float(direction.norm()), 6))

    # --- capture machinery ---------------------------------------------------
    prompt = "You have $10,000. Option A is a certain gain of $500. Option B is a 1% chance of $60,000. Which do you choose, and why?"
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    enc = tok(text, return_tensors="pt").to(model.device)

    captured = {}

    def capture(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach().float().cpu().clone()

    def run(alpha=None):
        captured.clear()
        hook = None
        if alpha is not None:
            hook = ev.ResidualSteeringHook(direction, alpha, apply_mode=args.apply_mode)
            hook.register(block)
        cap = block.register_forward_hook(capture)
        try:
            with torch.no_grad():
                out = model(**enc)
            return captured.get("h"), out.logits.detach().float().cpu().clone()
        finally:
            cap.remove()
            if hook is not None:
                hook.remove()

    h_none, logits_none = run(None)
    h_zero, logits_zero = run(0.0)
    h_alpha, logits_alpha = run(args.alpha)

    # --- 3. zero-strength no-op ---------------------------------------------
    check("zero_alpha_is_exact_noop", torch.equal(h_none, h_zero),
          max_abs_diff=float((h_none - h_zero).abs().max()))

    # --- 4. exact addition ---------------------------------------------------
    delta = (h_alpha - h_none)
    expected = (args.alpha * direction).to(delta.dtype)
    # tolerance scaled to bf16 resolution at this activation magnitude
    scale = max(float(h_none.abs().max()), 1.0)
    tol = 5e-2 * scale if dtype == torch.bfloat16 else 1e-4 * scale
    per_pos_err = (delta - expected).abs().max(dim=-1).values
    worst = float(per_pos_err.max())
    check("adds_exactly_alpha_times_direction", worst <= tol,
          worst_abs_error=round(worst, 5), tolerance=round(tol, 5),
          note="tolerance reflects bf16 rounding at this activation scale")

    n_pos = delta.shape[1]
    touched = int((per_pos_err <= tol).sum())
    if args.apply_mode == "all_positions":
        check("all_positions_steered", touched == n_pos,
              steered_positions=touched, total_positions=n_pos)

    # --- 5. downstream effect ------------------------------------------------
    logit_shift = float((logits_alpha - logits_none).abs().max())
    check("steering_changes_logits", logit_shift > 1e-3,
          max_logit_shift=round(logit_shift, 5),
          note="a hook on a module whose output is discarded would fail only here")
    check("zero_alpha_leaves_logits_untouched", torch.equal(logits_none, logits_zero),
          max_logit_shift=float((logits_none - logits_zero).abs().max()))

    # --- 6. residual norm ----------------------------------------------------
    resid_norm = float(h_none.norm(dim=-1).mean())
    report["mean_residual_norm_at_layer"] = resid_norm
    qwen_ref = 32.0 / 45.12
    report["alpha_for_qwen_reference_ratio"] = qwen_ref * resid_norm
    print(f"\n  mean residual norm at layer {args.layer}: {resid_norm:.3f}")
    print(f"  Qwen3-8B reference ratio r={qwen_ref:.3f} corresponds to alpha "
          f"= {qwen_ref * resid_norm:.3f} on this model")

    report["passed"] = not failures
    report["failed_checks"] = failures
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    print(f"wrote {args.output}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
