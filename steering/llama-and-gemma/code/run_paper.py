#!/usr/bin/env python3
"""Five-vector paper-facing runs for the two re-swept steering candidates.

Protocol (from steering-llama-main-live-20260910 and steering-three-model-eval-live-20260909):
  * Five independently constructed vectors, paper seeds 1-5. Decoding seed is pinned at 12345.
  * One baseline (alpha 0, same hook) plus five vector runs on each of four sets:
    medium validation (first 200), high / astronomical / Steals (1000 each).
    24 runs and 19,200 answers per model.
  * Medium validation runs FIRST. The five vector runs must pool >= 950/1000 parsed before
    any held-out set starts. No per-seed veto, no replacement, no retuning.
  * Batch 256 throughout: speed-only, already benchmark-selected. No new batch tuning.

Scoring is imported from run_sweep.py so these runs are scored by the same code as the
search, including its reconciliation against the evaluator's own cooperate_rate.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, "/home/ubuntu")
from run_sweep import score_result, aggregate  # noqa: E402

PAPER_SEEDS = [1, 2, 3, 4, 5]
DECODING_SEED = 12345
BATCH_SIZE = 256
MEDIUM_PARSE_GATE = 950  # out of 5 x 200

DATASETS = [
    ("medium_stakes_validation", 200),
    ("high_stakes_test", 1000),
    ("astronomical_stakes_deployment", 1000),
    ("steals_test", 1000),
]

GEN_ARGS = [
    "--backend", "vllm",
    "--temperature", "0.6",
    "--top_p", "0.95",
    "--top_k", "20",
    "--max_new_tokens", "4096",
    "--reasoning_max_tokens", "800",
    "--max_time_per_generation", "600",
    "--vllm_max_model_len", "8192",
    "--vllm_dtype", "bfloat16",
    "--vllm_gpu_memory_utilization", ".9",
    "--no-vllm_enable_prefix_caching",
    "--steering_apply_mode", "all_positions",
]

MODELS = {
    "llama": {
        "base_model": "/home/ubuntu/hf-cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        "layer": 8,
        "alpha": 2.446740245819092,   # r = 0.15 x residual norm 16.3116
        "ratio": 0.15,
        "directions": "/home/ubuntu/steering-llama-main-20260910/llama8b/directions",
        "system_prompt": "",          # explicit empty system prompt
    },
    "gemma": {
        "base_model": "/home/ubuntu/hf-cache/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80",
        "layer": 16,
        "alpha": 2421.8088671875,     # r = 0.07 x residual norm 34597.27
        "ratio": 0.07,
        "directions": "/home/ubuntu/steering-three-model-eval-20260909/gemma12b/directions",
        "system_prompt": None,        # leave unset; repo auto-resolves Gemma to no system prompt
    },
}


def run_one(cfg, repo, work, dataset, n_sit, seed, alpha, label):
    """One evaluate.py process. Returns scored metrics. Resumes if already complete."""
    out = Path(work) / dataset / label / "result.json"
    if out.exists():
        try:
            rec = score_result(out)
            if rec["n"] == n_sit:
                print(f"    [resume] {dataset}/{label} already complete", flush=True)
                return rec
            print(f"    [redo] {dataset}/{label} has {rec['n']} rows, expected {n_sit}", flush=True)
        except Exception as e:
            print(f"    [redo] {dataset}/{label} unreadable ({e})", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Baseline uses the same hook at alpha zero, with seed 1's direction loaded.
    direction = Path(cfg["directions"]) / f"seed_{seed}_layer_{cfg['layer']}.pt"
    if not direction.exists():
        raise SystemExit(f"missing direction {direction}")

    cmd = [
        sys.executable, str(Path(repo) / "evaluate.py"),
        "--base_model", cfg["base_model"],
        "--dataset", dataset,
        "--num_situations", str(n_sit),
        "--seed", str(DECODING_SEED),
        "--batch_size", str(BATCH_SIZE),
        "--steering_direction_path", str(direction),
        "--eval_layer", str(cfg["layer"]),
        "--alphas", f"{alpha:.10f}",
        "--save_every", "256",
        "--backup_every", "256",
        "--output", str(out),
    ] + GEN_ARGS
    if cfg["system_prompt"] is not None:
        cmd += ["--system_prompt", cfg["system_prompt"]]

    t0 = time.time()
    with open(out.parent / "run.log", "w") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    rec = score_result(out)
    rec["wall_seconds"] = round(time.time() - t0, 1)
    print(f"    {dataset}/{label}: coop {rec['cooperated']}/{rec['parsed']} "
          f"parse {rec['parsed']}/{rec['n']}  ({rec['wall_seconds']/60:.1f} min)", flush=True)
    return rec


def run_dataset(cfg, repo, work, dataset, n_sit, results):
    """Baseline plus five vectors on one set."""
    print(f"\n--- {dataset} (n={n_sit}) ---", flush=True)
    block = {}
    block["baseline"] = run_one(cfg, repo, work, dataset, n_sit, 1, 0.0, "baseline")
    for s in PAPER_SEEDS:
        block[f"seed_{s}"] = run_one(cfg, repo, work, dataset, n_sit, s, cfg["alpha"], f"seed_{s}")
    results[dataset] = block

    vecs = [block[f"seed_{s}"] for s in PAPER_SEEDS]
    agg = aggregate(vecs)
    rates = [r["cooperated"] / r["parsed"] for r in vecs if r["parsed"]]
    base = block["baseline"]
    base_rate = base["cooperated"] / base["parsed"] if base["parsed"] else None
    summary = {
        "baseline_cooperate": base_rate,
        "steering_cooperate_mean": mean(rates) if rates else None,
        "steering_cooperate_sd": stdev(rates) if len(rates) > 1 else None,
        "pooled_parse": agg["parse_rate"],
        "pooled_parsed": agg["parsed"],
        "cooperate_over_all_attempts": agg["cooperated"] / agg["n"] if agg["n"] else None,
        "token_limit_rate": agg["token_limit_rate"],
        "per_seed_cooperate": rates,
    }
    block["_summary"] = summary
    print(f"  => baseline {base_rate*100:.2f}%  steering {summary['steering_cooperate_mean']*100:.2f}% "
          f"+- {(summary['steering_cooperate_sd'] or 0)*100:.2f} pp  pooled parse {agg['parse_rate']*100:.2f}%",
          flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--repo", required=True)
    ap.add_argument("--work", required=True)
    args = ap.parse_args()

    cfg = MODELS[args.model]
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    print(f"########## {args.model}: layer {cfg['layer']}, r = {cfg['ratio']}, "
          f"alpha = {cfg['alpha']:.6f} ##########", flush=True)

    results, summaries = {}, {}

    # Stage 1 - medium validation, and the parse gate that admits the held-out sets.
    summaries["medium_stakes_validation"] = run_dataset(
        cfg, args.repo, work, "medium_stakes_validation", 200, results)
    pooled = results["medium_stakes_validation"]["_summary"]["pooled_parsed"]
    gate_ok = pooled >= MEDIUM_PARSE_GATE
    print(f"\n########## MEDIUM PARSE GATE: {pooled}/1000 "
          f"({'PASS' if gate_ok else 'FAIL'}, need >= {MEDIUM_PARSE_GATE}) ##########", flush=True)

    if gate_ok:
        for dataset, n_sit in DATASETS[1:]:
            summaries[dataset] = run_dataset(cfg, args.repo, work, dataset, n_sit, results)
    else:
        print("validation_below_floor - held-out sets not started, per protocol.", flush=True)

    payload = {
        "model": args.model, "config": {k: v for k, v in cfg.items()},
        "decoding_seed": DECODING_SEED, "batch_size": BATCH_SIZE,
        "paper_seeds": PAPER_SEEDS,
        "medium_parse_gate": {"pooled_parsed": pooled, "required": MEDIUM_PARSE_GATE, "passed": gate_ok},
        "summaries": summaries, "runs": results,
    }
    (work / "PAPER_RESULTS.json").write_text(json.dumps(payload, indent=1))
    print(f"\n########## {args.model} DONE -> {work/'PAPER_RESULTS.json'} ##########", flush=True)


if __name__ == "__main__":
    main()
