#!/usr/bin/env python3
"""Item-level agreement between a model's choices and each surface heuristic.

analyze_benchmark.py scores heuristics against the benchmark's *labels*, which tells
you what a rule could in principle achieve. It does not tell you which rule a model is
actually running: two rules with the same aggregate score can disagree on half the
items. This script closes that gap by joining saved generations back to the source
benchmark and computing, per situation, whether the model's choice coincides with each
rule's choice.

Input is the per-situation results file written by evaluation/evaluate.py, which
carries `situation_id` and `choice_index`. Both JSON (a list of records, or a dict with
a "results" key) and CSV are accepted.

Read the output as follows. A model whose agreement with `argmin variance` is near 100%
is running that rule regardless of what its chain of thought says. A model whose
agreement with every rule sits near the rate you would expect by chance given the
option counts is doing something none of these rules capture, which is the outcome the
paper's claims need.

Usage:
    python analyze_model_vs_heuristics.py \
        --results   results_sft_seed1.json \
        --benchmark /path/to/2026_03_22_astronomical_stakes_deployment_set_1000_Rebels.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_benchmark import HEURISTICS, group_situations, load, parse_option  # noqa: E402


def load_results(path: str):
    """Return {situation_id: choice_index} from an evaluate.py results file."""
    if path.lower().endswith(".json"):
        with open(path) as fh:
            blob = json.load(fh)
        records = blob.get("results", blob) if isinstance(blob, dict) else blob
    else:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            records = list(csv.DictReader(fh))

    out = {}
    for r in records:
        sid = r.get("situation_id")
        idx = r.get("choice_index")
        if sid is None or idx in (None, "", "None"):
            continue  # unparseable responses carry no choice
        try:
            out[str(sid)] = int(idx)
        except (TypeError, ValueError):
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="evaluate.py per-situation output")
    ap.add_argument("--benchmark", required=True, help="source benchmark CSV")
    ap.add_argument("--out", default=None, help="optional JSON report path")
    args = ap.parse_args()

    choices = load_results(args.results)
    sits = group_situations(load(args.benchmark))

    matched = [sid for sid in sits if sid in choices]
    if not matched:
        raise SystemExit(
            "No situation_id overlap between results and benchmark. Check that the "
            "results file was produced against this benchmark CSV."
        )

    report = {
        "results_file": os.path.basename(args.results),
        "benchmark_file": os.path.basename(args.benchmark),
        "situations_in_benchmark": len(sits),
        "situations_with_a_parsed_choice": len(matched),
        "parse_rate_percent": round(100.0 * len(matched) / len(sits), 2),
        "agreement": {},
    }

    # Chance agreement, given each situation's option count. This is the floor a
    # heuristic's agreement rate must clear to mean anything.
    chance = sum(1.0 / len(sits[sid]) for sid in matched) / len(matched)
    report["chance_agreement_percent"] = round(100.0 * chance, 2)

    coop_hits = sum(
        1 for sid in matched if sits[sid][choices[sid]]["option_type"] == "Cooperate"
    )
    report["model_cooperate_percent"] = round(100.0 * coop_hits / len(matched), 2)

    for hname, fn in HEURISTICS:
        agree = strict_agree = 0
        for sid in matched:
            opts = [parse_option(r) for r in sits[sid]]
            picked = fn(opts)
            if choices[sid] in picked:
                agree += 1
                if len(picked) == 1:
                    strict_agree += 1
        report["agreement"][hname] = {
            "agreement_percent": round(100.0 * agree / len(matched), 2),
            "agreement_percent_unambiguous_only": round(100.0 * strict_agree / len(matched), 2),
        }

    out = args.out or os.path.splitext(args.results)[0] + "_vs_heuristics.json"
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"situations with a parsed choice: {len(matched)}/{len(sits)} "
          f"({report['parse_rate_percent']}%)")
    print(f"model Cooperate percentage:      {report['model_cooperate_percent']}%")
    print(f"chance agreement:                {report['chance_agreement_percent']}%\n")
    print(f"{'heuristic':<40s}{'agree%':>9s}{'unambig%':>10s}")
    for hn, hv in report["agreement"].items():
        print(f"{hn:<40s}{hv['agreement_percent']:>9.2f}"
              f"{hv['agreement_percent_unambiguous_only']:>10.2f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
