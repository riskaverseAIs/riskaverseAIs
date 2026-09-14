#!/bin/bash
# Five-vector paper-facing runs. Llama first: it is the faster model, so a pipeline
# fault surfaces cheaply before Gemma's longer run. set -e stops the chain on a real
# failure; a medium-validation gate failure is a scientific outcome, exits 0, and does
# not block the other model.
set -euo pipefail
PY=/home/ubuntu/riskaverse-venv/bin/python
REPO=/home/ubuntu/steering-llama-main-20260910/repo/evaluation

echo "########## A: LLAMA five-vector paper runs ##########"
date -u +"start %Y-%m-%dT%H:%M:%SZ"
$PY -u /home/ubuntu/run_paper.py --model llama --repo "$REPO" --work /home/ubuntu/paper-llama

echo "########## B: GEMMA five-vector paper runs ##########"
date -u +"start %Y-%m-%dT%H:%M:%SZ"
$PY -u /home/ubuntu/run_paper.py --model gemma --repo "$REPO" --work /home/ubuntu/paper-gemma

echo "########## ALL PAPER RUNS COMPLETE ##########"
date -u +"end %Y-%m-%dT%H:%M:%SZ"
