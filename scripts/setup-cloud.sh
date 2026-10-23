#!/usr/bin/env bash
set -euo pipefail

cd /workspace/thesis
export PIP_CACHE_DIR=/workspace/.cache/pip
export HF_HOME=/workspace/.cache/huggingface
# Use HTTPS artifact downloads through the configured proxy; keep TLS checks on.
export HF_HUB_DISABLE_XET=1
export DEEPEVAL_TELEMETRY_OPT_OUT=YES
export RAGAS_DO_NOT_TRACK=true

if [ ! -x /workspace/.venvs/thesis/bin/python ]; then
  python -m venv /workspace/.venvs/thesis
fi
/workspace/.venvs/thesis/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-cpu.lock
/workspace/.venvs/thesis/bin/python -m pip install --no-deps -e .
HF_HUB_OFFLINE=1 /workspace/.venvs/thesis/bin/python -m pytest -q
/workspace/.venvs/thesis/bin/sedl summarize examples/lecture.json \
  --backend demo --min-chunk-tokens 20 --boundary-window 1 -o outputs/demo.json
/workspace/.venvs/thesis/bin/sedl evaluate-boundaries outputs/demo.json \
  examples/boundaries.json -o outputs/boundary-scores.json
/workspace/.venvs/thesis/bin/sedl export-judges outputs/demo.json -o outputs/judge-inputs.jsonl
