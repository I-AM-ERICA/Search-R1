#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
policy_python="${POLICY_PYTHON:-/root/anaconda3/envs/naver/bin/python}"
exec 9>/tmp/search-r1-policy-serving.lock
flock -n 9 || { echo 'Policy serving is already running; no process stopped.'; exit 1; }
gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
[[ -z "${gpu_pids//[[:space:]]/}" ]] || { echo 'GPU occupied; no process stopped.'; exit 1; }
checkpoint="$($policy_python -c 'import json; print(json.load(open("serving/model-selection.json"))["checkpoint"])')"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec "$policy_python" -u scripts/serve_policy.py --checkpoint "$checkpoint" \
  --pipeline mvp --search-engine naver --port "${POLICY_GRPC_PORT:-50051}"
