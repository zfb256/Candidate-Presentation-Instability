#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python_bin=${PYTHON_BIN:-/root/.venvs/research-gpu/bin/python}
model_dir=${MODEL_ROOT:-/root/models}/Qwen2.5-14B-Instruct
model_name=Qwen2.5-14B-Instruct
port=${VLLM_PORT:-8768}
output=runs/formal/p5/qwen2.5-14b.jsonl
mkdir -p "$(dirname "$output")"

"$python_bin" -m vllm.entrypoints.openai.api_server \
  --model "$model_dir" --served-model-name "$model_name" \
  --dtype bfloat16 --gpu-memory-utilization 0.9 --max-model-len 4096 \
  --seed 20260823 --port "$port" &
server_pid=$!
cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT

for _ in {1..120}; do
  curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null && break
  kill -0 "$server_pid" 2>/dev/null || { echo "vLLM server exited during startup" >&2; exit 1; }
  sleep 2
done
curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null

export LOCAL_VLLM_API_KEY=local
resume=()
[[ -e "$output" || -e "$output.manifest.json" ]] && resume=(--resume)
"$python_bin" -B scripts/run_two_stage_selective.py \
  --items data/processed/cross_task_v3_items.jsonl --output "$output" \
  --model "$model_name" --base-url "http://127.0.0.1:$port/v1" \
  --api-key-env LOCAL_VLLM_API_KEY --workers 20 --max-tokens 512 \
  --no-json-mode \
  --threshold 0.7 "${resume[@]}"
"$python_bin" -B scripts/evaluate_outputs.py --outputs "$output" \
  --baseline-outputs runs/formal/p4/qwen2.5-14b.jsonl --bootstrap-rounds 10000 \
  > runs/formal/p5/qwen2.5-14b.evaluation.json
