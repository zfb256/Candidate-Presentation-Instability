#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mode=${1:?Usage: $0 local|deepseek|doubao}
prompts=runs/probes/p1_direct_front_probe200.jsonl
output_dir=runs/formal/exact_repeat
python_bin=${PYTHON_BIN:-/root/.venvs/research-gpu/bin/python}
seed=20260823
mkdir -p "$output_dir"

resume_arg() {
  [[ -e "$1" || -e "$1.manifest.json" ]] && printf '%s\n' --resume
}

run_local() {
  local slug=$1 model_dir=$2 model_name=$3 output="$output_dir/$1.jsonl"
  mapfile -t resume < <(resume_arg "$output")
  "$python_bin" -B scripts/run_vllm.py \
    --model-dir "$model_dir" --model-name "$model_name" \
    --prompts "$prompts" --output "$output" \
    --max-tokens 512 --batch-size 500 --max-model-len 4096 \
    --gpu-memory-utilization 0.9 --seed "$seed" "${resume[@]}"
}

run_api() {
  local slug=$1 model=$2 base_url_env=$3 api_key_env=$4 workers=$5 output="$output_dir/$1.jsonl"
  mapfile -t resume < <(resume_arg "$output")
  "$python_bin" -B scripts/run_openai_compatible.py \
    --prompts "$prompts" --output "$output" --model "$model" \
    --base-url-env "$base_url_env" --api-key-env "$api_key_env" \
    --thinking disabled --workers "$workers" --max-tokens 512 "${resume[@]}"
}

case "$mode" in
  local)
    model_root=${MODEL_ROOT:-/root/models}
    run_local qwen2.5-1.5b "$model_root/Qwen2.5-1.5B-Instruct" Qwen2.5-1.5B-Instruct
    run_local qwen2.5-3b "$model_root/Qwen2.5-3B-Instruct" Qwen2.5-3B-Instruct
    run_local qwen2.5-7b "$model_root/Qwen2.5-7B-Instruct" Qwen2.5-7B-Instruct
    run_local qwen2.5-14b "$model_root/Qwen2.5-14B-Instruct" Qwen2.5-14B-Instruct
    ;;
  deepseek|doubao)
    set -a
    source .env
    set +a
    if [[ $mode == deepseek ]]; then
      run_api deepseek-v4-flash deepseek-v4-flash DEEPSEEK_BASE_URL DEEPSEEK_API_KEY 50
    else
      run_api doubao-seed-2-1-pro-260628 doubao-seed-2-1-pro-260628 ARK_BASE_URL ARK_API_KEY 5
    fi
    ;;
  *)
    echo "Usage: $0 local|deepseek|doubao" >&2
    exit 2
    ;;
esac
