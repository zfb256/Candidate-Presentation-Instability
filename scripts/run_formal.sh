#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
P1=runs/p1_direct_front/prompts.jsonl
SEED=20260823

resume_run=false
case ${1:-} in
  "") ;;
  --resume) resume_run=true ;;
  *) echo "Usage: $0 [--resume]" >&2; exit 2 ;;
esac
if [[ -e runs/formal && $resume_run == false ]]; then
  echo "Refusing to reuse runs/formal; remove it for a clean run or pass --resume." >&2
  exit 1
fi

set -a
source .env
set +a
PY=${PYTHON_BIN:-/root/.venvs/research-gpu/bin/python}
MODEL_ROOT=${MODEL_ROOT:-/root/models}
: "${DEEPSEEK_BASE_URL:?Missing DEEPSEEK_BASE_URL}"
: "${DEEPSEEK_API_KEY:?Missing DEEPSEEK_API_KEY}"
: "${ARK_BASE_URL:?Missing ARK_BASE_URL}"
: "${ARK_API_KEY:?Missing ARK_API_KEY}"
for config in \
  "$MODEL_ROOT/Qwen2.5-1.5B-Instruct/config.json" \
  "$MODEL_ROOT/Qwen2.5-3B-Instruct/config.json" \
  "$MODEL_ROOT/Qwen2.5-7B-Instruct/config.json" \
  "$MODEL_ROOT/Qwen2.5-14B-Instruct/config.json" \
  "$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-7B/config.json" \
  "$MODEL_ROOT/internlm2_5-7b-chat/config.json"; do
  [[ -s "$config" ]] || { echo "Missing model config: $config" >&2; exit 1; }
done
"$PY" -B -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else "CUDA unavailable")'
run_local() {
  local protocol=$1 slug=$2 model_dir=$3 model_name=$4 prompts=$5 baseline=${6:-}
  local output="runs/formal/$protocol/$slug.jsonl"
  local resume=()
  [[ $resume_run == true && ( -e "$output" || -e "$output.manifest.json" ) ]] && resume=(--resume)
  echo "$(date -Is) START $protocol $slug"
  "$PY" -B scripts/run_vllm.py \
    --model-dir "$model_dir" --model-name "$model_name" \
    --prompts "$prompts" --output "$output" \
    --max-tokens 512 --batch-size 500 --max-model-len 4096 \
    --gpu-memory-utilization 0.9 --seed "$SEED" "${resume[@]}"
  local baseline_args=()
  [[ -n "$baseline" ]] && baseline_args=(--baseline-outputs "$baseline")
  "$PY" -B scripts/evaluate_outputs.py --outputs "$output" \
    "${baseline_args[@]}" --bootstrap-rounds 1000 \
    > "runs/formal/$protocol/$slug.evaluation.json"
  echo "$(date -Is) DONE $protocol $slug"
}

run_api() {
  local protocol=$1 slug=$2 model=$3 base_url_env=$4 api_key_env=$5 prompts=$6 workers=$7 baseline=${8:-}
  local output="runs/formal/$protocol/$slug.jsonl"
  local resume=()
  [[ $resume_run == true && ( -e "$output" || -e "$output.manifest.json" ) ]] && resume=(--resume)
  echo "$(date -Is) START $protocol $slug"
  "$PY" -B scripts/run_openai_compatible.py \
    --prompts "$prompts" --output "$output" --model "$model" \
    --base-url-env "$base_url_env" --api-key-env "$api_key_env" \
    --thinking disabled --workers "$workers" --max-tokens 512 "${resume[@]}"
  local baseline_args=()
  [[ -n "$baseline" ]] && baseline_args=(--baseline-outputs "$baseline")
  "$PY" -B scripts/evaluate_outputs.py --outputs "$output" \
    "${baseline_args[@]}" --bootstrap-rounds 1000 \
    > "runs/formal/$protocol/$slug.evaluation.json"
  echo "$(date -Is) DONE $protocol $slug"
}

run_depth_model() {
  local slug=$1 model_dir=$2 model_name=$3
  run_local p1 "$slug" "$model_dir" "$model_name" "$P1"
  run_local p2 "$slug" "$model_dir" "$model_name" runs/p2_direct_last/prompts.jsonl "runs/formal/p1/$slug.jsonl"
  run_local p3 "$slug" "$model_dir" "$model_name" runs/p3_careful_direct_front/prompts.jsonl "runs/formal/p1/$slug.jsonl"
  run_local p4 "$slug" "$model_dir" "$model_name" runs/p4_constraint_first_last/prompts.jsonl "runs/formal/p2/$slug.jsonl"

  local calibration="runs/formal/p4/$slug.p6-calibration.json"
  "$PY" -B scripts/calibrate_p6.py --outputs "runs/formal/p4/$slug.jsonl" --output "$calibration" >/dev/null
  local target_words
  target_words=$(awk -F ': ' '/"target_rationale_words_for_p6"/ {gsub(/,/, "", $2); print $2}' "$calibration")
  "$PY" -B scripts/export_prompts.py --protocol p6_length_matched_direct \
    --target-rationale-words "$target_words" --output "runs/formal/p6/prompts/$slug.jsonl"
  run_local p6 "$slug" "$model_dir" "$model_name" "runs/formal/p6/prompts/$slug.jsonl" "runs/formal/p1/$slug.jsonl"
}

run_sc5() {
  local slug=$1 model_dir=$2 model_name=$3
  local output="runs/formal/sc5/$slug.jsonl" resume=()
  [[ $resume_run == true && ( -e "$output" || -e "$output.manifest.json" ) ]] && resume=(--resume)
  echo "$(date -Is) START sc5 $slug"
  "$PY" -B scripts/run_vllm.py \
    --model-dir "$model_dir" --model-name "$model_name" \
    --prompts "$P1" --output "$output" --max-tokens 512 \
    --temperature 0.7 --samples 5 --batch-size 250 --max-model-len 4096 \
    --gpu-memory-utilization 0.9 --seed "$SEED" "${resume[@]}"
  "$PY" -B scripts/evaluate_outputs.py --outputs "$output" \
    --baseline-outputs "runs/formal/p1/$slug.jsonl" --bootstrap-rounds 1000 \
    > "runs/formal/sc5/$slug.evaluation.json"
  echo "$(date -Is) DONE sc5 $slug"
}

run_local p1 qwen2.5-1.5b "$MODEL_ROOT/Qwen2.5-1.5B-Instruct" Qwen2.5-1.5B-Instruct "$P1"
run_depth_model qwen2.5-3b "$MODEL_ROOT/Qwen2.5-3B-Instruct" Qwen2.5-3B-Instruct
run_depth_model qwen2.5-14b "$MODEL_ROOT/Qwen2.5-14B-Instruct" Qwen2.5-14B-Instruct
run_sc5 qwen2.5-14b "$MODEL_ROOT/Qwen2.5-14B-Instruct" Qwen2.5-14B-Instruct
run_local p1 qwen2.5-7b "$MODEL_ROOT/Qwen2.5-7B-Instruct" Qwen2.5-7B-Instruct "$P1"
run_local p1 deepseek-r1-distill-qwen-7b "$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-7B" DeepSeek-R1-Distill-Qwen-7B "$P1"
run_local p1 internlm2.5-7b "$MODEL_ROOT/internlm2_5-7b-chat" InternLM2.5-7B-Chat "$P1"

run_api p1 deepseek-v4-flash deepseek-v4-flash DEEPSEEK_BASE_URL DEEPSEEK_API_KEY "$P1" 50
run_api p2 deepseek-v4-flash deepseek-v4-flash DEEPSEEK_BASE_URL DEEPSEEK_API_KEY runs/p2_direct_last/prompts.jsonl 50 runs/formal/p1/deepseek-v4-flash.jsonl
run_api p3 deepseek-v4-flash deepseek-v4-flash DEEPSEEK_BASE_URL DEEPSEEK_API_KEY runs/p3_careful_direct_front/prompts.jsonl 50 runs/formal/p1/deepseek-v4-flash.jsonl
run_api p4 deepseek-v4-flash deepseek-v4-flash DEEPSEEK_BASE_URL DEEPSEEK_API_KEY runs/p4_constraint_first_last/prompts.jsonl 50 runs/formal/p2/deepseek-v4-flash.jsonl
"$PY" -B scripts/calibrate_p6.py --outputs runs/formal/p4/deepseek-v4-flash.jsonl --output runs/formal/p4/deepseek-v4-flash.p6-calibration.json >/dev/null
deepseek_words=$(awk -F ': ' '/"target_rationale_words_for_p6"/ {gsub(/,/, "", $2); print $2}' runs/formal/p4/deepseek-v4-flash.p6-calibration.json)
"$PY" -B scripts/export_prompts.py --protocol p6_length_matched_direct --target-rationale-words "$deepseek_words" --output runs/formal/p6/prompts/deepseek-v4-flash.jsonl
run_api p6 deepseek-v4-flash deepseek-v4-flash DEEPSEEK_BASE_URL DEEPSEEK_API_KEY runs/formal/p6/prompts/deepseek-v4-flash.jsonl 50 runs/formal/p1/deepseek-v4-flash.jsonl

p5_output=runs/formal/p5/deepseek-v4-flash.jsonl
p5_resume=()
[[ $resume_run == true && ( -e "$p5_output" || -e "$p5_output.manifest.json" ) ]] && p5_resume=(--resume)
echo "$(date -Is) START p5 deepseek-v4-flash"
"$PY" -B scripts/run_two_stage_selective.py --items data/processed/cross_task_v3_items.jsonl \
  --output "$p5_output" --model deepseek-v4-flash \
  --base-url-env DEEPSEEK_BASE_URL --api-key-env DEEPSEEK_API_KEY \
  --thinking disabled --workers 50 --max-tokens 512 --threshold 0.7 "${p5_resume[@]}"
"$PY" -B scripts/evaluate_outputs.py --outputs "$p5_output" \
  --baseline-outputs runs/formal/p4/deepseek-v4-flash.jsonl --bootstrap-rounds 10000 \
  > runs/formal/p5/deepseek-v4-flash.evaluation.json
echo "$(date -Is) DONE p5 deepseek-v4-flash"

run_api p1 doubao-seed-2-1-pro-260628 doubao-seed-2-1-pro-260628 ARK_BASE_URL ARK_API_KEY "$P1" 5

"$PY" -B scripts/aggregate_evaluations.py
"$PY" -B scripts/summarize_api_cost.py --output reports/formal/api_cost.json >/dev/null
echo "$(date -Is) ALL FORMAL RUNS COMPLETE"
