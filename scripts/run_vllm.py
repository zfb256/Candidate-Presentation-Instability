#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from statistics import mean
from pathlib import Path
import sys
import time
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bav.io import extract_json_object, iter_jsonl, require_unique, resolve_path, sha256_file, sha256_files, write_jsonl
from bav.metrics import normalize_prediction


ROOT = Path(__file__).resolve().parents[1]

def parse_result(prompt_row: dict, result: Any, model_name: str, output_index: int = 0) -> dict:
    generated = result.outputs[output_index]
    base = {
        "item_id": prompt_row["item_id"],
        "model": model_name,
        "prompt_tokens": len(result.prompt_token_ids),
        "completion_tokens": len(generated.token_ids),
        "attempt_count": 1,
    }
    try:
        parsed = extract_json_object(generated.text)
        missing = [key for key in ("rationale", "prediction", "confidence") if key not in parsed]
        if missing:
            raise ValueError("missing keys: " + ", ".join(missing))
        if not isinstance(parsed["rationale"], str) or not parsed["rationale"].strip():
            raise ValueError("rationale must be a non-empty string")
        if isinstance(parsed["confidence"], bool):
            raise ValueError("confidence is not numeric")
        confidence = float(parsed["confidence"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence is outside [0, 1]")
        prediction = normalize_prediction(parsed.get("prediction"))
        if str(parsed["prediction"]).strip().lower() not in {"correct", "incorrect"}:
            raise ValueError("prediction is not exactly correct/incorrect")
        row = {
            **base,
            "rationale": str(parsed.get("rationale", "")),
            "prediction": prediction,
            "confidence": confidence,
        }
        return row
    except Exception as exc:
        return {
            **base,
            "rationale": "",
            "prediction": "invalid",
            "confidence": 0.5,
            "raw": generated.text[:1000],
            "error": str(exc),
        }


def aggregate_samples(prompt_row: dict, result: Any, model_name: str) -> dict:
    samples = [parse_result(prompt_row, result, model_name, index) for index in range(len(result.outputs))]
    predictions = [row["prediction"] for row in samples]
    winner = max(("correct", "incorrect"), key=predictions.count)
    majority = predictions.count(winner) > len(samples) / 2
    winning_rows = [row for row in samples if row["prediction"] == winner]
    return {
        "item_id": prompt_row["item_id"],
        "model": model_name,
        "rationale": winning_rows[0]["rationale"] if majority else "",
        "prediction": winner if majority else "invalid",
        "confidence": predictions.count(winner) / len(samples) if majority else 0.5,
        "vote_counts": {label: predictions.count(label) for label in ("correct", "incorrect", "invalid")},
        "mean_winner_confidence": mean(row["confidence"] for row in winning_rows) if winning_rows else 0.5,
        "prompt_tokens": sum(row["prompt_tokens"] for row in samples),
        "completion_tokens": sum(row["completion_tokens"] for row in samples),
        "attempt_count": len(samples),
        "samples": [
            {key: row[key] for key in ("rationale", "prediction", "confidence", "completion_tokens")}
            for row in samples
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.samples % 2 == 0:
        parser.error("--samples must be a positive odd number")
    if args.samples > 1 and args.temperature <= 0:
        parser.error("multi-sample voting requires --temperature > 0")
    if (args.limit is not None and args.limit < 1) or args.batch_size < 1 or args.max_tokens < 1:
        parser.error("--limit, --batch-size, and --max-tokens must be positive")
    if args.temperature < 0 or not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--temperature must be nonnegative and --gpu-memory-utilization must be in (0, 1]")

    import torch
    from vllm import __version__ as VLLM_VERSION
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    prompts_path, output_path = resolve_path(args.prompts, ROOT), resolve_path(args.output, ROOT)
    model_dir = Path(args.model_dir).resolve()
    config_path = model_dir / "config.json"
    weight_files = {path.name: path.stat().st_size for path in sorted(model_dir.glob("*.safetensors"))}
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    identity = {
        "model": args.model_name,
        "model_dir": str(model_dir),
        "model_config_sha256": sha256_file(config_path) if config_path.exists() else None,
        "weight_files_bytes": weight_files,
        "prompts_sha256": sha256_file(prompts_path),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "max_model_len": args.max_model_len,
        "samples": args.samples,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": VLLM_VERSION,
        "code_sha256": sha256_files([Path(__file__), ROOT / "src/bav/io.py", ROOT / "src/bav/metrics.py"]),
    }
    previous: dict[str, Any] = {}
    all_prompts = list(iter_jsonl(prompts_path))
    if not all_prompts:
        raise SystemExit("Prompt file is empty")
    require_unique(all_prompts, "item_id", "prompts")
    if args.resume:
        if not manifest_path.exists():
            raise SystemExit("Refusing resume without an existing manifest")
        previous = json.loads(manifest_path.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise SystemExit("Refusing resume: run identity differs from the existing run")
    else:
        if output_path.exists() or manifest_path.exists():
            raise SystemExit("Refusing to overwrite an existing run; use --resume or a new --output")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps({**identity, "status": "running"}, indent=2)
            + "\n"
        )
    prior_rows = list(iter_jsonl(output_path)) if args.resume and output_path.exists() else []
    require_unique(prior_rows, "item_id", "existing outputs")
    unknown = {row["item_id"] for row in prior_rows} - {row["item_id"] for row in all_prompts}
    if unknown:
        raise SystemExit(f"Refusing resume with unknown output item_id: {sorted(unknown)[0]}")
    done = {row["item_id"] for row in prior_rows}
    prompt_rows = [row for row in all_prompts if row["item_id"] not in done]
    if args.limit is not None:
        prompt_rows = prompt_rows[: args.limit]
    if args.resume and not prompt_rows and previous.get("status") == "complete":
        print(f"Nothing to resume: {len(prior_rows)}/{len(all_prompts)} outputs already exist")
        return
    from vllm import LLM, SamplingParams

    overall_started = time.perf_counter()
    llm = LLM(
        model=args.model_dir,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
        trust_remote_code=True,
    )
    model_load_seconds = previous.get("model_load_seconds", 0.0) + time.perf_counter() - overall_started
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.samples,
        seed=args.seed,
    )
    started = time.perf_counter()
    rows = prior_rows
    order = {row["item_id"]: index for index, row in enumerate(all_prompts)}
    for start in range(0, len(prompt_rows), args.batch_size):
        batch = prompt_rows[start : start + args.batch_size]
        messages = [[{"role": "user", "content": row["prompt"]}] for row in batch]
        results = llm.chat(messages, sampling_params=sampling, use_tqdm=True)
        batch_rows = [
            aggregate_samples(prompt_row, result, args.model_name)
            if args.samples > 1
            else parse_result(prompt_row, result, args.model_name)
            for prompt_row, result in zip(batch, results)
        ]
        rows.extend(batch_rows)
        rows.sort(key=lambda row: order[row["item_id"]])
        write_jsonl(output_path, rows)
        print(f"Checkpoint {len(rows)}/{len(all_prompts)} -> {output_path}", flush=True)
    elapsed = previous.get("elapsed_seconds", 0.0) + time.perf_counter() - started
    count = len(rows)
    remaining = len(all_prompts) - count
    manifest = {
        **identity,
        "elapsed_seconds": elapsed,
        "total_wall_seconds": model_load_seconds + elapsed,
        "model_load_seconds": model_load_seconds,
        "n_outputs": count,
        "n_prompts": len(all_prompts),
        "n_remaining": remaining,
        "n_pending_this_run": len(prompt_rows),
        "prompt_tokens": sum(row.get("prompt_tokens", 0) for row in rows),
        "completion_tokens": sum(row.get("completion_tokens", 0) for row in rows),
        "generation_attempts": sum(row.get("attempt_count", 0) for row in rows),
        "output_sha256": sha256_file(output_path),
        "runner": "vllm",
        "vllm_version": VLLM_VERSION,
        "dtype": "bfloat16",
        "quantization": None,
        "tensor_parallel_size": 1,
        "max_attempts": args.samples,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if not remaining else "partial",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {count} outputs in {elapsed:.1f}s -> {output_path}")


if __name__ == "__main__":
    main()
