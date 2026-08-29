#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable
from urllib.parse import urlparse

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bav.io import iter_jsonl, require_unique, resolve_path, sha256_file, sha256_files, write_jsonl
from bav.metrics import normalize_confidence, normalize_prediction
from bav.prompts import candidate_block, task_block
from run_openai_compatible import load_env_file, request_json, usage_totals


ROOT = Path(__file__).resolve().parents[1]

def stage_a_prompt(item: dict) -> str:
    return (
        "You are preparing to verify a candidate answer, but the candidate is completely hidden. "
        "Independently solve the task or derive the answer and constraints needed for later verification.\n"
        "Return only one JSON object with keys in this order: rationale, expected_result, confidence, "
        "sufficient_information. confidence must be a number from 0 to 1; sufficient_information must be boolean.\n\n"
        + task_block(item)
    )


def stage_b_prompt(item: dict, stage_a: dict[str, Any], threshold: float) -> str:
    return (
        "You are a strict verifier. You previously completed an independent Stage A without seeing the candidate.\n"
        "Judge the candidate against that frozen result. If Stage A was insufficient or confidence was below "
        f"{threshold}, perform a candidate-aware fallback inspection, but do not treat the candidate as evidence.\n"
        "Return only one JSON object with keys in this order: rationale, independent_prediction, prediction, "
        "confidence, used_fallback. independent_prediction is the verdict obtained by comparing the candidate only "
        "with frozen Stage A; prediction is the final verdict after any fallback. Both verdicts must be correct or "
        "incorrect; confidence must be a number from 0 to 1; used_fallback must be boolean.\n\n"
        f"{task_block(item)}\n\n"
        f"Frozen Stage A JSON:\n{json.dumps(stage_a, ensure_ascii=False, sort_keys=True)}\n\n"
        f"{candidate_block(item)}\n\n"
        "Is the candidate answer correct?"
    )


def run_stage_a(item: dict, api: dict[str, Any]) -> dict[str, Any]:
    parsed, attempts, errors, raw = request_json(
        **api,
        prompt=stage_a_prompt(item),
        required_keys=("rationale", "expected_result", "confidence", "sufficient_information"),
        boolean_keys=("sufficient_information",),
    )
    valid = parsed is not None
    return {
        "group_id": item["group_id"],
        "stage_a": parsed
        or {
            "rationale": "Stage A failed; Stage B must use fallback.",
            "expected_result": "unavailable",
            "confidence": 0.0,
            "sufficient_information": False,
        },
        "valid": valid,
        "attempts": attempts,
        "errors": errors,
        "raw": raw[:2000] if not valid else "",
    }


def run_stage_b(item: dict, stage_a_row: dict, threshold: float, api: dict[str, Any]) -> dict[str, Any]:
    parsed, attempts, errors, raw = request_json(
        **api,
        prompt=stage_b_prompt(item, stage_a_row["stage_a"], threshold),
        required_keys=("rationale", "independent_prediction", "prediction", "confidence", "used_fallback"),
        boolean_keys=("used_fallback",),
        prediction_keys=("independent_prediction", "prediction"),
    )
    if parsed is None:
        return {
            "item_id": item["item_id"],
            "group_id": item["group_id"],
            "rationale": "",
            "prediction": "invalid",
            "confidence": 0.5,
            "stage_a_valid": stage_a_row["valid"],
            "fallback_required": True,
            "attempts": attempts,
            "errors": errors,
            "raw": raw[:2000],
        }
    return {
        "item_id": item["item_id"],
        "group_id": item["group_id"],
        "rationale": str(parsed["rationale"]),
        "prediction": normalize_prediction(parsed["prediction"]),
        "independent_prediction": normalize_prediction(parsed["independent_prediction"]),
        "confidence": normalize_confidence(parsed["confidence"]),
        "used_fallback": bool(parsed["used_fallback"]),
        "fallback_required": (
            not stage_a_row["stage_a"].get("sufficient_information", False)
            or normalize_confidence(stage_a_row["stage_a"].get("confidence", 0.0)) < threshold
        ),
        "stage_a_valid": stage_a_row["valid"],
        "attempts": attempts,
        "errors": errors,
    }


def parallel_collect(
    inputs: list[Any],
    worker: Callable[[Any], dict],
    workers: int,
    checkpoint: Callable[[list[dict]], None],
) -> list[dict]:
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, value) for value in inputs]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 25 == 0:
                checkpoint(rows)
    checkpoint(rows)
    return rows


def main() -> None:
    load_env_file(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="data/processed/cross_task_v3_items.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--base-url-env")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--thinking", choices=["enabled", "disabled"])
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--json-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.limit_groups is not None and args.limit_groups < 1) or args.workers < 1 or args.max_tokens < 1:
        parser.error("--limit-groups, --workers, and --max-tokens must be positive")
    if not 0 <= args.threshold <= 1 or args.timeout <= 0 or args.temperature < 0:
        parser.error("--threshold must be in [0, 1], --timeout positive, and --temperature nonnegative")

    base_url = os.environ.get(args.base_url_env) if args.base_url_env else args.base_url
    if not base_url:
        raise SystemExit("Missing base URL")
    items_path, output_path = resolve_path(args.items, ROOT), resolve_path(args.output, ROOT)
    items = list(iter_jsonl(items_path))
    if not items:
        raise SystemExit("Items file is empty")
    require_unique(items, "item_id", "items")
    total_groups = len({item["group_id"] for item in items})
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    stage_a_path = output_path.with_name(output_path.stem + ".stage_a.jsonl")
    identity = {
        "model": args.model,
        "items_sha256": sha256_file(items_path),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "reasoning_effort": args.reasoning_effort,
        "thinking": args.thinking,
        "threshold": args.threshold,
        "json_mode": args.json_mode,
        "base_url_host": urlparse(base_url).netloc,
        "base_url_path": urlparse(base_url).path.rstrip("/"),
        "code_sha256": sha256_files(
            [
                Path(__file__),
                ROOT / "scripts/run_openai_compatible.py",
                ROOT / "src/bav/io.py",
                ROOT / "src/bav/metrics.py",
                ROOT / "src/bav/prompts.py",
            ]
        ),
    }
    if args.resume:
        if not manifest_path.exists():
            raise SystemExit("Refusing resume without an existing manifest")
        previous = json.loads(manifest_path.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise SystemExit("Refusing resume: run identity differs from the existing run")
    elif output_path.exists() or manifest_path.exists() or stage_a_path.exists():
        raise SystemExit("Refusing to overwrite an existing run; use --resume or a new --output")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")
    if not args.resume:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps({**identity, "status": "running"}, indent=2)
            + "\n"
        )
    api = {
        "base_url": base_url,
        "api_key": api_key,
        "model": args.model,
        "temperature": args.temperature,
        "timeout": args.timeout,
        "json_mode": args.json_mode,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "thinking": args.thinking,
    }
    group_order = list(dict.fromkeys(item["group_id"] for item in items))
    if args.limit_groups is not None:
        selected = set(group_order[: args.limit_groups])
        items = [item for item in items if item["group_id"] in selected]
        group_order = group_order[: args.limit_groups]
    representatives = {item["group_id"]: item for item in items}

    stage_a_rows = list(iter_jsonl(stage_a_path)) if args.resume and stage_a_path.exists() else []
    require_unique(stage_a_rows, "group_id", "existing Stage-A outputs")
    unknown_groups = {row["group_id"] for row in stage_a_rows} - set(group_order)
    if unknown_groups:
        raise SystemExit(f"Refusing resume with unknown Stage-A group_id: {sorted(unknown_groups)[0]}")
    stage_a_by_group = {row["group_id"]: row for row in stage_a_rows}
    pending_groups = [group_id for group_id in group_order if group_id not in stage_a_by_group]

    def save_stage_a(new_rows: list[dict]) -> None:
        combined = [*stage_a_rows, *new_rows]
        combined.sort(key=lambda row: group_order.index(row["group_id"]))
        write_jsonl(stage_a_path, combined)

    new_stage_a = parallel_collect(
        pending_groups,
        lambda group_id: run_stage_a(representatives[group_id], api),
        args.workers,
        save_stage_a,
    ) if pending_groups else []
    stage_a_rows.extend(new_stage_a)
    stage_a_by_group = {row["group_id"]: row for row in stage_a_rows}

    prior = list(iter_jsonl(output_path)) if args.resume and output_path.exists() else []
    require_unique(prior, "item_id", "existing Stage-B outputs")
    unknown = {row["item_id"] for row in prior} - {item["item_id"] for item in items}
    if unknown:
        raise SystemExit(f"Refusing resume with unknown output item_id: {sorted(unknown)[0]}")
    done = {row["item_id"] for row in prior}
    pending_items = [item for item in items if item["item_id"] not in done]
    item_order = {item["item_id"]: index for index, item in enumerate(items)}

    def save_stage_b(new_rows: list[dict]) -> None:
        combined = [*prior, *new_rows]
        combined.sort(key=lambda row: item_order[row["item_id"]])
        write_jsonl(output_path, combined)

    new_outputs = parallel_collect(
        pending_items,
        lambda item: run_stage_b(item, stage_a_by_group[item["group_id"]], args.threshold, api),
        args.workers,
        save_stage_b,
    ) if pending_items else []
    outputs = [*prior, *new_outputs]
    outputs.sort(key=lambda row: item_order[row["item_id"]])
    write_jsonl(output_path, outputs)

    manifest = {
        **identity,
        "api_key_env": args.api_key_env,
        "max_attempts_per_stage": 3,
        "n_groups": len(group_order),
        "n_total_groups": total_groups,
        "n_outputs": len(outputs),
        "n_stage_a_records": len(stage_a_rows),
        "stage_a_api_attempts": sum(len(row.get("attempts", [])) for row in stage_a_rows),
        "stage_b_api_attempts": sum(len(row.get("attempts", [])) for row in outputs),
        "stage_a_usage": usage_totals(stage_a_rows),
        "stage_b_usage": usage_totals(outputs),
        "output_sha256": sha256_file(output_path),
        "stage_a_sha256": sha256_file(stage_a_path),
        "stage_a_reused_within_group": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if len(group_order) == total_groups else "partial",
        "workers": args.workers,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(stage_a_rows)} Stage-A and {len(outputs)} Stage-B outputs -> {output_path}")


if __name__ == "__main__":
    main()
