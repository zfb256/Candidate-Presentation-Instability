#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from http.client import IncompleteRead, RemoteDisconnected
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
from urllib import error, request
from urllib.parse import urlparse

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bav.io import extract_json_object, iter_jsonl, require_unique, resolve_path, sha256_file, sha256_files, write_jsonl
from bav.metrics import normalize_confidence, normalize_prediction


ROOT = Path(__file__).resolve().parents[1]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value

def ordered_rows(rows: list[dict[str, Any]], order: dict[str, int]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (order.get(str(row.get("item_id")), len(order)), str(row.get("item_id"))))


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout: float,
    json_mode: bool,
    max_tokens: int,
    reasoning_effort: str | None,
    thinking: str | None,
) -> tuple[str, dict[str, Any]]:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    if thinking:
        body["thinking"] = {"type": thinking}
    data = json.dumps(body).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected API response: {payload!r}")
    if "error" in payload:
        raise ValueError(f"API error: {payload['error']}")
    choice = payload["choices"][0]
    message = choice["message"]
    metadata = {
        "model": payload.get("model"),
        "system_fingerprint": payload.get("system_fingerprint"),
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage") or {},
        "reasoning_chars": len(message.get("reasoning_content") or ""),
        "content_chars": len(message.get("content") or ""),
    }
    return message["content"], metadata


def request_json(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout: float,
    json_mode: bool,
    max_tokens: int,
    reasoning_effort: str | None,
    thinking: str | None,
    required_keys: tuple[str, ...] = ("rationale", "prediction", "confidence"),
    boolean_keys: tuple[str, ...] = (),
    prediction_keys: tuple[str, ...] = ("prediction",),
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[str], str]:
    attempts: list[dict[str, Any]] = []
    errors: list[str] = []
    last_content = ""
    for _ in range(3):
        attempts_before = len(attempts)
        try:
            content, metadata = chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=prompt,
                temperature=temperature,
                timeout=timeout,
                json_mode=json_mode,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                thinking=thinking,
            )
            attempts.append(metadata)
            last_content = content
            if metadata.get("finish_reason") == "length":
                errors.append("finish_reason=length")
                break
            parsed = extract_json_object(content)
            missing = [key for key in required_keys if key not in parsed]
            if missing:
                errors.append("missing keys: " + ", ".join(missing))
                continue
            if "rationale" in required_keys and (
                not isinstance(parsed["rationale"], str) or not parsed["rationale"].strip()
            ):
                errors.append("rationale must be a non-empty string")
                continue
            invalid_booleans = [key for key in boolean_keys if not isinstance(parsed.get(key), bool)]
            if invalid_booleans:
                errors.append("non-boolean keys: " + ", ".join(invalid_booleans))
                continue
            if "confidence" in required_keys:
                if isinstance(parsed["confidence"], bool):
                    errors.append("confidence is not numeric")
                    continue
                try:
                    confidence = float(parsed["confidence"])
                except (TypeError, ValueError):
                    errors.append("confidence is not numeric")
                    continue
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    errors.append("confidence is outside [0, 1]")
                    continue
            invalid_predictions = [
                key for key in prediction_keys
                if key in required_keys and str(parsed.get(key)).strip().lower() not in {"correct", "incorrect"}
            ]
            if invalid_predictions:
                errors.append("non-verdict keys: " + ", ".join(invalid_predictions))
                continue
            if "prediction" not in parsed or normalize_prediction(parsed["prediction"]) != "invalid":
                return parsed, attempts, errors, last_content
        except (
            ValueError,
            KeyError,
            IndexError,
            TypeError,
            ConnectionError,
            error.URLError,
            TimeoutError,
            RemoteDisconnected,
            IncompleteRead,
        ) as exc:
            if len(attempts) == attempts_before:
                attempts.append({"error_type": type(exc).__name__})
            errors.append(str(exc))
    return None, attempts, errors, last_content


def usage_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for row in rows:
        for attempt in row.get("attempts", []):
            for key, value in attempt.get("usage", {}).items():
                if isinstance(value, int):
                    totals[key] = totals.get(key, 0) + value
    return dict(sorted(totals.items()))


def run_prompt(
    *,
    prompt_row: dict[str, Any],
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float,
    json_mode: bool,
    max_tokens: int,
    reasoning_effort: str | None,
    thinking: str | None,
) -> dict[str, Any]:
    item_id = prompt_row["item_id"]
    parsed, attempts, errors, last_content = request_json(
        base_url=base_url,
        api_key=api_key,
        model=model,
        prompt=prompt_row["prompt"],
        temperature=temperature,
        timeout=timeout,
        json_mode=json_mode,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        thinking=thinking,
    )
    if parsed is not None and normalize_prediction(parsed.get("prediction")) != "invalid":
        row = {
            "item_id": item_id,
            "prediction": normalize_prediction(parsed.get("prediction")),
            "confidence": normalize_confidence(parsed.get("confidence", 0.5)),
            "rationale": str(parsed.get("rationale", "")),
            "model": model,
            "attempts": attempts,
        }
        if errors:
            row["errors"] = errors
        return row
    return {
        "item_id": item_id,
        "prediction": "invalid",
        "confidence": 0.5,
        "rationale": "",
        "model": model,
        "attempts": attempts,
        "errors": errors,
        "raw": last_content[:2000],
    }


def main() -> None:
    load_env_file(ROOT / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True, help="Prompt JSONL, e.g. runs/direct/prompts.jsonl")
    parser.add_argument("--output", required=True, help="Output JSONL, e.g. runs/direct/qwen_outputs.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--base-url-env", default=None, help="Read base URL from an environment variable")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--limit-new", type=int, default=None, help="Maximum number of new prompt rows to run")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--thinking", choices=["enabled", "disabled"])
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent API requests")
    parser.add_argument("--json-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.limit_new is not None and args.limit_new < 1) or args.workers < 1 or args.max_tokens < 1:
        parser.error("--limit-new, --workers, and --max-tokens must be positive")
    if args.timeout <= 0 or args.temperature < 0 or args.sleep < 0:
        parser.error("--timeout must be positive; --temperature and --sleep must be nonnegative")

    base_url = os.environ.get(args.base_url_env) if args.base_url_env else args.base_url
    if not base_url:
        raise SystemExit(f"Missing base URL env var: {args.base_url_env}")

    prompts_path = resolve_path(args.prompts, ROOT)
    output_path = resolve_path(args.output, ROOT)
    prompt_rows = list(iter_jsonl(prompts_path))
    if not prompt_rows:
        raise SystemExit("Prompt file is empty")
    require_unique(prompt_rows, "item_id", "prompts")
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    identity = {
        "model": args.model,
        "prompt_sha256": sha256_file(prompts_path),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "reasoning_effort": args.reasoning_effort,
        "thinking": args.thinking,
        "json_mode": args.json_mode,
        "base_url_host": urlparse(base_url).netloc,
        "base_url_path": urlparse(base_url).path.rstrip("/"),
        "code_sha256": sha256_files([Path(__file__), ROOT / "src/bav/io.py", ROOT / "src/bav/metrics.py"]),
    }
    if args.resume:
        if not manifest_path.exists():
            raise SystemExit("Refusing resume without an existing manifest")
        previous = json.loads(manifest_path.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise SystemExit("Refusing resume: run identity differs from the existing run")
    elif output_path.exists() or manifest_path.exists():
        raise SystemExit("Refusing to overwrite an existing run; use --resume or a new --output")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")
    if not args.resume:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({**identity, "status": "running"}, indent=2) + "\n")
    rows = list(iter_jsonl(output_path)) if args.resume and output_path.exists() else []
    require_unique(rows, "item_id", "existing outputs")
    unknown = {row["item_id"] for row in rows} - {row["item_id"] for row in prompt_rows}
    if unknown:
        raise SystemExit(f"Refusing resume with unknown output item_id: {sorted(unknown)[0]}")
    rows_by_id = {row["item_id"]: row for row in rows}
    done = set(rows_by_id)
    limit_new = args.limit_new
    prompt_order: dict[str, int] = {}
    pending = []
    total_prompts = 0
    for total_prompts, prompt_row in enumerate(prompt_rows, start=1):
        prompt_order[prompt_row["item_id"]] = total_prompts - 1
        if limit_new is not None and len(pending) >= limit_new:
            continue
        if prompt_row["item_id"] not in done:
            pending.append(prompt_row)

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for prompt_row in pending:
                futures.append(
                    executor.submit(
                        run_prompt,
                        prompt_row=prompt_row,
                        base_url=base_url,
                        api_key=api_key,
                        model=args.model,
                        temperature=args.temperature,
                        timeout=args.timeout,
                        json_mode=args.json_mode,
                        max_tokens=args.max_tokens,
                        reasoning_effort=args.reasoning_effort,
                        thinking=args.thinking,
                    )
                )
                if args.sleep:
                    time.sleep(args.sleep)
            for processed, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows_by_id[row["item_id"]] = row
                if processed % 25 == 0:
                    write_jsonl(output_path, ordered_rows(list(rows_by_id.values()), prompt_order))
                    print(f"Wrote checkpoint: {len(rows_by_id)} rows -> {output_path}", flush=True)
    else:
        processed = 0
        for prompt_row in pending:
            row = run_prompt(
                prompt_row=prompt_row,
                base_url=base_url,
                api_key=api_key,
                model=args.model,
                temperature=args.temperature,
                timeout=args.timeout,
                json_mode=args.json_mode,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
                thinking=args.thinking,
            )
            rows_by_id[row["item_id"]] = row
            processed += 1
            if processed % 25 == 0:
                write_jsonl(output_path, ordered_rows(list(rows_by_id.values()), prompt_order))
                print(f"Wrote checkpoint: {len(rows_by_id)} rows -> {output_path}", flush=True)
            if args.sleep:
                time.sleep(args.sleep)

    final_rows = ordered_rows(list(rows_by_id.values()), prompt_order)
    count = write_jsonl(output_path, final_rows)
    write_manifest(
        output_path,
        {
            **identity,
            "api_key_env": args.api_key_env,
            "base_url_env": args.base_url_env,
            "limit_new": limit_new,
            "n_outputs": count,
            "n_pending_this_run": len(pending),
            "n_prompts": total_prompts,
            "api_attempts": sum(len(row.get("attempts", [])) for row in final_rows),
            "usage": usage_totals(final_rows),
            "output": str(output_path.relative_to(ROOT) if output_path.is_relative_to(ROOT) else output_path),
            "output_sha256": sha256_file(output_path),
            "prompts": str(prompts_path.relative_to(ROOT) if prompts_path.is_relative_to(ROOT) else prompts_path),
            "response_metadata": "stored per attempt in each output row",
            "resume": args.resume,
            "sleep": args.sleep,
            "timeout": args.timeout,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete" if count == total_prompts else "partial",
            "workers": args.workers,
        },
    )
    print(f"Wrote {count} model outputs -> {output_path}")


if __name__ == "__main__":
    main()
