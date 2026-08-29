#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from bav.io import sha256_file, sha256_files


PRICE_CHECKED_ON = "2026-08-23"
PRICE_SOURCES = {
    "deepseek-v4-flash": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing",
    "doubao-seed-2-1-pro-260628": "https://www.volcengine.com/product/ark",
}


def api_cost_yuan(model: str, usage: dict[str, int]) -> float:
    if model == "deepseek-v4-flash":
        return (
            usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0)) * 1
            + usage.get("prompt_cache_hit_tokens", 0) * 0.02
            + usage.get("completion_tokens", 0) * 2
        ) / 1_000_000
    if model == "doubao-seed-2-1-pro-260628":
        return (usage.get("prompt_tokens", 0) * 6 + usage.get("completion_tokens", 0) * 30) / 1_000_000
    raise ValueError(f"No frozen price for {model}")


def combined_usage(manifest: dict) -> dict[str, int]:
    blocks = [manifest["usage"]] if "usage" in manifest else [
        value for key, value in manifest.items() if key.endswith("_usage") and isinstance(value, dict)
    ]
    totals: dict[str, int] = {}
    for block in blocks:
        for key, value in block.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs/formal")
    parser.add_argument("--output")
    args = parser.parse_args()
    runs = Path(args.runs)
    runs = runs if runs.is_absolute() else ROOT / runs
    rows = []
    for path in sorted(runs.glob("**/*.manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        model = manifest.get("model")
        if model not in {"deepseek-v4-flash", "doubao-seed-2-1-pro-260628"}:
            continue
        usage = combined_usage(manifest)
        rows.append({
            "protocol": path.relative_to(runs).parts[0],
            "model": model,
            "manifest_sha256": sha256_file(path),
            "status": manifest.get("status"),
            "usage": usage,
            "cost_yuan": api_cost_yuan(model, usage),
        })
    report = {
        "code_sha256": sha256_files([Path(__file__)]),
        "price_checked_on": PRICE_CHECKED_ON,
        "price_sources": PRICE_SOURCES,
        "runs": rows,
        "total_cost_yuan": sum(row["cost_yuan"] for row in rows),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output = output if output.is_absolute() else ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
