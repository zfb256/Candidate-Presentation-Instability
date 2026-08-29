#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from statistics import mean, median
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bav.io import read_jsonl, resolve_path, sha256_file


ROOT = Path(__file__).resolve().parents[1]

def completion_tokens(row: dict) -> int | None:
    value = row.get("completion_tokens")
    if isinstance(value, int):
        return value
    for attempt in reversed(row.get("attempts", [])):
        value = attempt.get("usage", {}).get("completion_tokens")
        if isinstance(value, int):
            return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", required=True, help="P4 probe outputs")
    parser.add_argument("--output", help="Optional calibration JSON")
    args = parser.parse_args()

    outputs_path = resolve_path(args.outputs, ROOT)
    rows = read_jsonl(outputs_path)
    valid = [row for row in rows if row.get("prediction") in {"correct", "incorrect"}]
    word_counts = [len(re.findall(r"\b[\w'-]+\b", str(row.get("rationale", "")))) for row in valid]
    if not word_counts:
        raise SystemExit("No valid rationales to calibrate")
    token_counts = [value for row in valid if (value := completion_tokens(row)) is not None]
    report = {
        "source": str(outputs_path.relative_to(ROOT)) if outputs_path.is_relative_to(ROOT) else str(outputs_path),
        "source_sha256": sha256_file(outputs_path),
        "model": valid[0].get("model"),
        "n_outputs": len(rows),
        "n_valid": len(valid),
        "invalid_rate": 1 - len(valid) / len(rows),
        "mean_rationale_words": mean(word_counts),
        "median_rationale_words": median(word_counts),
        "target_rationale_words_for_p6": max(1, round(mean(word_counts))),
        "mean_recorded_completion_tokens": mean(token_counts) if token_counts else None,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output_path = resolve_path(args.output, ROOT)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
