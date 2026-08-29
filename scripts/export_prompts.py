#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bav.io import iter_jsonl, resolve_path, sha256_file, write_jsonl
from bav.prompts import PROTOCOLS, length_matched_direct_prompt


ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="data/processed/cross_task_v3_items.jsonl")
    parser.add_argument("--protocol", choices=[*PROTOCOLS, "p6_length_matched_direct"], required=True)
    parser.add_argument("--target-rationale-words", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.protocol == "p6_length_matched_direct" and not args.target_rationale_words:
        parser.error("P6 requires --target-rationale-words from model-specific P4 outputs")

    items_path = resolve_path(args.items, ROOT)
    output = resolve_path(args.output or f"runs/{args.protocol}/prompts.jsonl", ROOT)
    prompt_fn = (
        (lambda item: length_matched_direct_prompt(item, args.target_rationale_words))
        if args.protocol == "p6_length_matched_direct"
        else PROTOCOLS[args.protocol]
    )
    rows = (
        {
            "item_id": item["item_id"],
            "group_id": item["group_id"],
            "dataset": item["dataset"],
            "polarity": item["polarity"],
            "family": item["family"],
            "protocol": args.protocol,
            "prompt": prompt_fn(item),
        }
        for item in iter_jsonl(items_path)
    )
    count = write_jsonl(output, rows)
    manifest = {
        "items": str(items_path.relative_to(ROOT)) if items_path.is_relative_to(ROOT) else str(items_path),
        "items_sha256": sha256_file(items_path),
        "target_rationale_words": args.target_rationale_words,
        "n_prompts": count,
        "output_sha256": sha256_file(output),
        "protocol": args.protocol,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {count} {args.protocol} prompts -> {output}")


if __name__ == "__main__":
    main()
