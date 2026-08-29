#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bav.io import iter_jsonl, resolve_path, sha256_file, write_jsonl


ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="data/processed/cross_task_v3_items.jsonl")
    parser.add_argument("--output", default="runs/oracle/outputs.jsonl")
    args = parser.parse_args()

    items_path, output_path = resolve_path(args.items, ROOT), resolve_path(args.output, ROOT)
    rows = (
        {"item_id": item["item_id"], "prediction": item["gold"], "confidence": 1.0}
        for item in iter_jsonl(items_path)
    )
    count = write_jsonl(output_path, rows)
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(
            {"items_sha256": sha256_file(items_path), "n_outputs": count, "output_sha256": sha256_file(output_path)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {count} oracle outputs -> {args.output}")


if __name__ == "__main__":
    main()
