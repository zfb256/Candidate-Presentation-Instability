#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bav.io import read_jsonl, resolve_path, sha256_file, sha256_files
from bav.metrics import (
    accuracy,
    bootstrap_metric_cis,
    bootstrap_invariant_flip_breakdown_ci,
    brier_score,
    coverage_stats,
    f1_for_incorrect,
    invariant_flip_breakdown,
    invariant_flip_stats,
    paired_coverage_stats,
    paired_bootstrap_delta_cis,
    paired_output_intersection,
    paired_valid_invariant_flip_delta,
    responsiveness_rate,
)


ROOT = Path(__file__).resolve().parents[1]

def output_map(rows: list[dict], label: str) -> dict[str, dict]:
    outputs: dict[str, dict] = {}
    duplicates: list[str] = []
    for row in rows:
        item_id = row["item_id"]
        if item_id in outputs:
            duplicates.append(item_id)
        outputs[item_id] = row
    if duplicates:
        preview = ", ".join(sorted(set(duplicates))[:5])
        raise SystemExit(f"{label} contains duplicate item_id values: {preview}")
    return outputs


def assert_complete_outputs(items: list[dict], outputs: dict[str, dict], label: str) -> None:
    missing = [item["item_id"] for item in items if item.get("gold") in {"correct", "incorrect"} and item["item_id"] not in outputs]
    if missing:
        preview = ", ".join(missing[:5])
        raise SystemExit(
            f"{label} is missing {len(missing)} binary-evaluable outputs, e.g. {preview}. "
            "Use --allow-missing to evaluate a partial run."
        )


def assert_known_outputs(items: list[dict], outputs: dict[str, dict], label: str) -> None:
    item_ids = {item["item_id"] for item in items}
    unknown = sorted(set(outputs) - item_ids)
    if unknown:
        raise SystemExit(f"{label} contains {len(unknown)} unknown item_id values, e.g. {', '.join(unknown[:5])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="data/processed/cross_task_v3_items.jsonl")
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--baseline-outputs", default=None)
    parser.add_argument("--bootstrap-rounds", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_rounds < 1:
        parser.error("--bootstrap-rounds must be positive")

    items_path = resolve_path(args.items, ROOT)
    outputs_path = resolve_path(args.outputs, ROOT)
    items = read_jsonl(items_path)
    output_rows = read_jsonl(outputs_path)
    outputs = output_map(output_rows, args.outputs)
    assert_known_outputs(items, outputs, args.outputs)
    if not args.allow_missing:
        assert_complete_outputs(items, outputs, args.outputs)
    baseline_outputs = None
    baseline_path = None
    if args.baseline_outputs:
        baseline_path = resolve_path(args.baseline_outputs, ROOT)
        baseline_rows = read_jsonl(baseline_path)
        baseline_outputs = output_map(baseline_rows, args.baseline_outputs)
        assert_known_outputs(items, baseline_outputs, args.baseline_outputs)
        if not args.allow_missing:
            assert_complete_outputs(items, baseline_outputs, args.baseline_outputs)

    invariant_ci = bootstrap_invariant_flip_breakdown_ci(
        items, outputs, rounds=args.bootstrap_rounds, seed=args.bootstrap_seed
    )
    metric_cis = bootstrap_metric_cis(items, outputs, rounds=args.bootstrap_rounds, seed=args.bootstrap_seed)

    report = {
        "baseline_outputs_sha256": sha256_file(baseline_path) if baseline_path else None,
        "bootstrap_rounds": args.bootstrap_rounds,
        "bootstrap_seed": args.bootstrap_seed,
        "code_sha256": sha256_files([Path(__file__), ROOT / "src/bav/io.py", ROOT / "src/bav/metrics.py"]),
        "items_sha256": sha256_file(items_path),
        "outputs_sha256": sha256_file(outputs_path),
        "n_items": len(items),
        "n_outputs": len(outputs),
        "coverage": coverage_stats(items, outputs),
        "accuracy": accuracy(items, outputs),
        "incorrect_f1": f1_for_incorrect(items, outputs),
        "brier_score": brier_score(items, outputs),
        "invariant_flip": invariant_flip_breakdown(items, outputs),
        "invariant_flip_ci95": invariant_ci,
        "responsiveness": responsiveness_rate(items, outputs),
        "ci95": {**metric_cis, "invariant_flip_rate": invariant_ci["overall"]},
    }
    if baseline_outputs is not None:
        paired_baseline_outputs, paired_method_outputs = paired_output_intersection(baseline_outputs, outputs)
        paired_cis = paired_bootstrap_delta_cis(
            items, baseline_outputs, outputs, rounds=args.bootstrap_rounds, seed=args.bootstrap_seed
        )
        report["baseline_n_outputs"] = len(baseline_outputs)
        report["paired_coverage"] = paired_coverage_stats(items, baseline_outputs, outputs)
        report["delta_vs_baseline"] = {
            "accuracy": accuracy(items, paired_method_outputs) - accuracy(items, paired_baseline_outputs),
            "incorrect_f1": f1_for_incorrect(items, paired_method_outputs)["f1"]
            - f1_for_incorrect(items, paired_baseline_outputs)["f1"],
            "invariant_flip_rate": invariant_flip_stats(items, paired_method_outputs)["rate"]
            - invariant_flip_stats(items, paired_baseline_outputs)["rate"],
            "paired_valid_invariant_flip_rate": paired_valid_invariant_flip_delta(
                items, baseline_outputs, outputs
            ),
            "ci95": paired_cis,
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
