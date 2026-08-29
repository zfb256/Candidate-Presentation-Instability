#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bav.io import iter_jsonl, sha256_file
from bav.metrics import invariant_flip_stats, normalize_prediction


ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    "qwen2.5-1.5b",
    "qwen2.5-3b",
    "qwen2.5-7b",
    "qwen2.5-14b",
    "deepseek-v4-flash",
    "doubao-seed-2-1-pro-260628",
)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def repeat_counts(items: list[dict], first: dict[str, dict], repeat: dict[str, dict]) -> tuple[int, int]:
    valid = flips = 0
    for item in items:
        left = normalize_prediction(first[item["item_id"]].get("prediction"))
        right = normalize_prediction(repeat[item["item_id"]].get("prediction"))
        if "invalid" in {left, right}:
            continue
        valid += 1
        flips += left != right
    return flips, valid


def presentation_counts(items: list[dict], first: dict[str, dict]) -> tuple[int, int]:
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in items:
        cells[(item["group_id"], item["polarity"])].append(item)
    valid = flips = 0
    for rows in cells.values():
        originals = [item for item in rows if item["family"] == "original"]
        if not originals:
            continue
        original = normalize_prediction(first[originals[0]["item_id"]].get("prediction"))
        for item in rows:
            if item["family"] == "original":
                continue
            variant = normalize_prediction(first[item["item_id"]].get("prediction"))
            if "invalid" in {original, variant}:
                continue
            valid += 1
            flips += original != variant
    return flips, valid


def bootstrap(items: list[dict], first: dict[str, dict], repeat: dict[str, dict], rounds: int, seed: int) -> tuple[list[float], list[float], list[float]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        grouped[item["group_id"]].append(item)
    groups = sorted(grouped.values(), key=lambda rows: rows[0]["group_id"])
    counts = [(presentation_counts(rows, first), repeat_counts(rows, first, repeat)) for rows in groups]
    rng = random.Random(seed)
    presentation_rates, repeat_rates, gaps = [], [], []
    for _ in range(rounds):
        sampled = [rng.choice(counts) for _ in counts]
        presentation_flips = sum(value[0][0] for value in sampled)
        presentation_valid = sum(value[0][1] for value in sampled)
        repeat_flips = sum(value[1][0] for value in sampled)
        repeat_valid = sum(value[1][1] for value in sampled)
        repeat_rate = repeat_flips / repeat_valid if repeat_valid else 0.0
        presentation_rate = presentation_flips / presentation_valid if presentation_valid else 0.0
        presentation_rates.append(presentation_rate)
        repeat_rates.append(repeat_rate)
        gaps.append(presentation_rate - repeat_rate)
    return presentation_rates, repeat_rates, gaps


def summarize_slice(items: list[dict], first: dict[str, dict], repeat: dict[str, dict], rounds: int, seed: int) -> dict:
    repeat_flips, repeat_valid = repeat_counts(items, first, repeat)
    presentation_flips, presentation_valid = presentation_counts(items, first)
    presentation_rates, repeat_rates, gaps = bootstrap(items, first, repeat, rounds, seed)
    repeat_rate = repeat_flips / repeat_valid if repeat_valid else 0.0
    presentation_rate = presentation_flips / presentation_valid if presentation_valid else 0.0
    return {
        "repeat": {
            "flips": repeat_flips,
            "common_valid": repeat_valid,
            "rate": repeat_rate,
            "ci95": [percentile(repeat_rates, 0.025), percentile(repeat_rates, 0.975)],
        },
        "presentation": {
            "flips": presentation_flips,
            "common_valid": presentation_valid,
            "rate": presentation_rate,
            "ci95": [percentile(presentation_rates, 0.025), percentile(presentation_rates, 0.975)],
        },
        "presentation_minus_repeat": presentation_rate - repeat_rate,
        "presentation_minus_repeat_ci95": [percentile(gaps, 0.025), percentile(gaps, 0.975)],
    }


def invalid_rate(items: list[dict], outputs: dict[str, dict]) -> float:
    return sum(normalize_prediction(outputs[item["item_id"]].get("prediction")) == "invalid" for item in items) / len(items)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", default="reports/formal/exact_repeat.json")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")

    items_path = ROOT / "data/processed/cross_task_v3_probe200.jsonl"
    items = list(iter_jsonl(items_path))
    assert len(items) == 2800 and len({item["group_id"] for item in items}) == 200
    report = {"n_groups": 200, "n_items": 2800, "bootstrap_rounds": args.rounds, "models": {}}
    for model in MODELS:
        first_path = ROOT / f"runs/formal/p1/{model}.jsonl"
        repeat_path = ROOT / f"runs/formal/exact_repeat/{model}.jsonl"
        first = {row["item_id"]: row for row in iter_jsonl(first_path)}
        repeat = {row["item_id"]: row for row in iter_jsonl(repeat_path)}
        assert all(item["item_id"] in first and item["item_id"] in repeat for item in items)
        overall = summarize_slice(items, first, repeat, args.rounds, args.seed)
        presentation = invariant_flip_stats(items, first)
        assert presentation["flipped"] == overall["presentation"]["flips"]
        assert presentation["comparisons"] == overall["presentation"]["common_valid"]
        report["models"][model] = {
            "first_sha256": sha256_file(first_path),
            "repeat_sha256": sha256_file(repeat_path),
            "first_invalid_rate": invalid_rate(items, first),
            "repeat_invalid_rate": invalid_rate(items, repeat),
            "repeat": overall["repeat"],
            "presentation_ifr": presentation,
            "presentation_ifr_ci95": overall["presentation"]["ci95"],
            "presentation_minus_repeat": overall["presentation_minus_repeat"],
            "presentation_minus_repeat_ci95": overall["presentation_minus_repeat_ci95"],
            "by_dataset": {
                dataset: summarize_slice(
                    [item for item in items if item["dataset"] == dataset], first, repeat, args.rounds, args.seed
                )
                for dataset in sorted({item["dataset"] for item in items})
            },
            "by_polarity": {
                polarity: summarize_slice(
                    [item for item in items if item["polarity"] == polarity], first, repeat, args.rounds, args.seed
                )
                for polarity in ("correct", "incorrect")
            },
        }

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({model: {
        "repeat": values["repeat"],
        "presentation_minus_repeat": values["presentation_minus_repeat"],
        "presentation_minus_repeat_ci95": values["presentation_minus_repeat_ci95"],
    } for model, values in report["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
