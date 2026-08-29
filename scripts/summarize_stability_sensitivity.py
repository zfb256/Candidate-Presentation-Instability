#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from bav.io import read_jsonl
from bav.metrics import normalize_prediction


def summarize(items: list[dict], rows: list[dict]) -> dict:
    outputs = {row["item_id"]: row for row in rows}
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in items:
        cells[(item["group_id"], item["polarity"])].append(item)

    anchored_flips = harmful = beneficial = anchored_invalid = 0
    pairwise_disagreements = pairwise_valid = pairwise_invalid = 0
    for group_items in cells.values():
        original = next(item for item in group_items if item["family"] == "original")
        original_prediction = normalize_prediction(outputs.get(original["item_id"], {}).get("prediction"))
        for variant in (item for item in group_items if item["family"] != "original"):
            variant_prediction = normalize_prediction(outputs.get(variant["item_id"], {}).get("prediction"))
            if "invalid" in {original_prediction, variant_prediction}:
                anchored_invalid += 1
                continue
            flipped = original_prediction != variant_prediction
            anchored_flips += flipped
            harmful += flipped and original_prediction == original["gold"]
            beneficial += flipped and variant_prediction == original["gold"]

        for left, right in itertools.combinations(group_items, 2):
            predictions = (
                normalize_prediction(outputs.get(left["item_id"], {}).get("prediction")),
                normalize_prediction(outputs.get(right["item_id"], {}).get("prediction")),
            )
            if "invalid" in predictions:
                pairwise_invalid += 1
                continue
            pairwise_valid += 1
            pairwise_disagreements += predictions[0] != predictions[1]

    anchored_total = len(cells) * 6
    pairwise_total = len(cells) * 21
    assert anchored_total == 9_000 and pairwise_total == 31_500
    assert anchored_flips + anchored_invalid <= anchored_total
    assert pairwise_total == pairwise_valid + pairwise_invalid
    return {
        "anchored": {
            "beneficial_flips": beneficial,
            "comparisons": anchored_total - anchored_invalid,
            "flips": anchored_flips,
            "harmful_flips": harmful,
            "invalid_pairs": anchored_invalid,
            "rate": anchored_flips / (anchored_total - anchored_invalid),
            "rate_bounds_invalid_as_no_flip_or_flip": [
                anchored_flips / anchored_total,
                (anchored_flips + anchored_invalid) / anchored_total,
            ],
            "total_pairs": anchored_total,
        },
        "all_pairwise": {
            "comparisons": pairwise_valid,
            "disagreements": pairwise_disagreements,
            "invalid_pairs": pairwise_invalid,
            "rate": pairwise_disagreements / pairwise_valid,
            "total_pairs": pairwise_total,
        },
    }


def main() -> None:
    items = read_jsonl(ROOT / "data/processed/cross_task_v3_items.jsonl")
    reports = {}
    for path in sorted((ROOT / "runs/formal").glob("*/*.jsonl")):
        if path.name.endswith(".stage_a.jsonl"):
            continue
        key = f"{path.parent.name}/{path.stem}"
        reports[key] = summarize(items, read_jsonl(path))
    p5 = reports["p5/deepseek-v4-flash"]
    assert p5["anchored"]["harmful_flips"] == 11 and p5["anchored"]["beneficial_flips"] == 10
    assert round(p5["all_pairwise"]["rate"], 6) == 0.003333
    output = ROOT / "reports/formal/stability_sensitivity.json"
    output.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(reports)} cells -> {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
