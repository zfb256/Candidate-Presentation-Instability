#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from bav.io import sha256_file, sha256_files


def evaluation_row(protocol: str, model: str, report: dict) -> dict:
    delta = report.get("delta_vs_baseline", {})
    paired = delta.get("paired_valid_invariant_flip_rate", {})
    return {
        "protocol": protocol,
        "model": model,
        "n_outputs": report["n_outputs"],
        "accuracy": report["accuracy"],
        "accuracy_ci95": report["ci95"]["accuracy"],
        "incorrect_f1": report["incorrect_f1"]["f1"],
        "brier_score": report["brier_score"],
        "invariant_flip_rate": report["invariant_flip"]["overall"]["rate"],
        "invariant_flip_ci95": report["ci95"]["invariant_flip_rate"],
        "responsiveness": report["responsiveness"]["rate"],
        "invalid_rate": report["coverage"]["invalid_rate"],
        "delta_accuracy": delta.get("accuracy"),
        "delta_accuracy_ci95": delta.get("ci95", {}).get("accuracy"),
        "delta_paired_valid_ifr": paired.get("delta"),
        "delta_paired_valid_ifr_ci95": delta.get("ci95", {}).get("paired_valid_invariant_flip_rate"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs/formal")
    parser.add_argument("--json-output", default="reports/formal/evaluations.json")
    parser.add_argument("--csv-output", default="reports/formal/evaluations.csv")
    args = parser.parse_args()
    runs = Path(args.runs)
    runs = runs if runs.is_absolute() else ROOT / runs
    rows = []
    for path in sorted(runs.glob("**/*.evaluation.json")):
        relative = path.relative_to(runs)
        row = evaluation_row(relative.parts[0], path.name.removesuffix(".evaluation.json"), json.loads(path.read_text()))
        row["evaluation_sha256"] = sha256_file(path)
        rows.append(row)

    json_output = Path(args.json_output)
    json_output = json_output if json_output.is_absolute() else ROOT / json_output
    csv_output = Path(args.csv_output)
    csv_output = csv_output if csv_output.is_absolute() else ROOT / csv_output
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps({"code_sha256": sha256_files([Path(__file__)]), "evaluations": rows}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(rows)} evaluations -> {json_output} and {csv_output}")


if __name__ == "__main__":
    main()
