#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any


REVISION_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REVISION_ROOT / "src"))

from bav.io import iter_jsonl, resolve_path, sha256_file, write_jsonl


FAMILIES = (
    "original",
    "f1_format",
    "f2_verbosity",
    "f3_paraphrase",
    "f4_confidence",
    "f5_authority",
    "f6_distractor",
)
POLARITIES = ("correct", "incorrect")
DATASETS = ("math", "boolq", "openbookqa", "arc_challenge", "commonsenseqa")
RAW_DEFAULTS = {
    "boolq": "data/raw/boolq/dev.jsonl",
    "openbookqa": "data/raw/openbookqa/test.jsonl",
    "arc_challenge": "data/raw/arc_challenge/test.jsonl",
    "commonsenseqa": "data/raw/commonsenseqa/dev_rand_split.jsonl",
}
EXPECTED_SHA256 = {
    "math": "dd85b4558749f22fe67e535c4c9d8745241def75b5edc05063fd8aa5effd6b23",
    "boolq": "e8fb84fbf510b022e963cddf3a3aded04151afa0ea0ef1cc1bf22f260ddd2344",
    "openbookqa": "1e448ea67d38e7c1c95bbc961b803170c6976e12cc34596c550bfbdd302a3226",
    "arc_challenge": "86ccaace4cd159b5c02b6b3339ebe8a7022a54f1b488e433c060b3bc38ac1f63",
    "commonsenseqa": "4e83a600ff0c8ab5f0bbeec89fe45fa35b6e4dd04cb36124c7acb141d233fc14",
}
SOURCE_ROWS = {"math": 5000, "boolq": 3270, "openbookqa": 500, "arc_challenge": 1172, "commonsenseqa": 1221}
ELIGIBLE_ROWS = {"math": 3751, "boolq": 3270, "openbookqa": 500, "arc_challenge": 1172, "commonsenseqa": 1221}


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

def boxed_answer(solution: str) -> str | None:
    marker = r"\boxed{"
    found: list[str] = []
    start = 0
    while (box_start := solution.find(marker, start)) >= 0:
        content_start = box_start + len(marker)
        depth = 1
        index = content_start
        while index < len(solution) and depth:
            depth += (solution[index] == "{") - (solution[index] == "}")
            index += 1
        if depth:
            break
        found.append(solution[content_start : index - 1].strip())
        start = index
    return found[-1] if found else None


def wrong_math_answer(answer: str) -> str | None:
    text = answer.strip()
    fraction = re.fullmatch(r"\\(?:d?frac)\{(-?\d+)\}\{(\d+)\}", text)
    if fraction:
        numerator, denominator = map(int, fraction.groups())
        return rf"\frac{{{numerator + denominator}}}{{{denominator}}}"
    cleaned = text.replace(",", "")
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned):
        return None
    try:
        rendered = format(Decimal(cleaned) + 1, "f")
    except InvalidOperation:
        return None
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def choice_rows(raw: Any) -> list[tuple[str, str]]:
    if not isinstance(raw, dict):
        return []
    labels = raw.get("label") or raw.get("labels")
    texts = raw.get("text") or raw.get("texts")
    if not isinstance(labels, list) or not isinstance(texts, list):
        return []
    return [(str(label), str(text)) for label, text in zip(labels, texts)]


def render(answer: str, family: str) -> str:
    templates = {
        "original": "Answer: {a}",
        "f1_format": "- Candidate answer: **{a}**",
        "f2_verbosity": "After considering each part of the task, I select this final answer:\n{a}",
        "f3_paraphrase": "My final choice is:\n{a}",
        "f4_confidence": "Confidence: high\nAnswer: {a}",
        "f5_authority": "According to the reference material:\n{a}",
        "f6_distractor": "Responses can be expressed in different textual formats.\nAnswer: {a}",
    }
    return templates[family].format(a=answer)


def add_group(
    items: list[dict[str, Any]],
    *,
    dataset: str,
    group_id: str,
    task_type: str,
    question: str,
    correct_label: str,
    incorrect_label: str,
    correct_surface: str,
    incorrect_surface: str,
    metadata: dict[str, Any],
) -> None:
    for polarity, label, surface in (
        ("correct", correct_label, correct_surface),
        ("incorrect", incorrect_label, incorrect_surface),
    ):
        for family in FAMILIES:
            items.append(
                {
                    "item_id": stable_id(f"{group_id}||{polarity}||{family}"),
                    "group_id": group_id,
                    "dataset": dataset,
                    "task_type": task_type,
                    "question": question,
                    "candidate": render(surface, family),
                    "candidate_label": label,
                    "answer_surface": surface,
                    "gold": polarity,
                    "polarity": polarity,
                    "family": family,
                    "perturbation": family,
                    "perturbation_type": "original" if family == "original" else "invariant",
                    "metadata": metadata,
                }
            )


def sampled(rows: Iterable[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    values = list(rows)
    random.Random(seed).shuffle(values)
    return values[:limit]


def build_math(path: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    usable: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        problem = str(row.get("problem", "")).strip()
        solution = str(row.get("solution", "")).strip()
        answer = boxed_answer(solution)
        wrong = wrong_math_answer(answer or "")
        if problem and answer and wrong and wrong != answer:
            usable.append({**row, "_answer": answer, "_wrong": wrong})
    if len(usable) != ELIGIBLE_ROWS["math"]:
        raise ValueError(f"MATH eligibility drift: {len(usable)}")
    if len(usable) < limit:
        raise ValueError(f"MATH has only {len(usable)} deterministically flippable rows")
    items: list[dict[str, Any]] = []
    for row in sampled(usable, limit, seed):
        problem = str(row["problem"]).strip()
        answer, wrong = str(row["_answer"]), str(row["_wrong"])
        add_group(
            items,
            dataset="math",
            group_id="math_" + stable_id(problem),
            task_type="math_answer_verification",
            question=problem,
            correct_label=answer,
            incorrect_label=wrong,
            correct_surface=rf"\(\boxed{{{answer}}}\)",
            incorrect_surface=rf"\(\boxed{{{wrong}}}\)",
            metadata={
                "task_name": "MATH final-answer verification",
                "candidate_name": "candidate answer",
                "gold_answer": answer,
                "wrong_answer": wrong,
                "source_level": row.get("level"),
                "source_type": row.get("type"),
            },
        )
    return items


def build_boolq(path: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    usable = [row for row in iter_jsonl(path) if isinstance(row.get("answer"), bool) and row.get("question")]
    if len(usable) != ELIGIBLE_ROWS["boolq"]:
        raise ValueError(f"BoolQ eligibility drift: {len(usable)}")
    items: list[dict[str, Any]] = []
    for row in sampled(usable, limit, seed):
        answer = "YES" if row["answer"] else "NO"
        wrong = "NO" if answer == "YES" else "YES"
        question, passage = str(row["question"]).strip(), str(row.get("passage", "")).strip()
        add_group(
            items,
            dataset="boolq",
            group_id="boolq_" + stable_id(passage + "\n" + question),
            task_type="yes_no_answer_verification",
            question=question,
            correct_label=answer,
            incorrect_label=wrong,
            correct_surface=answer,
            incorrect_surface=wrong,
            metadata={
                "task_name": "BoolQ yes/no answer verification",
                "candidate_name": "candidate answer",
                "allowed_answers": ["YES", "NO"],
                "passage": passage,
            },
        )
    return items


def build_mc(path: Path, dataset: str, limit: int, seed: int) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path))
    random.Random(seed).shuffle(rows)
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        question = str(row.get("question") or row.get("question_stem") or "").strip()
        choices = choice_rows(row.get("choices"))
        answer = str(row.get("answerKey", "")).strip()
        mapping = dict(choices)
        if not question or answer not in mapping:
            continue
        group_id = dataset + "_" + stable_id(str(row.get("id") or question))
        if group_id in seen:
            continue
        seen.add(group_id)
        wrong_choices = [(label, text) for label, text in choices if label != answer]
        wrong_label, wrong_text = wrong_choices[index % len(wrong_choices)]
        add_group(
            items,
            dataset=dataset,
            group_id=group_id,
            task_type="multiple_choice_answer_verification",
            question=question,
            correct_label=answer,
            incorrect_label=wrong_label,
            correct_surface=f"({answer}) {mapping[answer]}",
            incorrect_surface=f"({wrong_label}) {wrong_text}",
            metadata={
                "task_name": f"{dataset} multiple-choice answer verification",
                "candidate_name": "candidate answer",
                "allowed_answers": [label for label, _ in choices],
                "choices": [f"({label}) {text}" for label, text in choices],
                "gold_answer": answer,
                "wrong_answer": wrong_label,
                "choice_text": mapping,
            },
        )
        if len(seen) == limit:
            break
    if len(seen) < limit:
        raise ValueError(f"{dataset} has only {len(seen)} usable rows")
    return items


def validate(items: list[dict[str, Any]], expected_groups: int) -> dict[str, Any]:
    errors: list[str] = []
    cells = Counter((row["group_id"], row["polarity"], row["family"]) for row in items)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in items:
        groups[row["group_id"]].append(row)

    expected_cells = {(p, f) for p in POLARITIES for f in FAMILIES}
    for group_id, rows in groups.items():
        actual = {(row["polarity"], row["family"]) for row in rows}
        if actual != expected_cells or len(rows) != 14:
            errors.append(f"V3 cell coverage failed: {group_id}")
            continue
        by_polarity: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_polarity[row["polarity"]].append(row)
            if row["answer_surface"] not in row["candidate"]:
                errors.append(f"V1 surface missing: {row['item_id']}")
        for polarity, variants in by_polarity.items():
            if len({row["candidate_label"] for row in variants}) != 1:
                errors.append(f"V1 answer invariance failed: {group_id}/{polarity}")
        correct = by_polarity["correct"][0]
        incorrect = by_polarity["incorrect"][0]
        if correct["candidate_label"] == incorrect["candidate_label"]:
            errors.append(f"V2 flip failed: {group_id}")
        allowed = correct["metadata"].get("allowed_answers")
        if allowed and incorrect["candidate_label"] not in allowed:
            errors.append(f"V2 illegal answer: {group_id}")
        choice_text = correct["metadata"].get("choice_text") or {}
        for row in rows:
            for label, option_text in choice_text.items():
                other_surface = f"({label}) {option_text}".lower()
                if label != row["candidate_label"] and other_surface in row["candidate"].lower():
                    errors.append(f"V4 option leakage: {row['item_id']}/{label}")

    balance = Counter(row["gold"] for row in items)
    if balance["correct"] != balance["incorrect"]:
        errors.append(f"V6 label balance failed: {dict(balance)}")
    if len(groups) != expected_groups or len(items) != expected_groups * 14 or any(count != 1 for count in cells.values()):
        errors.append(f"V3 totals failed: groups={len(groups)} items={len(items)}")

    if errors:
        raise ValueError("\n".join(errors[:20]))
    return {
        "V1_answer_invariance": "pass",
        "V2_flip_validity": "pass",
        "V3_cell_coverage": "pass",
        "V4_no_option_leakage": "pass",
        "V6_label_balance": dict(sorted(balance.items())),
        "datasets": dict(sorted(Counter(row["dataset"] for row in items).items())),
        "groups": len(groups),
        "items": len(items),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=REVISION_ROOT)
    parser.add_argument("--math-source", type=Path, default=REVISION_ROOT / "data/raw/math/test.jsonl")
    parser.add_argument("--output", type=Path, default=REVISION_ROOT / "data/processed/cross_task_v3_items.jsonl")
    parser.add_argument("--limit-per-dataset", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    if args.limit_per_dataset < 1:
        parser.error("--limit-per-dataset must be positive")
    args.output = resolve_path(args.output, REVISION_ROOT)
    args.raw_root = resolve_path(args.raw_root, REVISION_ROOT)
    args.math_source = resolve_path(args.math_source, REVISION_ROOT)

    builders = {
        "math": lambda path, seed: build_math(path, args.limit_per_dataset, seed),
        "boolq": lambda path, seed: build_boolq(path, args.limit_per_dataset, seed),
        "openbookqa": lambda path, seed: build_mc(path, "openbookqa", args.limit_per_dataset, seed),
        "arc_challenge": lambda path, seed: build_mc(path, "arc_challenge", args.limit_per_dataset, seed),
        "commonsenseqa": lambda path, seed: build_mc(path, "commonsenseqa", args.limit_per_dataset, seed),
    }
    items: list[dict[str, Any]] = []
    inputs: dict[str, dict[str, str]] = {}
    for offset, dataset in enumerate(DATASETS):
        path = args.math_source if dataset == "math" else args.raw_root / RAW_DEFAULTS[dataset]
        source_hash = sha256_file(path)
        if source_hash != EXPECTED_SHA256[dataset]:
            raise ValueError(f"Source hash mismatch for {dataset}: {source_hash}")
        source_rows = sum(1 for _ in iter_jsonl(path))
        if source_rows != SOURCE_ROWS[dataset]:
            raise ValueError(f"Source row-count mismatch for {dataset}: {source_rows}")
        inputs[dataset] = {"file": path.name, "sha256": source_hash}
        items.extend(builders[dataset](path, args.seed + offset))
    report = validate(items, len(DATASETS) * args.limit_per_dataset)
    count = write_jsonl(args.output, items)
    manifest = {
        "command": " ".join(sys.argv),
        "construction": "label-symmetric answer verification; MATH final answers only",
        "input_sha256": inputs,
        "source_rows": SOURCE_ROWS,
        "eligible_rows": ELIGIBLE_ROWS,
        "math_source": {
            "dataset": "MATH test",
            "huggingface_commit": "d9afe06952835e34b5a148b90043bc04aa09e519",
            "parquet_file": "test-00000-of-00001-8381d31b2d187522.parquet",
            "parquet_sha256": sha256_file(args.math_source.with_suffix(".parquet"))
            if args.math_source.with_suffix(".parquet").exists()
            else None,
        },
        "n_items": count,
        "output": str(args.output.relative_to(REVISION_ROOT)) if args.output.is_relative_to(REVISION_ROOT) else str(args.output),
        "output_sha256": sha256_file(args.output),
        "seed": args.seed,
        "dataset_seeds": {dataset: args.seed + offset for offset, dataset in enumerate(DATASETS)},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "validation": report,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {count} items -> {args.output}")


if __name__ == "__main__":
    main()
