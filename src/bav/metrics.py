from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean
from typing import Callable


BINARY_GOLDS = {"correct", "incorrect"}


def binary_items(items: list[dict]) -> list[dict]:
    return [item for item in items if item.get("gold") in BINARY_GOLDS]


def normalize_prediction(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"correct", "true", "yes", "1"}:
        return "correct"
    if text in {"incorrect", "false", "no", "0", "wrong"}:
        return "incorrect"
    return "invalid"


def normalize_confidence(value: object) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.5
    return min(1.0, max(0.0, confidence))


def coverage_stats(items: list[dict], outputs: dict[str, dict]) -> dict[str, float | int]:
    eval_items = binary_items(items)
    missing = 0
    invalid = 0
    for item in eval_items:
        output = outputs.get(item["item_id"])
        if output is None:
            missing += 1
            continue
        invalid += int(normalize_prediction(output.get("prediction")) == "invalid")
    total = len(eval_items)
    return {
        "n_eval_items": total,
        "missing": missing,
        "invalid": invalid,
        "missing_rate": missing / total if total else 0.0,
        "invalid_rate": invalid / (total - missing) if total - missing else 0.0,
    }


def accuracy(items: list[dict], outputs: dict[str, dict]) -> float:
    total = 0
    right = 0
    for item in binary_items(items):
        output = outputs.get(item["item_id"])
        if output is None:
            continue
        total += 1
        right += int(normalize_prediction(output.get("prediction")) == item["gold"])
    return right / total if total else 0.0


def f1_for_incorrect(items: list[dict], outputs: dict[str, dict]) -> dict[str, float]:
    tp = fp = fn = 0
    for item in binary_items(items):
        output = outputs.get(item["item_id"])
        if output is None:
            continue
        pred = normalize_prediction(output.get("prediction"))
        gold_positive = item["gold"] == "incorrect"
        pred_positive = pred == "incorrect"
        tp += int(gold_positive and pred_positive)
        fp += int(not gold_positive and pred_positive)
        fn += int(gold_positive and not pred_positive)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def brier_score(items: list[dict], outputs: dict[str, dict]) -> float:
    scores: list[float] = []
    for item in binary_items(items):
        output = outputs.get(item["item_id"])
        if output is None:
            continue
        confidence = normalize_confidence(output.get("confidence", 0.5))
        pred = normalize_prediction(output.get("prediction"))
        if pred == "correct":
            prob_correct = confidence
        elif pred == "incorrect":
            prob_correct = 1.0 - confidence
        else:
            prob_correct = 0.5
        gold_correct = 1.0 if item["gold"] == "correct" else 0.0
        scores.append((prob_correct - gold_correct) ** 2)
    return mean(scores) if scores else 0.0


def invariant_flip_stats(
    items: list[dict],
    outputs: dict[str, dict],
    *,
    polarity: str | None = None,
    family: str | None = None,
) -> dict[str, float | int]:
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in binary_items(items):
        item_polarity = item.get("polarity", item["gold"])
        if polarity is not None and item_polarity != polarity:
            continue
        if item["perturbation_type"] in {"original", "invariant"}:
            if family is None or item.get("family", item.get("perturbation")) in {"original", family}:
                by_group[(item["group_id"], item_polarity)].append(item)

    comparisons = 0
    flipped = 0
    missing_original = 0
    missing_invariant = 0
    invalid_pairs = 0
    for group_items in by_group.values():
        originals = [item for item in group_items if item.get("family", item.get("perturbation")) == "original"]
        invariants = [item for item in group_items if item.get("family", item.get("perturbation")) != "original"]
        if not originals:
            continue
        original = originals[0]
        original_output = outputs.get(original["item_id"])
        if original_output is None:
            missing_original += len(invariants)
            continue
        original_pred = normalize_prediction(original_output.get("prediction"))
        for invariant in invariants:
            invariant_output = outputs.get(invariant["item_id"])
            if invariant_output is None:
                missing_invariant += 1
                continue
            invariant_pred = normalize_prediction(invariant_output.get("prediction"))
            if "invalid" in {original_pred, invariant_pred}:
                invalid_pairs += 1
                continue
            comparisons += 1
            flipped += int(invariant_pred != original_pred)
    return {
        "rate": flipped / comparisons if comparisons else 0.0,
        "flipped": flipped,
        "comparisons": comparisons,
        "missing_original": missing_original,
        "missing_invariant": missing_invariant,
        "invalid_pairs": invalid_pairs,
    }


def invariant_flip_breakdown(items: list[dict], outputs: dict[str, dict]) -> dict:
    families = sorted(
        {
            item.get("family", item.get("perturbation"))
            for item in binary_items(items)
            if item.get("family", item.get("perturbation")) != "original"
        }
    )
    return {
        "overall": invariant_flip_stats(items, outputs),
        "by_polarity": {
            polarity: invariant_flip_stats(items, outputs, polarity=polarity)
            for polarity in ("correct", "incorrect")
        },
        "by_family": {family: invariant_flip_stats(items, outputs, family=family) for family in families},
    }


def responsiveness_rate(items: list[dict], outputs: dict[str, dict]) -> dict[str, float | int]:
    target = [
        item
        for item in binary_items(items)
        if item.get("polarity", item["gold"]) == "incorrect"
        and item.get("family", item.get("perturbation")) == "original"
    ]
    predictions = [
        normalize_prediction(outputs[item["item_id"]].get("prediction"))
        for item in target
        if item["item_id"] in outputs
    ]
    valid = [prediction for prediction in predictions if prediction != "invalid"]
    rejected = sum(prediction == "incorrect" for prediction in predictions)
    return {
        "rate": rejected / len(target) if target else 0.0,
        "rejected": rejected,
        "valid": len(valid),
        "invalid": len(predictions) - len(valid),
        "missing": len(target) - len(predictions),
    }


def paired_valid_invariant_flip_delta(
    items: list[dict],
    baseline_outputs: dict[str, dict],
    method_outputs: dict[str, dict],
) -> dict[str, float | int]:
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in binary_items(items):
        if item["perturbation_type"] in {"original", "invariant"}:
            by_group[(item["group_id"], item.get("polarity", item["gold"]))].append(item)

    comparisons = 0
    baseline_flipped = 0
    method_flipped = 0
    skipped_invalid_pairs = 0
    skipped_missing_pairs = 0
    for group_items in by_group.values():
        originals = [item for item in group_items if item.get("family", item.get("perturbation")) == "original"]
        invariants = [item for item in group_items if item.get("family", item.get("perturbation")) != "original"]
        if not originals:
            continue
        original = originals[0]
        for invariant in invariants:
            rows = (
                baseline_outputs.get(original["item_id"]),
                baseline_outputs.get(invariant["item_id"]),
                method_outputs.get(original["item_id"]),
                method_outputs.get(invariant["item_id"]),
            )
            if any(row is None for row in rows):
                skipped_missing_pairs += 1
                continue
            base_original, base_invariant, method_original, method_invariant = rows
            predictions = (
                normalize_prediction(base_original.get("prediction")),
                normalize_prediction(base_invariant.get("prediction")),
                normalize_prediction(method_original.get("prediction")),
                normalize_prediction(method_invariant.get("prediction")),
            )
            if "invalid" in predictions:
                skipped_invalid_pairs += 1
                continue
            comparisons += 1
            baseline_flipped += int(predictions[0] != predictions[1])
            method_flipped += int(predictions[2] != predictions[3])

    baseline_rate = baseline_flipped / comparisons if comparisons else 0.0
    method_rate = method_flipped / comparisons if comparisons else 0.0
    return {
        "delta": method_rate - baseline_rate,
        "baseline_rate": baseline_rate,
        "method_rate": method_rate,
        "baseline_flipped": baseline_flipped,
        "method_flipped": method_flipped,
        "comparisons": comparisons,
        "skipped_invalid_pairs": skipped_invalid_pairs,
        "skipped_missing_pairs": skipped_missing_pairs,
    }


def invariant_flip_rate(items: list[dict], outputs: dict[str, dict]) -> float:
    return float(invariant_flip_stats(items, outputs)["rate"])


def _by_group(items: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[item["group_id"]].append(item)
    return groups


def _sample_by_group(items: list[dict], rng: random.Random) -> list[dict]:
    groups = _by_group(items)
    group_ids = sorted(groups)
    sampled_items: list[dict] = []
    for draw, group_id in enumerate(rng.choice(group_ids) for _ in group_ids):
        sampled_items.extend({**item, "group_id": f"{draw}:{group_id}"} for item in groups[group_id])
    return sampled_items


def bootstrap_ci(
    items: list[dict],
    outputs: dict[str, dict],
    metric_name: str,
    rounds: int = 1000,
    seed: int = 13,
) -> tuple[float, float]:
    metric = metric_fn(metric_name)
    rng = random.Random(seed)
    values = [metric(_sample_by_group(items, rng), outputs) for _ in range(rounds)]
    return percentile_ci(values)


def bootstrap_metric_cis(
    items: list[dict], outputs: dict[str, dict], rounds: int = 1000, seed: int = 13
) -> dict[str, tuple[float, float]]:
    clusters: dict[str, list[float]] = {
        group_id: [0.0] * 10 for group_id in {item["group_id"] for item in items}
    }
    for item in binary_items(items):
        values = clusters[item["group_id"]]
        output = outputs.get(item["item_id"])
        if item.get("polarity", item["gold"]) == "incorrect" and item.get("family", item.get("perturbation")) == "original":
            values[8] += int(output is not None and normalize_prediction(output.get("prediction")) == "incorrect")
            values[9] += 1
        if output is None:
            continue
        prediction = normalize_prediction(output.get("prediction"))
        values[0] += int(prediction == item["gold"])
        values[1] += 1
        confidence = normalize_confidence(output.get("confidence", 0.5))
        probability = confidence if prediction == "correct" else 1 - confidence if prediction == "incorrect" else 0.5
        values[2] += (probability - (item["gold"] == "correct")) ** 2
        values[3] += 1
        gold_positive, pred_positive = item["gold"] == "incorrect", prediction == "incorrect"
        values[4] += int(gold_positive and pred_positive)
        values[5] += int(not gold_positive and pred_positive)
        values[6] += int(gold_positive and not pred_positive)

    group_ids = sorted(clusters)
    rng = random.Random(seed)
    sampled_metrics = {key: [] for key in ("accuracy", "incorrect_f1", "brier_score", "responsiveness")}
    for _ in range(rounds):
        totals = [0.0] * 10
        for _ in group_ids:
            cluster = clusters[rng.choice(group_ids)]
            totals = [left + right for left, right in zip(totals, cluster)]
        sampled_metrics["accuracy"].append(totals[0] / totals[1] if totals[1] else 0.0)
        sampled_metrics["brier_score"].append(totals[2] / totals[3] if totals[3] else 0.0)
        precision = totals[4] / (totals[4] + totals[5]) if totals[4] + totals[5] else 0.0
        recall = totals[4] / (totals[4] + totals[6]) if totals[4] + totals[6] else 0.0
        sampled_metrics["incorrect_f1"].append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        sampled_metrics["responsiveness"].append(totals[8] / totals[9] if totals[9] else 0.0)
    return {key: percentile_ci(values) for key, values in sampled_metrics.items()}


def bootstrap_invariant_flip_breakdown_ci(
    items: list[dict], outputs: dict[str, dict], rounds: int = 1000, seed: int = 13
) -> dict:
    families = sorted({item["family"] for item in items if item.get("family") not in {None, "original"}})
    keys = ["overall", "polarity:correct", "polarity:incorrect", *(f"family:{family}" for family in families)]
    clusters: dict[str, dict[str, list[int]]] = {
        group_id: {key: [0, 0] for key in keys} for group_id in {item["group_id"] for item in items}
    }
    paired: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in binary_items(items):
        paired[(item["group_id"], item.get("polarity", item["gold"]))].append(item)
    for (group_id, polarity), rows in paired.items():
        original = next(row for row in rows if row["family"] == "original")
        original_output = outputs.get(original["item_id"])
        if original_output is None:
            continue
        original_prediction = normalize_prediction(original_output.get("prediction"))
        for variant in (row for row in rows if row["family"] != "original"):
            variant_output = outputs.get(variant["item_id"])
            if variant_output is None:
                continue
            variant_prediction = normalize_prediction(variant_output.get("prediction"))
            if "invalid" in {original_prediction, variant_prediction}:
                continue
            flipped = int(original_prediction != variant_prediction)
            for key in ("overall", f"polarity:{polarity}", f"family:{variant['family']}"):
                clusters[group_id][key][0] += flipped
                clusters[group_id][key][1] += 1
    group_ids = sorted(clusters)
    rng = random.Random(seed)
    values = {key: [] for key in keys}
    for _ in range(rounds):
        sampled = [clusters[rng.choice(group_ids)] for _ in group_ids]
        for key in keys:
            flipped = sum(cluster[key][0] for cluster in sampled)
            comparisons = sum(cluster[key][1] for cluster in sampled)
            values[key].append(flipped / comparisons if comparisons else 0.0)
    return {
        "overall": percentile_ci(values["overall"]),
        "by_polarity": {
            polarity: percentile_ci(values[f"polarity:{polarity}"]) for polarity in ("correct", "incorrect")
        },
        "by_family": {family: percentile_ci(values[f"family:{family}"]) for family in families},
    }


def paired_bootstrap_delta_cis(
    items: list[dict],
    baseline_outputs: dict[str, dict],
    method_outputs: dict[str, dict],
    rounds: int = 1000,
    seed: int = 13,
) -> dict[str, tuple[float, float]]:
    baseline_outputs, method_outputs = paired_output_intersection(baseline_outputs, method_outputs)
    # Per model: correct, total, TP, FP, FN, invariant flips, comparisons;
    # then jointly valid baseline flips, method flips, comparisons.
    clusters = {group_id: [0] * 17 for group_id in {item["group_id"] for item in items}}
    for item in binary_items(items):
        item_id = item["item_id"]
        if item_id not in baseline_outputs:
            continue
        for offset, outputs in ((0, baseline_outputs), (7, method_outputs)):
            prediction = normalize_prediction(outputs[item_id].get("prediction"))
            gold_positive = item["gold"] == "incorrect"
            pred_positive = prediction == "incorrect"
            values = clusters[item["group_id"]]
            values[offset] += int(prediction == item["gold"])
            values[offset + 1] += 1
            values[offset + 2] += int(gold_positive and pred_positive)
            values[offset + 3] += int(not gold_positive and pred_positive)
            values[offset + 4] += int(gold_positive and not pred_positive)

    paired: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in binary_items(items):
        paired[(item["group_id"], item.get("polarity", item["gold"]))].append(item)
    for (group_id, _), rows in paired.items():
        original = next(row for row in rows if row["family"] == "original")
        for invariant in (row for row in rows if row["family"] != "original"):
            item_ids = original["item_id"], invariant["item_id"]
            if any(item_id not in baseline_outputs for item_id in item_ids):
                continue
            predictions = tuple(
                normalize_prediction(outputs[item_id].get("prediction"))
                for outputs in (baseline_outputs, method_outputs)
                for item_id in item_ids
            )
            values = clusters[group_id]
            for offset, pair_predictions in ((0, predictions[:2]), (7, predictions[2:])):
                if "invalid" not in pair_predictions:
                    values[offset + 5] += int(pair_predictions[0] != pair_predictions[1])
                    values[offset + 6] += 1
            if "invalid" not in predictions:
                values[14] += int(predictions[0] != predictions[1])
                values[15] += int(predictions[2] != predictions[3])
                values[16] += 1

    def f1(values: list[int], offset: int) -> float:
        tp, fp, fn = values[offset + 2 : offset + 5]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    sampled = {name: [] for name in ("accuracy", "incorrect_f1", "invariant_flip_rate", "paired_valid_invariant_flip_rate")}
    group_ids = sorted(clusters)
    rng = random.Random(seed)
    for _ in range(rounds):
        totals = [0] * 17
        for _ in group_ids:
            cluster = clusters[rng.choice(group_ids)]
            totals = [left + right for left, right in zip(totals, cluster)]
        sampled["accuracy"].append(
            (totals[7] / totals[8] if totals[8] else 0.0) - (totals[0] / totals[1] if totals[1] else 0.0)
        )
        sampled["incorrect_f1"].append(f1(totals, 7) - f1(totals, 0))
        sampled["invariant_flip_rate"].append(
            (totals[12] / totals[13] if totals[13] else 0.0)
            - (totals[5] / totals[6] if totals[6] else 0.0)
        )
        sampled["paired_valid_invariant_flip_rate"].append(
            (totals[15] / totals[16] - totals[14] / totals[16]) if totals[16] else 0.0
        )
    return {name: percentile_ci(values) for name, values in sampled.items()}


def metric_fn(metric_name: str) -> Callable[[list[dict], dict[str, dict]], float]:
    if metric_name == "accuracy":
        return accuracy
    if metric_name == "flip_rate":
        return invariant_flip_rate
    if metric_name == "f1":
        return lambda items, outputs: f1_for_incorrect(items, outputs)["f1"]
    if metric_name == "brier":
        return brier_score
    if metric_name == "responsiveness":
        return lambda items, outputs: float(responsiveness_rate(items, outputs)["rate"])
    raise ValueError(f"Unknown metric: {metric_name}")


def paired_output_intersection(
    baseline_outputs: dict[str, dict], method_outputs: dict[str, dict]
) -> tuple[dict[str, dict], dict[str, dict]]:
    common_ids = baseline_outputs.keys() & method_outputs.keys()
    return (
        {item_id: baseline_outputs[item_id] for item_id in common_ids},
        {item_id: method_outputs[item_id] for item_id in common_ids},
    )


def paired_coverage_stats(
    items: list[dict],
    baseline_outputs: dict[str, dict],
    method_outputs: dict[str, dict],
) -> dict[str, float | int]:
    eval_ids = {item["item_id"] for item in binary_items(items)}
    baseline_ids = set(baseline_outputs)
    method_ids = set(method_outputs)
    common_ids = eval_ids & baseline_ids & method_ids
    return {
        "n_eval_items": len(eval_ids),
        "baseline_eval_outputs": len(eval_ids & baseline_ids),
        "method_eval_outputs": len(eval_ids & method_ids),
        "paired_eval_outputs": len(common_ids),
        "paired_eval_rate": len(common_ids) / len(eval_ids) if eval_ids else 0.0,
        "baseline_only_eval_outputs": len((eval_ids & baseline_ids) - method_ids),
        "method_only_eval_outputs": len((eval_ids & method_ids) - baseline_ids),
    }


def percentile_ci(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    values = sorted(values)
    low_idx = int(0.025 * len(values))
    high_idx = max(0, int(0.975 * len(values)) - 1)
    return values[low_idx], values[high_idx]
