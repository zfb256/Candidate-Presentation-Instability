#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "src"), str(ROOT / "scripts")])

from bav.io import extract_json_object, read_jsonl, sha256_file
from bav.metrics import (
    accuracy,
    bootstrap_ci,
    bootstrap_invariant_flip_breakdown_ci,
    bootstrap_metric_cis,
    invariant_flip_breakdown,
    paired_bootstrap_delta_cis,
    responsiveness_rate,
)
from bav.prompts import (
    careful_direct_front_prompt,
    candidate_block,
    constraint_first_last_prompt,
    direct_front_prompt,
    direct_last_prompt,
    length_matched_direct_prompt,
    task_block,
)
from run_two_stage_selective import stage_a_prompt, stage_b_prompt
from calibrate_p6 import completion_tokens
from summarize_api_cost import api_cost_yuan
from aggregate_evaluations import evaluation_row
import run_openai_compatible as api_runner
import run_vllm as local_runner

def main() -> None:
    items_path = ROOT / "data/processed/cross_task_v3_items.jsonl"
    items = read_jsonl(items_path)
    assert sha256_file(items_path) == "73db47b2bbe4ee3c65710aab0b0eee0a17c42f349abf52560fef04d9907304de"
    assert len(items) == 10_500
    assert len({row["item_id"] for row in items}) == 10_500
    assert len({row["group_id"] for row in items}) == 750
    assert sum(row["gold"] == "correct" for row in items) == 5_250
    assert not any(".. is" in row["candidate"] or row["candidate"].endswith("..") for row in items)
    probe = read_jsonl(ROOT / "data/processed/cross_task_v3_probe200.jsonl")
    assert len(probe) == 2_800 and len({row["group_id"] for row in probe}) == 200
    assert {row["item_id"] for row in probe} <= {row["item_id"] for row in items}

    item = items[0]
    candidate, task = candidate_block(item), task_block(item)
    p1, p2 = direct_front_prompt(item), direct_last_prompt(item)
    caution = " Check your judgment carefully before answering."
    constraint = (
        " First independently solve the task or derive the necessary constraints; "
        "then compare the candidate against that independent result. Do not treat the candidate as evidence."
    )
    assert p1.index(candidate) < p1.index(task)
    assert p2.index(task) < p2.index(candidate)
    assert careful_direct_front_prompt(item).replace(caution, "", 1) == p1
    assert constraint_first_last_prompt(item).replace(constraint, "", 1) == p2
    assert "approximately 37 words" in length_matched_direct_prompt(item, 37)
    stage_a = stage_a_prompt(item)
    assert candidate not in stage_a and item["candidate"] not in stage_a
    assert candidate in stage_b_prompt(item, {"expected_result": "x"}, 0.7)
    by_group: dict[str, set[str]] = {}
    for row in items:
        by_group.setdefault(row["group_id"], set()).add(task_block(row))
    assert all(len(tasks) == 1 for tasks in by_group.values())

    parsed = extract_json_object(
        '```json\n{"rationale":"Use \\(x\\) and \\boxed{1}","prediction":"correct","confidence":1}\n```'
    )
    assert parsed["prediction"] == "correct" and r"\boxed" in parsed["rationale"]
    mixed_escapes = extract_json_object(
        r'{"rationale":"Use \theta, \beta, and \cdots with \\left(\\frac{x}{2}\\right)","prediction":"correct","confidence":1}'
    )
    assert mixed_escapes["rationale"] == r"Use \theta, \beta, and \cdots with \left(\frac{x}{2}\right)"
    assert extract_json_object(r'{"rationale":"\u03b8","prediction":"correct","confidence":1}')["rationale"] == "θ"

    calls: list[str] = []
    responses = iter(
        [
            ('{"rationale":"","prediction":"correct","confidence":1}', {"usage": {"total_tokens": 1}}),
            ('{"rationale":"ok","prediction":"correct","confidence":1}', {"usage": {"total_tokens": 2}}),
        ]
    )
    request_args = {
        "base_url": "https://example.invalid",
        "api_key": "x",
        "model": "mock",
        "prompt": "frozen prompt",
        "temperature": 0,
        "timeout": 1,
        "json_mode": True,
        "max_tokens": 10,
        "reasoning_effort": None,
        "thinking": "disabled",
    }
    original_completion = api_runner.chat_completion
    api_runner.chat_completion = lambda **kwargs: (calls.append(kwargs["prompt"]) or next(responses))
    try:
        retried, attempts, errors, _ = api_runner.request_json(**request_args)
    finally:
        api_runner.chat_completion = original_completion
    assert retried and retried["prediction"] == "correct"
    assert calls == ["frozen prompt", "frozen prompt"] and len(attempts) == 2 and len(errors) == 1

    api_runner.chat_completion = lambda **_: ("{", {"finish_reason": "length"})
    try:
        truncated, attempts, errors, _ = api_runner.request_json(**request_args)
    finally:
        api_runner.chat_completion = original_completion
    assert truncated is None and len(attempts) == 1 and errors == ["finish_reason=length"]

    texts = [
        '{"rationale":"c","prediction":"correct","confidence":0.9}',
        '{"rationale":"c","prediction":"correct","confidence":0.8}',
        '{"rationale":"c","prediction":"correct","confidence":0.7}',
        '{"rationale":"i","prediction":"incorrect","confidence":0.6}',
        '{',
    ]
    fake_result = SimpleNamespace(
        prompt_token_ids=[1, 2],
        outputs=[SimpleNamespace(text=text, token_ids=[1]) for text in texts],
    )
    voted = local_runner.aggregate_samples({"item_id": "x"}, fake_result, "mock")
    assert voted["prediction"] == "correct" and voted["confidence"] == 0.6
    assert voted["vote_counts"] == {"correct": 3, "incorrect": 1, "invalid": 1}
    assert voted["prompt_tokens"] == 10 and voted["attempt_count"] == 5

    assert completion_tokens({"attempts": [{"usage": {"completion_tokens": 7}}]}) == 7
    assert api_cost_yuan("deepseek-v4-flash", {"prompt_cache_miss_tokens": 1_000_000}) == 1
    assert api_cost_yuan("deepseek-v4-flash", {"prompt_cache_hit_tokens": 1_000_000}) == 0.02
    assert api_cost_yuan("deepseek-v4-flash", {"completion_tokens": 1_000_000}) == 2
    assert api_cost_yuan("doubao-seed-2-1-pro-260628", {"prompt_tokens": 1_000_000}) == 6
    assert api_cost_yuan("doubao-seed-2-1-pro-260628", {"completion_tokens": 1_000_000}) == 30
    compact = evaluation_row("p1", "mock", {
        "n_outputs": 1, "accuracy": 1, "ci95": {"accuracy": [1, 1], "invariant_flip_rate": [0, 0]},
        "incorrect_f1": {"f1": 1}, "brier_score": 0, "invariant_flip": {"overall": {"rate": 0}},
        "responsiveness": {"rate": 1}, "coverage": {"invalid_rate": 0},
    })
    assert compact["model"] == "mock" and compact["delta_accuracy"] is None

    oracle = {row["item_id"]: {"prediction": row["gold"], "confidence": 1} for row in items}
    constant = {row["item_id"]: {"prediction": "correct", "confidence": 1} for row in items}
    assert accuracy(items, oracle) == 1
    assert invariant_flip_breakdown(items, oracle)["overall"]["rate"] == 0
    assert responsiveness_rate(items, oracle)["rate"] == 1
    assert invariant_flip_breakdown(items, constant)["overall"]["rate"] == 0
    assert responsiveness_rate(items, constant)["rate"] == 0
    perturbed = {item_id: dict(output) for item_id, output in oracle.items()}
    first_group = items[0]["group_id"]
    targets = [
        row
        for row in items
        if row["group_id"] == first_group
        and ((row["polarity"], row["family"]) == ("correct", "f1_format")
             or (row["polarity"], row["family"]) == ("incorrect", "f2_verbosity"))
    ]
    for row in targets:
        perturbed[row["item_id"]]["prediction"] = "incorrect" if row["polarity"] == "correct" else "correct"
    breakdown = invariant_flip_breakdown(items, perturbed)
    assert breakdown["overall"]["flipped"] == 2 and breakdown["overall"]["comparisons"] == 9000
    assert breakdown["by_polarity"]["correct"]["flipped"] == 1
    assert breakdown["by_polarity"]["incorrect"]["flipped"] == 1
    assert breakdown["by_family"]["f1_format"]["flipped"] == 1
    assert breakdown["by_family"]["f2_verbosity"]["flipped"] == 1
    invalid_rr = {item_id: dict(output) for item_id, output in oracle.items()}
    incorrect_original = next(
        row for row in items if row["polarity"] == "incorrect" and row["family"] == "original"
    )
    invalid_rr[incorrect_original["item_id"]]["prediction"] = "invalid"
    assert responsiveness_rate(items, invalid_rr)["rate"] == 749 / 750
    ci = bootstrap_invariant_flip_breakdown_ci(items, oracle, rounds=10)
    assert ci["overall"] == (0.0, 0.0) and all(value == (0.0, 0.0) for value in ci["by_family"].values())
    fast_cis = bootstrap_metric_cis(items, perturbed, rounds=20)
    for fast_name, reference_name in (
        ("accuracy", "accuracy"),
        ("incorrect_f1", "f1"),
        ("brier_score", "brier"),
        ("responsiveness", "responsiveness"),
    ):
        assert fast_cis[fast_name] == bootstrap_ci(items, perturbed, reference_name, rounds=20)
    assert bootstrap_invariant_flip_breakdown_ci(items, perturbed, rounds=20)["overall"] == bootstrap_ci(
        items, perturbed, "flip_rate", rounds=20
    )
    paired_cis = paired_bootstrap_delta_cis(items, oracle, oracle, rounds=20)
    assert all(value == (0.0, 0.0) for value in paired_cis.values())

    prompt_hashes = {
        "p1_direct_front": "659fc919208632e07475fced4f56d3ee5b04506805b9d80a11bb9fe856138c75",
        "p2_direct_last": "e56ff57fffd54e3af209ef3914d983c68da48f984e8e15d97d70e256cff57e4d",
        "p3_careful_direct_front": "7da5fff8f30e311faf435b8910bf0f6bc944779e62358ebcd2b352506bdc6f56",
        "p4_constraint_first_last": "14ad6d777a0eed069f474ed342706f46d860d030b1772bddf23b2b8fd7d55d97",
    }
    for protocol, expected_hash in prompt_hashes.items():
        prompt_path = ROOT / f"runs/{protocol}/prompts.jsonl"
        manifest = json.loads(prompt_path.with_suffix(".jsonl.manifest.json").read_text())
        assert manifest["n_prompts"] == 10_500
        assert manifest["items_sha256"] == sha256_file(items_path)
        assert manifest["output_sha256"] == sha256_file(prompt_path)
        assert manifest["output_sha256"] == expected_hash

    repeat = json.loads((ROOT / "reports/formal/exact_repeat.json").read_text())
    assert len(repeat["models"]) == 6
    assert repeat["models"]["qwen2.5-14b"]["repeat"]["flips"] == 101
    assert repeat["models"]["deepseek-v4-flash"]["repeat"]["flips"] == 81
    qwen_p5 = json.loads((ROOT / "runs/formal/p5/qwen2.5-14b.evaluation.json").read_text())
    deepseek_p5 = json.loads((ROOT / "runs/formal/p5/deepseek-v4-flash.evaluation.json").read_text())
    assert qwen_p5["bootstrap_rounds"] == deepseek_p5["bootstrap_rounds"] == 10_000
    assert qwen_p5["n_outputs"] == 10_500
    assert qwen_p5["invariant_flip"]["overall"]["flipped"] == 113
    assert qwen_p5["delta_vs_baseline"]["paired_valid_invariant_flip_rate"]["method_flipped"] == 110
    valid_labels = {"correct", "incorrect"}
    for path, expected in (
        (ROOT / "runs/formal/p5/deepseek-v4-flash.jsonl", (65, 18, 182, 0)),
        (ROOT / "runs/formal/p5/qwen2.5-14b.jsonl", (27, 75, 1431, 1197)),
    ):
        rows = read_jsonl(path)
        observed = (
            sum(row.get("independent_prediction") not in valid_labels for row in rows),
            sum(row.get("independent_prediction") in valid_labels
                and row.get("prediction") in valid_labels
                and row["independent_prediction"] != row["prediction"] for row in rows),
            sum(row.get("used_fallback") is True for row in rows),
            sum(row.get("used_fallback") is True and row.get("fallback_required") is False for row in rows),
        )
        assert observed == expected
    qwen3_p4 = [row for row in read_jsonl(ROOT / "runs/formal/p4/qwen2.5-3b.jsonl") if row["prediction"] == "invalid"]
    assert len(qwen3_p4) == 2_458
    assert sum(row["error"] == "prediction is not exactly correct/incorrect" for row in qwen3_p4) == 3
    assert sum(row["error"] != "prediction is not exactly correct/incorrect" for row in qwen3_p4) == 2_455
    assert sum(row["completion_tokens"] == 512 for row in qwen3_p4) == 1_388
    assert all(not row["raw"].lstrip().startswith("{") for row in qwen3_p4)

    print("artifact self-check: pass")


if __name__ == "__main__":
    main()
