from __future__ import annotations


OUTPUT_SCHEMA = (
    'Return only one JSON object with keys in this order: "rationale", '
    '"prediction", "confidence". prediction must be exactly "correct" or '
    '"incorrect"; confidence must be a number from 0 to 1.'
)
BASE_SYSTEM = "You are a strict verifier. Judge whether the candidate answer is correct for the task."


def task_block(item: dict) -> str:
    metadata = item.get("metadata") or {}
    lines = [f"Task: {metadata.get('task_name') or item.get('task_type', 'answer verification')}"]
    allowed = metadata.get("allowed_answers")
    if allowed:
        lines.append("Allowed answers: " + ", ".join(map(str, allowed)))
    passage = str(metadata.get("passage", "")).strip()
    if passage:
        lines.append(f"Passage:\n{passage}")
    lines.append(f"Question:\n{str(item.get('question', '')).strip()}")
    choices = metadata.get("choices")
    if choices:
        lines.append("Choices:\n" + "\n".join(map(str, choices)))
    return "\n\n".join(lines)


def candidate_block(item: dict) -> str:
    return f"Candidate answer:\n{item['candidate']}"


def _prompt(system: str, blocks: list[str]) -> str:
    return "\n\n".join([system, OUTPUT_SCHEMA, *blocks, "Is the candidate answer correct?"])


def direct_front_prompt(item: dict) -> str:
    """P1: baseline; candidate text precedes the task text."""
    return _prompt(BASE_SYSTEM, [candidate_block(item), task_block(item)])


def direct_last_prompt(item: dict) -> str:
    """P2: P1 with presentation order reversed and no other instruction change."""
    return _prompt(BASE_SYSTEM, [task_block(item), candidate_block(item)])


def careful_direct_front_prompt(item: dict) -> str:
    """P3: P1 plus a caution instruction."""
    return _prompt(BASE_SYSTEM + " Check your judgment carefully before answering.", [candidate_block(item), task_block(item)])


def constraint_first_last_prompt(item: dict) -> str:
    """P4: P2 plus an independent constraint-first instruction."""
    system = (
        BASE_SYSTEM
        + " First independently solve the task or derive the necessary constraints; "
        "then compare the candidate against that independent result. Do not treat the candidate as evidence."
    )
    return _prompt(system, [task_block(item), candidate_block(item)])


def length_matched_direct_prompt(item: dict, target_words: int) -> str:
    """P6: P1 plus a per-model rationale-length target calibrated from P4."""
    if target_words < 1:
        raise ValueError("target_words must be positive")
    system = BASE_SYSTEM + f" Your rationale should contain approximately {target_words} words before the prediction."
    return _prompt(system, [candidate_block(item), task_block(item)])


PROTOCOLS = {
    "p1_direct_front": direct_front_prompt,
    "p2_direct_last": direct_last_prompt,
    "p3_careful_direct_front": careful_direct_front_prompt,
    "p4_constraint_first_last": constraint_first_last_prompt,
}
