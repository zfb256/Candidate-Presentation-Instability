from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a model JSON object, tolerating Markdown fences and bare LaTeX escapes."""
    cleaned = text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    cleaned = re.sub(
        r"(?<!\\)\\(?!u[0-9a-fA-F]{4})(?=[A-Za-z]{2,}|[()[\]])",
        r"\\\\",
        cleaned,
    )
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        value = json.loads(re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r"\\\\", cleaned))
    if not isinstance(value, dict):
        raise ValueError("response JSON is not an object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def require_unique(rows: Iterable[dict[str, Any]], key: str, label: str) -> None:
    seen: set[Any] = set()
    for row in rows:
        value = row[key]
        if value in seen:
            raise ValueError(f"{label} contains duplicate {key}: {value}")
        seen.add(value)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    count = 0
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    tmp_path.replace(path)
    return count


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()
