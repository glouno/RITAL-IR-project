"""Dataset download and inspection helpers for the HiRAG paper."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlopen

from .constants import ALL_DATASETS, MAIN_PAPER_DATASETS


DEFAULT_DATA_DIR = Path("data/ultradomain")


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def download_file(url: str, destination: str | Path, chunk_size: int = 2**20) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(url) as response, destination.open("wb") as output:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            output.write(chunk)

    return destination


def download_ultradomain_dataset(name: str, output_dir: str | Path = DEFAULT_DATA_DIR) -> Path:
    spec = ALL_DATASETS.get(name)
    if spec is None:
        raise KeyError(f"Unknown dataset: {name}")
    if not spec.url.endswith(".jsonl"):
        raise ValueError(f"{name} is not a direct JSONL download target.")

    destination = Path(output_dir) / f"{name}.jsonl"
    if destination.exists():
        return destination
    return download_file(spec.url, destination)


def list_main_dataset_names() -> list[str]:
    return [spec.name for spec in MAIN_PAPER_DATASETS]


def collect_dataset_stats(path: str | Path, limit: int | None = None) -> dict[str, Any]:
    rows = 0
    labels: dict[str, int] = {}
    total_context_chars = 0
    total_question_chars = 0

    for row in iter_jsonl(path):
        rows += 1
        label = row.get("label") or row.get("dataset") or "unknown"
        labels[label] = labels.get(label, 0) + 1
        total_context_chars += len(str(row.get("context", "")))
        total_question_chars += len(str(row.get("input", "")))
        if limit is not None and rows >= limit:
            break

    average_context_chars = total_context_chars / rows if rows else 0
    average_question_chars = total_question_chars / rows if rows else 0

    return {
        "path": str(path),
        "rows": rows,
        "labels": labels,
        "average_context_chars": round(average_context_chars, 2),
        "average_question_chars": round(average_question_chars, 2),
        "size_bytes": os.path.getsize(path),
    }
