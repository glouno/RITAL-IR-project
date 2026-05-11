#!/usr/bin/env python3
"""Submit a JSONL file to OpenAI Batch and optionally wait for outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI


TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Upload a JSONL request file, create an OpenAI Batch, poll, and download results."
    )
    parser.add_argument("--batch-file", required=True)
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--completion-window", default="24h")
    parser.add_argument("--output-dir", default=f".runs/openai_batch_submit/{timestamp}")
    parser.add_argument("--metadata", nargs="*", default=[])
    parser.add_argument("--description", default=None)
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional dotenv-style file to load before reading OPENAI_API_KEY.",
    )
    parser.add_argument("--poll-interval-seconds", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=int, default=0, help="0 means no timeout")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                raise ValueError(f"{path}:{line_number}: empty environment key")
            os.environ.setdefault(key, value)


def parse_metadata(items: list[str], description: str | None) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if description:
        metadata["description"] = description
    for item in items:
        if "=" not in item:
            raise ValueError(f"metadata item must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        metadata[key] = value
    return metadata


def validate_batch_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix != ".jsonl":
        raise ValueError(f"Batch input must be .jsonl, got {path}")
    rows = 0
    custom_ids: set[str] = set()
    models: set[str] = set()
    urls: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            rows += 1
            custom_id = str(row.get("custom_id", ""))
            if not custom_id:
                raise ValueError(f"{path}:{line_number}: missing custom_id")
            custom_ids.add(custom_id)
            urls.add(str(row.get("url", "")))
            body = row.get("body") or {}
            if body.get("model"):
                models.add(str(body["model"]))
    if rows != len(custom_ids):
        raise ValueError(f"{path}: duplicate custom_id values found")
    size_bytes = path.stat().st_size
    if rows > 50_000:
        raise ValueError(f"{path}: {rows} rows exceeds the 50,000 request per-batch limit")
    if size_bytes > 200 * 1024 * 1024:
        raise ValueError(f"{path}: {size_bytes} bytes exceeds the 200 MB Batch file limit")
    return {
        "rows": rows,
        "size_bytes": size_bytes,
        "models": sorted(models),
        "urls": sorted(urls),
    }


def as_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return obj
    return json.loads(obj.model_dump_json())


def write_file_content(client: OpenAI, file_id: str, path: Path) -> None:
    content = client.files.content(file_id)
    if hasattr(content, "write_to_file"):
        content.write_to_file(path)
        return
    data = getattr(content, "content", None)
    if data is None and hasattr(content, "read"):
        data = content.read()
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    elif isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(str(content), encoding="utf-8")


def write_manifest(output_dir: Path, data: dict[str, Any]) -> None:
    (output_dir / "batch_run_manifest.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    load_env_file(Path(args.env_file))

    batch_file = Path(args.batch_file)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite to reuse it")
    output_dir.mkdir(parents=True, exist_ok=True)

    file_stats = validate_batch_file(batch_file)
    if args.validate_only:
        print(json.dumps(file_stats, indent=2))
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in the environment or .env; do not put API keys in command arguments.")

    metadata = parse_metadata(args.metadata, args.description)
    client = OpenAI()

    uploaded = client.files.create(file=batch_file.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=args.endpoint,
        completion_window=args.completion_window,
        metadata=metadata or None,
    )
    manifest: dict[str, Any] = {
        "batch_file": str(batch_file),
        "endpoint": args.endpoint,
        "completion_window": args.completion_window,
        "output_dir": str(output_dir),
        "input_file_id": uploaded.id,
        "batch_id": batch.id,
        "file_stats": file_stats,
        "metadata": metadata,
        "created_batch": as_dict(batch),
    }
    write_manifest(output_dir, manifest)
    print(f"Created batch {batch.id} from {batch_file}")

    if args.no_wait:
        print(f"Manifest: {output_dir / 'batch_run_manifest.json'}")
        return

    started = time.time()
    while True:
        batch = client.batches.retrieve(batch.id)
        batch_data = as_dict(batch)
        manifest["latest_batch"] = batch_data
        write_manifest(output_dir, manifest)
        status = batch_data.get("status")
        counts = batch_data.get("request_counts") or {}
        print(
            f"{datetime.now().isoformat(timespec='seconds')} status={status} "
            f"completed={counts.get('completed')} failed={counts.get('failed')} total={counts.get('total')}",
            flush=True,
        )
        if status in TERMINAL_STATUSES:
            break
        if args.timeout_seconds and time.time() - started > args.timeout_seconds:
            raise TimeoutError(f"Batch {batch.id} did not finish within {args.timeout_seconds}s")
        time.sleep(max(1, args.poll_interval_seconds))

    batch_data = as_dict(batch)
    output_file_id = batch_data.get("output_file_id")
    error_file_id = batch_data.get("error_file_id")
    if output_file_id:
        output_path = output_dir / f"{batch.id}_output.jsonl"
        write_file_content(client, output_file_id, output_path)
        manifest["output_path"] = str(output_path)
        print(f"Downloaded output to {output_path}")
    if error_file_id:
        error_path = output_dir / f"{batch.id}_error.jsonl"
        write_file_content(client, error_file_id, error_path)
        manifest["error_path"] = str(error_path)
        print(f"Downloaded errors to {error_path}")
    write_manifest(output_dir, manifest)


if __name__ == "__main__":
    main()
