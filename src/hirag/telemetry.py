from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


def new_telemetry_session(label: str | None = None) -> dict[str, Any]:
    return {
        "label": label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "calls": [],
    }


def record_llm_call(telemetry: dict[str, Any] | None, entry: dict[str, Any]) -> None:
    if telemetry is None:
        return
    telemetry.setdefault("calls", []).append(entry)


def summarize_telemetry(telemetry: dict[str, Any] | None) -> dict[str, Any]:
    if telemetry is None:
        return {"summary": {}, "calls": []}

    calls = list(telemetry.get("calls", []))
    stages: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "call_count": 0,
            "cache_hits": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_seconds_total": 0.0,
            "latency_seconds_avg": 0.0,
        }
    )

    total_cache_hits = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_latency_seconds = 0.0

    for call in calls:
        stage = call.get("stage", "unknown")
        stage_summary = stages[stage]
        stage_summary["call_count"] += 1
        if call.get("cache_hit"):
            stage_summary["cache_hits"] += 1
            total_cache_hits += 1
        prompt_tokens = int(call.get("prompt_tokens", 0) or 0)
        completion_tokens = int(call.get("completion_tokens", 0) or 0)
        total_call_tokens = int(call.get("total_tokens", 0) or 0)
        latency_seconds = float(call.get("latency_seconds", 0.0) or 0.0)

        stage_summary["prompt_tokens"] += prompt_tokens
        stage_summary["completion_tokens"] += completion_tokens
        stage_summary["total_tokens"] += total_call_tokens
        stage_summary["latency_seconds_total"] += latency_seconds

        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_tokens += total_call_tokens
        total_latency_seconds += latency_seconds

    for stage_summary in stages.values():
        if stage_summary["call_count"]:
            stage_summary["latency_seconds_avg"] = (
                stage_summary["latency_seconds_total"] / stage_summary["call_count"]
            )

    summary = {
        "label": telemetry.get("label"),
        "created_at": telemetry.get("created_at"),
        "call_count": len(calls),
        "cache_hits": total_cache_hits,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "latency_seconds_total": total_latency_seconds,
        "latency_seconds_avg": (total_latency_seconds / len(calls)) if calls else 0.0,
        "stages": dict(stages),
    }
    return {"summary": summary, "calls": calls}
