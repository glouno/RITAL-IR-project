#!/usr/bin/env python3
"""Build slide-ready retrieval/context-composition artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("artifacts/openai_full_hirag_2026-05")
CORE7 = ROOT / "core7_eval"
OUT = CORE7 / "summary" / "retrieval_thesis"
VARIANTS = ["naive", "hi", "hi_weighted", "hi_minmax_budgeted", "hi_mcts"]
LABELS = {
    "naive": "Naive",
    "hi": "HiRAG",
    "hi_weighted": "Weighted",
    "hi_minmax_budgeted": "Minmax budgeted",
    "hi_mcts": "MCTS",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def section_tokens(text: str) -> dict[str, int]:
    # Use a light approximation for slide metrics; previous deep analysis used
    # tiktoken and showed the same conclusion.
    sections = {
        "Backgrounds": 0,
        "Reasoning Path": 0,
        "Detail Entity Information": 0,
        "Source Documents": 0,
    }
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        for section in sections:
            if stripped.startswith(f"-----{section}-----") or stripped == section:
                current = section
                break
        else:
            if current:
                sections[current] += len(stripped.split())
    return sections


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def read_win_rates() -> dict[tuple[str, str], float]:
    out = {}
    for dataset in ["agriculture", "mix"]:
        summary = CORE7 / dataset / "judge_summary.md"
        cur = None
        for line in summary.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                cur = line[3:].strip()
            elif cur and "win rate:" in line:
                out[(dataset, cur)] = float(line.rsplit(" ", 1)[-1])
    return out


def build_metrics() -> list[dict[str, object]]:
    win_rates = read_win_rates()
    rows = []
    for dataset in ["agriculture", "mix"]:
        traces = load_jsonl(CORE7 / dataset / "retrieval_traces.jsonl")
        by_variant = {variant: [row for row in traces if row["variant"] == variant] for variant in VARIANTS}
        for variant, values in by_variant.items():
            source_ratios = []
            graph_ratios = []
            source_token_values = []
            graph_token_values = []
            for row in values:
                context_tokens = row.get("context_tokens") or 0
                if variant == "naive":
                    source_ratios.append(1.0)
                    graph_ratios.append(0.0)
                    source_token_values.append(context_tokens)
                    graph_token_values.append(0)
                    continue
                tokens = section_tokens(row.get("context", ""))
                total = sum(tokens.values())
                if total:
                    source_ratio = tokens["Source Documents"] / total
                    graph_ratio = (
                        (tokens["Backgrounds"] + tokens["Reasoning Path"] + tokens["Detail Entity Information"]) / total
                    )
                    source_ratios.append(source_ratio)
                    graph_ratios.append(graph_ratio)
                    source_token_values.append(context_tokens * source_ratio)
                    graph_token_values.append(context_tokens * graph_ratio)
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "label": LABELS[variant],
                    "mean_context_tokens": mean([row.get("context_tokens") or 0 for row in values]),
                    "mean_retrieval_time_seconds": mean([row.get("latency_seconds") or 0 for row in values]),
                    "mean_local_entities": mean([(row.get("retrieval") or {}).get("local_entity_count") or 0 for row in values]),
                    "mean_communities": mean([(row.get("retrieval") or {}).get("community_count") or 0 for row in values]),
                    "mean_bridge_edges": mean([(row.get("retrieval") or {}).get("bridge_edge_count") or 0 for row in values]),
                    "mean_bridge_path_edges": mean([(row.get("retrieval") or {}).get("bridge_path_edges") or 0 for row in values]),
                    "mean_mcts_iterations": mean([(row.get("retrieval") or {}).get("bridge_path_decisions", [{}])[0].get("iterations", 0) if (row.get("retrieval") or {}).get("bridge_path_decisions") else 0 for row in values]),
                    "source_evidence_share": mean(source_ratios),
                    "graph_abstraction_share": mean(graph_ratios),
                    "source_evidence_tokens": mean(source_token_values),
                    "graph_abstraction_tokens": mean(graph_token_values),
                    "raw_llm_judge_win_rate_vs_hi": None if variant == "hi" else win_rates.get((dataset, variant)),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_source_share(rows: list[dict[str, object]]) -> None:
    for dataset in ["agriculture", "mix"]:
        selected = [row for row in rows if row["dataset"] == dataset and row["variant"] in ["naive", "hi", "hi_minmax_budgeted", "hi_mcts"]]
        labels = [row["label"] for row in selected]
        source = [float(row["source_evidence_tokens"]) for row in selected]
        graph = [float(row["graph_abstraction_tokens"]) for row in selected]
        source_share = [float(row["source_evidence_share"]) * 100 for row in selected]
        total_tokens = [float(row["mean_context_tokens"]) for row in selected]
        x = range(len(selected))
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        ax.bar(x, source, label="Direct source evidence", color="#4169a8")
        ax.bar(x, graph, bottom=source, label="Graph abstractions", color="#d67c35")
        ax.set_ylim(0, max(total_tokens) * 1.18)
        ax.set_ylabel("Mean final context size (tokens)")
        ax.set_title(f"Context composition: {dataset.capitalize()}")
        ax.set_xticks(list(x), labels, rotation=12)
        ax.legend(frameon=False, loc="upper right")
        for i, value in enumerate(source):
            if value:
                ax.text(i, value / 2, f"{source_share[i]:.0f}%", ha="center", va="center", color="white", fontweight="bold")
            ax.text(i, total_tokens[i] + max(total_tokens) * 0.03, f"{total_tokens[i]:.0f}", ha="center", va="bottom", fontsize=9)
        savefig(OUT / f"context_composition_{dataset}.png")


def plot_context_vs_win(rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for row in rows:
        if row["variant"] == "hi":
            continue
        x = float(row["source_evidence_share"]) * 100
        y = float(row["raw_llm_judge_win_rate_vs_hi"]) * 100
        marker = "o" if row["dataset"] == "agriculture" else "s"
        ax.scatter(x, y, s=120, marker=marker, label=f"{row['label']} ({row['dataset']})")
        ax.text(x + 1.0, y + 0.5, f"{row['label']}\\n{row['dataset']}", fontsize=8)
    ax.axhline(50, color="#555555", linestyle="--", linewidth=1)
    ax.set_xlabel("Direct source evidence share in context (%)")
    ax.set_ylabel("LLM judge win rate vs HiRAG (%)")
    ax.set_title("More source evidence often predicts stronger QA preference")
    ax.set_xlim(0, 105)
    ax.set_ylim(30, 95)
    savefig(OUT / "source_evidence_share_vs_win_rate.png")


def plot_retrieval_structure(rows: list[dict[str, object]]) -> None:
    selected = [row for row in rows if row["dataset"] == "agriculture" and row["variant"] in ["hi", "hi_weighted", "hi_minmax_budgeted", "hi_mcts"]]
    labels = [row["label"] for row in selected]
    edges = [float(row["mean_bridge_edges"]) for row in selected]
    times = [float(row["mean_retrieval_time_seconds"]) for row in selected]
    fig, ax1 = plt.subplots(figsize=(8.4, 4.8))
    x = range(len(selected))
    ax1.bar(x, edges, color="#4169a8", label="Bridge edges")
    ax1.set_ylabel("Mean bridge edges")
    ax1.set_xticks(list(x), labels, rotation=12)
    ax2 = ax1.twinx()
    ax2.scatter(list(x), times, color="#c7473d", marker="o", s=80, label="Retrieval time")
    ax2.set_ylabel("Mean retrieval time (s)")
    ax1.set_title("Graph variants change retrieval structure and cost")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left")
    savefig(OUT / "graph_retrieval_structure_agriculture.png")


def write_summary(rows: list[dict[str, object]]) -> None:
    selected = [row for row in rows if row["variant"] in ["naive", "hi", "hi_minmax_budgeted", "hi_mcts"]]
    lines = [
        "# Retrieval Thesis Slide Artifacts",
        "",
        "Purpose: provide slide-ready statistics for the thesis that better graph traversal alone is not enough; final context composition matters.",
        "",
        "## Key Stats",
        "",
        "| Dataset | Variant | Context tokens | Source evidence share | Graph abstraction share | Bridge edges | Retrieval time | LLM win rate vs HiRAG |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        win = "-" if row["raw_llm_judge_win_rate_vs_hi"] is None else f"{float(row['raw_llm_judge_win_rate_vs_hi']) * 100:.1f}%"
        lines.append(
            f"| {row['dataset']} | {row['label']} | {float(row['mean_context_tokens']):.0f} | "
            f"{float(row['source_evidence_share']) * 100:.1f}% | {float(row['graph_abstraction_share']) * 100:.1f}% | "
            f"{float(row['mean_bridge_edges']):.1f} | {float(row['mean_retrieval_time_seconds']):.2f}s | {win} |"
        )
    lines.extend(
        [
            "",
            "## How To Use These Slides",
            "",
            "1. Show `graph_retrieval_structure_agriculture.png` to prove the variants really change graph traversal.",
            "2. Show `context_composition_mix.png` or `context_composition_agriculture.png` to show why this is not sufficient: graph modes allocate much less final context to direct source evidence. These plots use absolute context size in tokens, not normalized percentages, so Naive's larger prompt is visible.",
            "3. Show `source_evidence_share_vs_win_rate.png` to connect evidence share with end-to-end judge preference.",
            "",
            "Suggested speaking line:",
            "",
            "> Our variants do change graph traversal: they select different bridge paths and pay different retrieval costs. But the final prompt still contains much less direct source evidence than naive RAG. For QA tasks, this evidence composition matters more than graph structure alone.",
            "",
            "Caveat: source evidence share is computed from final context sections, so it is a prompt-composition metric, not a ground-truth relevance metric.",
        ]
    )
    (OUT / "RETRIEVAL_THESIS_SLIDE_NOTES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_metrics()
    write_csv(OUT / "retrieval_thesis_metrics.csv", rows)
    plot_source_share(rows)
    plot_context_vs_win(rows)
    plot_retrieval_structure(rows)
    write_summary(rows)
    print(f"Wrote retrieval thesis artifacts to {OUT}")


if __name__ == "__main__":
    main()
