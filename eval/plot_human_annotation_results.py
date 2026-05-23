#!/usr/bin/env python3
"""Plot human annotation vs LLM judge summaries for Core 7."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


VARIANT_LABELS = {
    "naive": "Naive",
    "hi_minmax_budgeted": "Minmax budgeted",
    "hi_mcts": "MCTS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create human annotation result figures.")
    parser.add_argument(
        "--input-dir",
        default="artifacts/openai_full_hirag_2026-05/core7_eval/human_annotation_results",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def maybe_float(value: str) -> float | None:
    if value in {"", None}:
        return None
    return float(value)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_win_rates(summary_rows: list[dict[str, str]], output_dir: Path) -> None:
    rows = [row for row in summary_rows if row["variant"] != "hi"]
    labels = [row["label"] for row in rows]
    human = [float(row["human_win_rate_vs_hi_half_credit"]) * 100 for row in rows]
    llm = [float(row["llm_strict_win_rate_vs_hi_on_same_items"]) * 100 for row in rows]
    comparable = [row["n_human_llm_strict_comparable"] for row in rows]

    x = list(range(len(rows)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar([i - width / 2 for i in x], human, width, label="Human annotators", color="#4169a8")
    ax.bar([i + width / 2 for i in x], llm, width, label="LLM judge (strict)", color="#d67c35")
    ax.axhline(50, color="#555555", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_ylabel("Win rate vs classic HiRAG (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x, labels)
    ax.set_title("Human vs LLM-as-judge preferences")
    ax.legend(frameon=False)
    for i, (h, l, n) in enumerate(zip(human, llm, comparable)):
        ax.text(i - width / 2, h + 2, f"{h:.1f}%", ha="center", fontsize=9)
        ax.text(i + width / 2, l + 2, f"{l:.1f}%\n(n={n})", ha="center", fontsize=9)
    savefig(output_dir / "human_vs_llm_win_rates.png")


def plot_stacked_decisions(pair_rows: list[dict[str, str]], output_dir: Path) -> None:
    labels = [VARIANT_LABELS[row["variant_vs_hi"]] for row in pair_rows]
    x = list(range(len(pair_rows)))

    human_wins = [int(row["human_other_wins"]) for row in pair_rows]
    human_losses = [int(row["human_hi_wins"]) for row in pair_rows]
    human_ties = [int(row["human_ties"]) for row in pair_rows]

    llm_wins = [int(row["llm_strict_other_wins"]) for row in pair_rows]
    llm_losses = [int(row["llm_strict_hi_wins"]) for row in pair_rows]
    llm_ties = [int(row["llm_strict_ties"]) for row in pair_rows]
    llm_disagree = [int(row["llm_strict_disagree"]) for row in pair_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    colors = {
        "variant": "#4169a8",
        "hirag": "#c7473d",
        "tie": "#9a9a9a",
        "disagree": "#f0c35b",
    }

    bottoms = [0] * len(x)
    axes[0].bar(x, human_wins, label="Variant wins", color=colors["variant"])
    bottoms = human_wins[:]
    axes[0].bar(x, human_losses, bottom=bottoms, label="HiRAG wins", color=colors["hirag"])
    bottoms = [a + b for a, b in zip(bottoms, human_losses)]
    axes[0].bar(x, human_ties, bottom=bottoms, label="Tie", color=colors["tie"])
    axes[0].set_title("Human decisions")
    axes[0].set_ylabel("Annotated rows")
    annotate_stacked_bar(axes[0], x, [human_wins, human_losses, human_ties], [20] * len(x))

    bottoms = [0] * len(x)
    axes[1].bar(x, llm_wins, label="Variant wins", color=colors["variant"])
    bottoms = llm_wins[:]
    axes[1].bar(x, llm_losses, bottom=bottoms, label="HiRAG wins", color=colors["hirag"])
    bottoms = [a + b for a, b in zip(bottoms, llm_losses)]
    axes[1].bar(x, llm_ties, bottom=bottoms, label="Tie", color=colors["tie"])
    bottoms = [a + b for a, b in zip(bottoms, llm_ties)]
    axes[1].bar(x, llm_disagree, bottom=bottoms, label="Swapped-order disagree", color=colors["disagree"])
    axes[1].set_title("LLM judge strict decisions")
    annotate_stacked_bar(axes[1], x, [llm_wins, llm_losses, llm_ties, llm_disagree], [20] * len(x))

    for ax in axes:
        ax.set_xticks(x, labels, rotation=15, ha="right")
        ax.set_ylim(0, 21)
    fig.suptitle("Decision breakdown on the same human-annotated pairs", y=1.02)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
        fontsize=9,
    )
    plt.tight_layout(rect=(0, 0.08, 1, 0.96))
    plt.savefig(output_dir / "human_vs_llm_decision_breakdown.png", dpi=220, bbox_inches="tight")
    plt.close()


def annotate_stacked_bar(ax: plt.Axes, x_values: list[int], segments: list[list[int]], totals: list[int]) -> None:
    bottoms = [0] * len(x_values)
    for segment in segments:
        for index, value in enumerate(segment):
            if value <= 0:
                continue
            y = bottoms[index] + value / 2
            percent = value / totals[index] * 100 if totals[index] else 0
            label = f"{value}\n{percent:.0f}%"
            # Keep tiny segments readable by only printing the count.
            if value <= 2:
                label = str(value)
            ax.text(
                x_values[index],
                y,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value >= 4 else "#222222",
                fontweight="bold" if value >= 4 else "normal",
            )
        bottoms = [bottom + value for bottom, value in zip(bottoms, segment)]


def plot_span_density_vs_gain(summary_rows: list[dict[str, str]], output_dir: Path) -> None:
    rows = summary_rows
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for row in rows:
        x = float(row["mean_useful_span_density"]) * 100
        gain = maybe_float(row["human_preference_gain_vs_hi"])
        y = 0.0 if gain is None else gain
        label = row["label"]
        color = "#777777" if row["variant"] == "hi" else "#4169a8"
        ax.scatter(x, y, s=130, color=color, edgecolor="white", linewidth=1.4)
        ax.text(x + 0.35, y + 0.015, label, fontsize=10)
    ax.axhline(0, color="#555555", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xlabel("Useful span density (%)")
    ax.set_ylabel("Mean preference gain vs classic HiRAG")
    ax.set_title("Evidence compactness vs human preference")
    ax.set_xlim(26, 34)
    ax.set_ylim(-0.18, 0.38)
    savefig(output_dir / "span_density_vs_preference_gain.png")


def plot_answer_question_scores(summary_rows: list[dict[str, str]], output_dir: Path) -> None:
    labels = [row["label"] for row in summary_rows]
    scores = [float(row["mean_answers_question_score"]) * 100 for row in summary_rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    bars = ax.bar(labels, scores, color=["#777777", "#4169a8", "#4169a8", "#4169a8"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Answers-the-question score (%)")
    ax.set_title("Human answer-question ratings")
    ax.tick_params(axis="x", rotation=12)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 1.2, f"{score:.1f}%", ha="center", fontsize=9)
    savefig(output_dir / "human_answer_question_scores.png")


def write_slide_table(summary_rows: list[dict[str, str]], output_dir: Path) -> None:
    rows = []
    for row in summary_rows:
        win = "-"
        gain = "-"
        if row["variant"] != "hi":
            win = f"{float(row['human_win_rate_vs_hi_half_credit']) * 100:.1f}%"
            gain = f"{float(row['human_preference_gain_vs_hi']):+.2f}"
        rows.append(
            {
                "Variante": row["label"],
                "Win rate": win,
                "Densite du span utile": f"{float(row['mean_useful_span_density']) * 100:.1f}%",
                "Gain moyen": gain,
                "LLM win rate strict": "-"
                if row["variant"] == "hi"
                else f"{float(row['llm_strict_win_rate_vs_hi_on_same_items']) * 100:.1f}%",
            }
        )
    write_csv(output_dir / "slide_table_with_llm.csv", rows)

    lines = [
        "# Slide Table With Human And LLM Judge",
        "",
        "| Variante | Win rate humain | Densite du span utile | Gain moyen humain | LLM win rate strict |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['Variante']} | {row['Win rate']} | {row['Densite du span utile']} | "
            f"{row['Gain moyen']} | {row['LLM win rate strict']} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- Human win rate counts ties as 0.5 wins against classic HiRAG.",
            "- LLM win rate strict uses only swapped-order-consistent LLM judgments on the same human-annotated rows.",
            "- Useful span density is useful highlighted span length divided by answer length.",
        ]
    )
    (output_dir / "slide_table_with_llm.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures_readme(output_dir: Path) -> None:
    lines = [
        "# Human Annotation Figures",
        "",
        "Generated from `human_annotation_variant_summary.csv` and `human_vs_llm_pair_summary.csv`.",
        "",
        "Files:",
        "",
        "- `human_vs_llm_win_rates.png`: grouped bars comparing human win rate and strict LLM win rate vs classic HiRAG.",
        "- `human_vs_llm_decision_breakdown.png`: stacked decisions showing variant wins, HiRAG wins, ties, and LLM swapped-order disagreements.",
        "- `span_density_vs_preference_gain.png`: useful span density against human preference gain.",
        "- `human_answer_question_scores.png`: human yes/partial/no answer-question score per variant.",
        "- `slide_table_with_llm.md` and `.csv`: slide-ready table including human and LLM win rates.",
        "",
        "Recommended slide narrative:",
        "",
        "- Naive is the only clearly preferred variant by both humans and the LLM judge.",
        "- Minmax budgeted and MCTS are close to HiRAG under human annotation.",
        "- LLM judge has substantial swapped-order disagreement on close comparisons, so use it as a high-throughput proxy rather than an oracle.",
        "",
        "Swapped-order disagreement means the same answer pair was judged twice, once as A/B and once as B/A, and the LLM did not pick the same underlying variant after mapping the labels back. This is a judge-stability warning, not a separate model variant.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv(input_dir / "human_annotation_variant_summary.csv")
    pair_rows = read_csv(input_dir / "human_vs_llm_pair_summary.csv")

    plot_win_rates(summary_rows, output_dir)
    plot_stacked_decisions(pair_rows, output_dir)
    plot_span_density_vs_gain(summary_rows, output_dir)
    plot_answer_question_scores(summary_rows, output_dir)
    write_slide_table(summary_rows, output_dir)
    write_figures_readme(output_dir)
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
