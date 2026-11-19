"""Dependency-free SVG summaries for V12 next-move results."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ACTIONS = ("propose", "accept", "reject", "inquire", "insist", "other")


def _bar_svg(title, labels, values, ylabel, path, *, width=820, height=470):
    left, right, top, bottom = 90, 25, 55, 90
    plot_height = height - top - bottom
    maximum = max(values) * 1.15
    group = (width - left - right) / len(labels)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>',
    ]
    for tick in range(6):
        value = maximum * tick / 5
        y = top + plot_height * (1 - value / maximum)
        parts.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-8}" y="{y+4}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.3f}</text>')
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + group * index + group * .18
        y = top + plot_height * (1 - value / maximum)
        parts.append(f'<rect x="{x}" y="{y}" width="{group*.64}" height="{top+plot_height-y}" fill="#2563eb"/>')
        parts.append(f'<text x="{x+group*.32}" y="{y-6}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:.3f}</text>')
        parts.append(f'<text x="{x+group*.32}" y="{height-60}" text-anchor="middle" transform="rotate(-25 {x+group*.32} {height-60})" font-family="sans-serif" font-size="11">{label}</text>')
    parts.append(f'<text x="20" y="{top+plot_height/2}" transform="rotate(-90 20 {top+plot_height/2})" text-anchor="middle" font-family="sans-serif" font-size="12">{ylabel}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts_v12"
    human = json.loads((output / "strategic_move_results.json").read_text())
    methods = ("prior", "history_logit", "strategic", "hybrid", "game_logit")
    labels = ("Conversation prior", "History logit", "Latent SERF", "SERF hybrid", "Nash geometry")
    _bar_svg(
        "Held-out human next-move log loss",
        labels,
        [human["all_move_metrics"][method]["nll"] for method in methods],
        "NLL (lower is better)",
        output / "human_next_move_nll.svg",
    )
    rows = list(csv.DictReader((output / "all_move_predictions.csv").open()))
    gains = []
    for action in ACTIONS:
        selected = [row for row in rows if row["outcome"] == action]
        history = math.fsum(-math.log(float(row[f"history_{action}"])) for row in selected) / len(selected)
        game = math.fsum(-math.log(float(row[f"nash_geometry_{action}"])) for row in selected) / len(selected)
        gains.append(history - game)
    # Positive means the game geometry helps; use a centered plot for negative values.
    width, height, left, right, top, bottom = 820, 450, 100, 25, 55, 75
    bound = max(abs(value) for value in gains) * 1.25
    plot_height = height - top - bottom
    zero = top + plot_height / 2
    group = (width - left - right) / len(ACTIONS)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">Where payoff geometry improves the forecast</text>',
        f'<line x1="{left}" y1="{zero}" x2="{width-right}" y2="{zero}" stroke="#111827"/>',
    ]
    for index, (action, value) in enumerate(zip(ACTIONS, gains)):
        x = left + group * index + group * .18
        magnitude = abs(value) / bound * (plot_height / 2)
        y = zero - magnitude if value >= 0 else zero
        color = "#16a34a" if value >= 0 else "#dc2626"
        parts.append(f'<rect x="{x}" y="{y}" width="{group*.64}" height="{magnitude}" fill="{color}"/>')
        label_y = y - 6 if value >= 0 else y + magnitude + 16
        parts.append(f'<text x="{x+group*.32}" y="{label_y}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:+.3f}</text>')
        parts.append(f'<text x="{x+group*.32}" y="{height-38}" text-anchor="middle" font-family="sans-serif" font-size="11">{action}</text>')
    parts.append(f'<text x="20" y="{top+plot_height/2}" transform="rotate(-90 20 {top+plot_height/2})" text-anchor="middle" font-family="sans-serif" font-size="12">History NLL minus Nash-geometry NLL</text>')
    parts.append('</svg>')
    (output / "per_action_game_gain.svg").write_text("\n".join(parts) + "\n")

    synthetic = json.loads((output / "explicit_game_suite.json").read_text())
    _bar_svg(
        "Explicit-payoff synthetic NLL improvement",
        [record["family"].replace("_", " ") for record in synthetic["records"]],
        [record["nll_improvement"] for record in synthetic["records"]],
        "Base NLL minus model NLL",
        output / "synthetic_game_improvement.svg",
        width=940,
    )


if __name__ == "__main__":
    main()
