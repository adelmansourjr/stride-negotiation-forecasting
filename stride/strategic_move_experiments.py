"""Held-out negotiation benchmark for game-theoretic next-move forecasting."""

from __future__ import annotations

from collections import Counter
import csv
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Callable, Sequence

from .strategic_move import (
    ACTIONS,
    ConversationalPrior,
    FailClosedMoveWrapper,
    MoveForecast,
    NegotiationDialogue,
    SoftmaxMoveBaseline,
    StrategicMoveForecaster,
    aps_score,
    dialogue_states,
    load_negotiation_dialogues,
    selected_move_index,
)


Probability = tuple[float, ...]


def _nll(probabilities: Sequence[float], outcome: int) -> float:
    return -math.log(max(1e-15, probabilities[outcome]))


def _brier(probabilities: Sequence[float], outcome: int) -> float:
    return math.fsum((value - float(index == outcome)) ** 2 for index, value in enumerate(probabilities))


def _metrics(rows: Sequence[dict[str, object]]) -> dict[str, float]:
    losses = [_nll(row["probabilities"], row["outcome"]) for row in rows]
    briers = [_brier(row["probabilities"], row["outcome"]) for row in rows]
    accuracy = [
        max(range(len(ACTIONS)), key=lambda index: (row["probabilities"][index], -index)) == row["outcome"]
        for row in rows
    ]
    bins = [[] for _ in range(10)]
    for row in rows:
        probability = max(row["probabilities"])
        prediction = max(range(len(ACTIONS)), key=lambda index: (row["probabilities"][index], -index))
        bins[min(9, int(probability * 10))].append((probability, float(prediction == row["outcome"])))
    ece = math.fsum(
        len(bucket) / len(rows) * abs(
            math.fsum(value[0] for value in bucket) / len(bucket)
            - math.fsum(value[1] for value in bucket) / len(bucket)
        )
        for bucket in bins if bucket
    )
    return {
        "n": len(rows),
        "nll": math.fsum(losses) / len(losses),
        "brier": math.fsum(briers) / len(briers),
        "accuracy": math.fsum(accuracy) / len(accuracy),
        "ece_10": ece,
    }


def _predict_dialogue(
    dialogue: NegotiationDialogue,
    prior: ConversationalPrior,
    strategic: StrategicMoveForecaster,
    history_logit: SoftmaxMoveBaseline,
    game_logit: SoftmaxMoveBaseline,
    strategic_weight: float,
) -> list[dict[str, object]]:
    sequence = strategic.start_dialogue()
    rows = []
    for state, move in zip(dialogue_states(dialogue), dialogue.moves):
        strategic_forecast = sequence.predict(state)
        prior_probability = prior.probabilities(state)
        history_probability = history_logit.predict(state)
        game_probability = game_logit.predict(state)
        hybrid = tuple(
            strategic_weight * strategic_value + (1.0 - strategic_weight) * game_value
            for strategic_value, game_value in zip(strategic_forecast.probabilities, game_probability)
        )
        rows.append({
            "dialogue_id": dialogue.identifier,
            "turn": state.turn,
            "state": state,
            "outcome": move.action,
            "prior": prior_probability,
            "strategic": strategic_forecast.probabilities,
            "history_logit": history_probability,
            "game_logit": game_probability,
            "hybrid": hybrid,
            "effective_types": strategic_forecast.strategic_type_effective_count,
            "type_disagreement": strategic_forecast.maximum_type_disagreement,
        })
        sequence.update(move.action)
    return rows


def _predict_corpus(
    dialogues: Sequence[NegotiationDialogue],
    prior: ConversationalPrior,
    strategic: StrategicMoveForecaster,
    history_logit: SoftmaxMoveBaseline,
    game_logit: SoftmaxMoveBaseline,
    strategic_weight: float,
) -> list[dict[str, object]]:
    return [
        row
        for dialogue in dialogues
        for row in _predict_dialogue(
            dialogue, prior, strategic, history_logit, game_logit, strategic_weight
        )
    ]


def _selected_rows(rows: Sequence[dict[str, object]], dialogues: Sequence[NegotiationDialogue]) -> list[dict[str, object]]:
    by_dialogue: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_dialogue.setdefault(row["dialogue_id"], []).append(row)
    selected = []
    for dialogue in dialogues:
        ordered = sorted(by_dialogue[dialogue.identifier], key=lambda row: row["turn"])
        selected.append(ordered[selected_move_index(dialogue)])
    return selected


def _first_rows(rows: Sequence[dict[str, object]], dialogues: Sequence[NegotiationDialogue]) -> list[dict[str, object]]:
    """One exchangeable, pre-outcome target per independent dialogue."""

    first = {}
    for row in rows:
        first.setdefault(row["dialogue_id"], row)
    return [first[dialogue.identifier] for dialogue in dialogues]


def _as_method_rows(rows: Sequence[dict[str, object]], method: str) -> list[dict[str, object]]:
    return [{"probabilities": row[method], "outcome": row["outcome"]} for row in rows]


def _choose_hybrid_weight(rows: Sequence[dict[str, object]]) -> tuple[float, list[dict[str, float]]]:
    landscape = []
    for step in range(51):
        weight = step / 50.0
        losses = []
        for row in rows:
            probabilities = tuple(
                weight * strategic + (1.0 - weight) * logistic
                for strategic, logistic in zip(row["strategic"], row["game_logit"])
            )
            losses.append(_nll(probabilities, row["outcome"]))
        landscape.append({"strategic_weight": weight, "nll": math.fsum(losses) / len(losses)})
    chosen = min(landscape, key=lambda row: (row["nll"], row["strategic_weight"]))["strategic_weight"]
    return chosen, landscape


def _paired_bootstrap(
    candidate: Sequence[dict[str, object]],
    baseline: Sequence[dict[str, object]],
    *,
    samples: int = 10_000,
    seed: int = 72_011,
) -> dict[str, float]:
    differences = tuple(
        _nll(candidate_row["probabilities"], candidate_row["outcome"])
        - _nll(baseline_row["probabilities"], baseline_row["outcome"])
        for candidate_row, baseline_row in zip(candidate, baseline)
    )
    rng = random.Random(seed)
    draws = sorted(
        math.fsum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences)
        for _ in range(samples)
    )
    return {
        "candidate_minus_baseline_mean_nll": math.fsum(differences) / len(differences),
        "bootstrap_95_low": draws[math.floor(.025 * samples)],
        "bootstrap_95_high": draws[min(samples - 1, math.ceil(.975 * samples) - 1)],
        "samples": samples,
        "seed": seed,
    }


def _dialogue_cluster_bootstrap(
    candidate: Sequence[dict[str, object]],
    baseline: Sequence[dict[str, object]],
    *,
    samples: int = 10_000,
    seed: int = 74_011,
) -> dict[str, float]:
    """Bootstrap equal-weighted dialogue-average loss differences."""

    grouped: dict[str, list[float]] = {}
    for candidate_row, baseline_row in zip(candidate, baseline):
        if candidate_row["dialogue_id"] != baseline_row["dialogue_id"]:
            raise ValueError("candidate and baseline rows are misaligned")
        difference = (
            _nll(candidate_row["probabilities"], candidate_row["outcome"])
            - _nll(baseline_row["probabilities"], baseline_row["outcome"])
        )
        grouped.setdefault(candidate_row["dialogue_id"], []).append(difference)
    dialogue_differences = tuple(
        math.fsum(values) / len(values) for _, values in sorted(grouped.items())
    )
    rng = random.Random(seed)
    draws = sorted(
        math.fsum(dialogue_differences[rng.randrange(len(dialogue_differences))]
                  for _ in dialogue_differences) / len(dialogue_differences)
        for _ in range(samples)
    )
    return {
        "candidate_minus_baseline_mean_dialogue_average_nll":
            math.fsum(dialogue_differences) / len(dialogue_differences),
        "bootstrap_95_low": draws[math.floor(.025 * samples)],
        "bootstrap_95_high": draws[min(samples - 1, math.ceil(.975 * samples) - 1)],
        "dialogues": len(dialogue_differences),
        "samples": samples,
        "seed": seed,
    }


def _conformal_quantile(rows: Sequence[dict[str, object]], alpha: float, method: str) -> float:
    scores = sorted(aps_score(row[method], row["outcome"]) for row in rows)
    rank = math.ceil((len(scores) + 1) * (1.0 - alpha))
    return math.inf if rank > len(scores) else scores[rank - 1]


def _conformal_evaluation(
    rows: Sequence[dict[str, object]],
    quantile: float,
    prior: ConversationalPrior,
    *,
    maximum_set_size: int = 3,
    minimum_top_probability: float = .30,
    maximum_type_disagreement: float = .35,
    method: str = "hybrid",
    apply_type_disagreement: bool = True,
) -> dict[str, object]:
    statuses = Counter()
    covered = 0
    issued = 0
    issued_correct = 0
    set_sizes = []
    for row in rows:
        probabilities = row[method]
        ranked = sorted(range(len(ACTIONS)), key=lambda index: (-probabilities[index], index))
        prediction_set = tuple(
            action for action in range(len(ACTIONS))
            if aps_score(probabilities, action) <= quantile
        )
        set_sizes.append(len(prediction_set))
        covered += row["outcome"] in prediction_set
        state = row["state"]
        if not prior.in_support(state):
            status = "ABSTAIN_OUT_OF_SUPPORT"
        elif not prediction_set:
            status = "ABSTAIN_EMPTY_SET"
        elif len(prediction_set) > maximum_set_size:
            status = "ABSTAIN_SET_TOO_WIDE"
        elif max(probabilities) < minimum_top_probability:
            status = "ABSTAIN_LOW_POINT_CONFIDENCE"
        elif apply_type_disagreement and row["type_disagreement"] > maximum_type_disagreement:
            status = "ABSTAIN_TYPE_DISAGREEMENT"
        elif sum(abs(value - probabilities[ranked[0]]) <= 1e-15 for value in probabilities) > 1:
            status = "ABSTAIN_POINT_TIE"
        else:
            status = "FORECAST"
            issued += 1
            issued_correct += ranked[0] == row["outcome"]
        statuses[status] += 1
    return {
        "quantile": quantile,
        "marginal_prediction_set_coverage": covered / len(rows),
        "mean_prediction_set_size": math.fsum(set_sizes) / len(set_sizes),
        "issuance_rate": issued / len(rows),
        "selective_accuracy": issued_correct / issued if issued else None,
        "statuses": dict(sorted(statuses.items())),
    }


def _write_landscape(path: Path, landscape: Sequence[dict[str, float]]) -> None:
    width, height, left, top, bottom = 760, 430, 80, 45, 65
    values = [row["nll"] for row in landscape]
    low, high = min(values), max(values)
    padding = max(1e-4, .12 * (high - low))

    def sx(value: float) -> float:
        return left + value * (width - left - 25)

    def sy(value: float) -> float:
        return top + (high + padding - value) / (high + padding - (low - padding)) * (height - top - bottom)

    points = " ".join(f"{sx(row['strategic_weight']):.2f},{sy(row['nll']):.2f}" for row in landscape)
    best = min(landscape, key=lambda row: (row["nll"], row["strategic_weight"]))
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="17">Validation loss across the game-theory mixture</text>',
    ]
    for tick in range(6):
        value = low - padding + tick * (high - low + 2 * padding) / 5
        y = sy(value)
        content.append(f'<line x1="{left}" y1="{y}" x2="{width-25}" y2="{y}" stroke="#e5e7eb"/>')
        content.append(f'<text x="{left-8}" y="{y+4}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.4f}</text>')
    content.append(f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2.5"/>')
    for row in landscape:
        content.append(f'<circle cx="{sx(row["strategic_weight"]):.2f}" cy="{sy(row["nll"]):.2f}" r="3" fill="#2563eb"/>')
    content.append(f'<circle cx="{sx(best["strategic_weight"]):.2f}" cy="{sy(best["nll"]):.2f}" r="7" fill="none" stroke="#16a34a" stroke-width="3"/>')
    content.append(f'<text x="{sx(best["strategic_weight"])+10:.2f}" y="{sy(best["nll"])-8:.2f}" font-family="sans-serif" font-size="12">minimum w={best["strategic_weight"]:.2f}</text>')
    content.append(f'<text x="{(left+width-25)/2}" y="{height-16}" text-anchor="middle" font-family="sans-serif" font-size="12">Strategic mixture weight</text>')
    content.append(f'<text x="18" y="{(top+height-bottom)/2}" text-anchor="middle" transform="rotate(-90 18 {(top+height-bottom)/2})" font-family="sans-serif" font-size="12">Selected-move NLL</text>')
    content.append('</svg>')
    path.write_text("\n".join(content) + "\n")


def run(output: Path) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    data = root / "external" / "end-to-end-negotiator" / "src" / "data" / "negotiate"
    train = load_negotiation_dialogues(data / "train.txt")
    validation_all = load_negotiation_dialogues(data / "val.txt")
    test = load_negotiation_dialogues(data / "test.txt")
    train_ids = {dialogue.identifier for dialogue in train}
    validation = [dialogue for dialogue in validation_all if dialogue.identifier not in train_ids]
    overlaps = {
        "train_validation_removed": len(validation_all) - len(validation),
        "train_test": len(train_ids & {dialogue.identifier for dialogue in test}),
        "validation_test": len({dialogue.identifier for dialogue in validation} & {dialogue.identifier for dialogue in test}),
    }
    started = time.perf_counter()
    prior = ConversationalPrior().fit(train)
    strategic = StrategicMoveForecaster(prior).fit_population(train)
    history_logit = SoftmaxMoveBaseline(include_game=False).fit(train)
    game_logit = SoftmaxMoveBaseline(include_game=True).fit(train)

    validation_rows_zero = _predict_corpus(validation, prior, strategic, history_logit, game_logit, 0.0)
    strategic_weight, landscape = _choose_hybrid_weight(validation_rows_zero)
    validation_rows = _predict_corpus(validation, prior, strategic, history_logit, game_logit, strategic_weight)
    test_rows = _predict_corpus(test, prior, strategic, history_logit, game_logit, strategic_weight)
    validation_first = _first_rows(validation_rows, validation)
    test_first = _first_rows(test_rows, test)

    methods = ("prior", "history_logit", "game_logit", "strategic", "hybrid")
    all_move_metrics = {
        method: _metrics(_as_method_rows(test_rows, method)) for method in methods
    }
    hybrid_comparisons = {
        baseline: _dialogue_cluster_bootstrap(
            [dict(row, probabilities=row["hybrid"]) for row in test_rows],
            [dict(row, probabilities=row[baseline]) for row in test_rows],
            seed=72_011 + index,
        )
        for index, baseline in enumerate(("prior", "history_logit", "game_logit", "strategic"))
    }
    game_comparisons = {
        baseline: _dialogue_cluster_bootstrap(
            [dict(row, probabilities=row["game_logit"]) for row in test_rows],
            [dict(row, probabilities=row[baseline]) for row in test_rows],
            seed=73_011 + index,
        )
        for index, baseline in enumerate(("prior", "history_logit", "strategic"))
    }
    quantile = _conformal_quantile(validation_first, .10, "game_logit")
    conformal = _conformal_evaluation(
        test_first,
        quantile,
        prior,
        method="game_logit",
        apply_type_disagreement=False,
    )
    action_counts = Counter(ACTIONS[row["outcome"]] for row in test_rows)
    offer_coverage = math.fsum(row["state"].last_offer is not None for row in test_rows) / len(test_rows)
    result = {
        "status": "COMPLETE",
        "claim_boundary": "Historical low-stakes coarse dialogue-act prediction; not intent inference or operational conflict forecasting.",
        "data": {
            "train_dialogues": len(train),
            "validation_dialogues": len(validation),
            "test_dialogues": len(test),
            "train_moves": sum(len(dialogue.moves) for dialogue in train),
            "validation_moves": sum(len(dialogue.moves) for dialogue in validation),
            "test_moves": sum(len(dialogue.moves) for dialogue in test),
            "split_overlaps": overlaps,
            "test_explicit_offer_parse_coverage": offer_coverage,
            "test_action_counts": dict(sorted(action_counts.items())),
        },
        "model": {
            "surviving_name": "Nash Geometry Softmax",
            "rejected_name": "Strategic Entropic Response Filter",
            "equation": "pi_theta(a|s) proportional to mu(a|h) exp(beta Q_theta(s,a))",
            "strategic_weight_selected_on_validation": strategic_weight,
            "serf_test_verdict": "REJECT: validation-selected hybrid is worse than Nash Geometry Softmax",
            "population_optimization": strategic.optimization,
            "history_logit_optimization": history_logit.optimization,
            "game_logit_optimization": game_logit.optimization,
        },
        "all_move_metrics": all_move_metrics,
        "nash_geometry_dialogue_cluster_bootstrap": game_comparisons,
        "serf_hybrid_dialogue_cluster_bootstrap": hybrid_comparisons,
        "nash_geometry_fail_closed_first_move": conformal,
        "validation_weight_landscape": landscape,
        "runtime_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "strategic_move_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output / "all_move_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "dialogue_id", "turn", "outcome",
            *[f"prior_{action}" for action in ACTIONS],
            *[f"history_{action}" for action in ACTIONS],
            *[f"nash_geometry_{action}" for action in ACTIONS],
            *[f"strategic_{action}" for action in ACTIONS],
            *[f"hybrid_{action}" for action in ACTIONS],
        ))
        for row in test_rows:
            writer.writerow((
                row["dialogue_id"], row["turn"], ACTIONS[row["outcome"]],
                *row["prior"], *row["history_logit"], *row["game_logit"],
                *row["strategic"], *row["hybrid"],
            ))
    _write_landscape(output / "strategic_weight_landscape.svg", landscape)
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts_v12"
    result = run(output)
    print(json.dumps({
        "status": result["status"],
        "data": result["data"],
        "model": result["model"],
        "all_move_metrics": result["all_move_metrics"],
        "nash_geometry_dialogue_cluster_bootstrap": result["nash_geometry_dialogue_cluster_bootstrap"],
        "nash_geometry_fail_closed_first_move": result["nash_geometry_fail_closed_first_move"],
        "runtime_seconds": result["runtime_seconds"],
        "artifact": str(output / "strategic_move_results.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
