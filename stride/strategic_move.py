"""Game-theoretic next-move forecasting for bounded object negotiations.

The model predicts coarse dialogue acts, not text and not hidden intentions.
It combines an empirical conversational base measure with entropy-regularized
counterfactual continuation values and a Bayesian filter over strategic types.
"""

from __future__ import annotations

from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import random
import re
from typing import Iterable, Sequence


ACTIONS = ("propose", "accept", "reject", "inquire", "insist", "other")
ACTION_INDEX = {name: index for index, name in enumerate(ACTIONS)}
START_ACTION = len(ACTIONS)
ITEMS = ("book", "hat", "ball")
NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "both": 2,
    "three": 3, "four": 4, "five": 5,
}


Vector = tuple[float, ...]


def classify_dialogue_act(text: str) -> str | None:
    """Map an utterance to a frozen, auditable six-action abstraction."""

    value = " ".join(text.lower().strip().split())
    if "<selection>" in value:
        return None
    if re.search(r"\b(no deal|disagree|deal is off|not a deal)\b", value):
        return "reject"
    negative = re.search(r"\b(no|not|can't|cannot|won't|dont|don't)\b", value)
    if (
        re.search(r"\b(deal|agreed|sounds good|works for me|okay|ok|yes|sure|fine|alright)\b", value)
        and not negative
    ):
        return "accept"
    if "?" in value or re.search(
        r"\b(what|would you|could you|can you|how about|do you|will you)\b", value
    ):
        return "inquire"
    if re.search(r"\b(need|must|only|or else|have to|won't|cannot|can't|keeping)\b", value):
        return "insist"
    if re.search(
        r"\b(want|take|give|have|get|keep|split|offer|book|books|hat|hats|ball|balls|rest)\b",
        value,
    ):
        return "propose"
    return "other"


@dataclass(frozen=True, slots=True)
class DialogueMove:
    actor: int  # 0 is YOU in the selected perspective, 1 is THEM.
    text: str
    action: int


@dataclass(frozen=True, slots=True)
class NegotiationDialogue:
    identifier: str
    item_counts: tuple[int, int, int]
    values_you: tuple[int, int, int]
    values_them: tuple[int, int, int]
    moves: tuple[DialogueMove, ...]


@dataclass(frozen=True, slots=True)
class AllocationPoint:
    allocation_self: tuple[int, int, int]
    utility_self: float
    utility_other: float


@dataclass(frozen=True, slots=True)
class BargainingGeometry:
    allocations: tuple[AllocationPoint, ...]
    nash: AllocationPoint
    max_self: AllocationPoint
    conflict: float
    pareto_fraction: float
    integrative_potential: float


@dataclass(frozen=True, slots=True)
class StrategicType:
    beta: float
    discount: float
    social: float
    toughness: float

    @property
    def label(self) -> str:
        return f"b={self.beta:g},d={self.discount:g},s={self.social:g},t={self.toughness:g}"


@dataclass(frozen=True, slots=True)
class MoveState:
    dialogue_id: str
    actor: int
    turn: int
    item_counts: tuple[int, int, int]
    values_self: tuple[int, int, int]
    values_other: tuple[int, int, int]
    history: tuple[int, ...]
    last_utterance: str | None
    geometry: BargainingGeometry
    last_offer: AllocationPoint | None


@dataclass(frozen=True, slots=True)
class MoveForecast:
    probabilities: Vector
    strategic_type_effective_count: float
    maximum_type_disagreement: float


@dataclass(frozen=True, slots=True)
class FailClosedForecast:
    status: str
    probabilities: Vector
    prediction_set: tuple[int, ...]
    point_action: int | None
    diagnostics: MoveForecast


def _extract_numbers(block: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    values = tuple(int(value) for value in block.strip().split())
    if len(values) != 6:
        raise ValueError("negotiation input must contain three count/value pairs")
    return (values[0], values[2], values[4]), (values[1], values[3], values[5])


def _parse_line(line: str) -> NegotiationDialogue:
    input_match = re.search(r"<input>(.*?)</input>", line)
    dialogue_match = re.search(r"<dialogue>(.*?)</dialogue>", line)
    partner_match = re.search(r"<partner_input>(.*?)</partner_input>", line)
    if not input_match or not dialogue_match or not partner_match:
        raise ValueError("malformed negotiation row")
    counts, values_you = _extract_numbers(input_match.group(1))
    partner_counts, values_them = _extract_numbers(partner_match.group(1))
    if counts != partner_counts:
        raise ValueError("the two parties disagree about item counts")
    raw_moves = re.findall(r"(YOU|THEM):\s*(.*?)\s*<eos>", dialogue_match.group(1))
    moves = []
    canonical_text = []
    for role, text in raw_moves:
        clean = " ".join(text.strip().split())
        canonical_text.append(clean)
        action = classify_dialogue_act(clean)
        if action is not None:
            moves.append(DialogueMove(0 if role == "YOU" else 1, clean, ACTION_INDEX[action]))
    # Mirror perspectives share transcript, inventory, and the unordered pair
    # of private value vectors.  Including the game prevents distinct short
    # conversations with identical text from being collapsed.
    game_key = repr((counts, tuple(sorted((values_you, values_them)))))
    digest = hashlib.sha256(
        ("\x1f".join(canonical_text) + "\x1e" + game_key).encode()
    ).hexdigest()[:20]
    return NegotiationDialogue(digest, counts, values_you, values_them, tuple(moves))


def load_negotiation_dialogues(path: str | Path) -> list[NegotiationDialogue]:
    """Load one perspective per unique dialogue, preventing mirror leakage."""

    selected: dict[str, NegotiationDialogue] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        dialogue = _parse_line(line)
        selected.setdefault(dialogue.identifier, dialogue)
    return [selected[key] for key in sorted(selected)]


def bargaining_geometry(
    item_counts: Sequence[int], values_self: Sequence[int], values_other: Sequence[int]
) -> BargainingGeometry:
    if len(item_counts) != 3 or len(values_self) != 3 or len(values_other) != 3:
        raise ValueError("the object-division game must have exactly three item types")
    maximum_self = math.fsum(count * value for count, value in zip(item_counts, values_self))
    maximum_other = math.fsum(count * value for count, value in zip(item_counts, values_other))
    scale_self = maximum_self if maximum_self > 0.0 else 1.0
    scale_other = maximum_other if maximum_other > 0.0 else 1.0
    allocations = []
    for q0 in range(item_counts[0] + 1):
        for q1 in range(item_counts[1] + 1):
            for q2 in range(item_counts[2] + 1):
                own = (q0, q1, q2)
                raw_self = math.fsum(q * value for q, value in zip(own, values_self))
                raw_other = math.fsum(
                    (count - q) * value
                    for count, q, value in zip(item_counts, own, values_other)
                )
                allocations.append(AllocationPoint(own, raw_self / scale_self, raw_other / scale_other))
    nash = max(
        allocations,
        key=lambda point: (
            point.utility_self * point.utility_other,
            min(point.utility_self, point.utility_other),
            point.utility_self + point.utility_other,
            point.allocation_self,
        ),
    )
    max_self = max(
        allocations,
        key=lambda point: (point.utility_self, point.utility_other, point.allocation_self),
    )
    pareto = []
    for candidate in allocations:
        dominated = any(
            other.utility_self >= candidate.utility_self
            and other.utility_other >= candidate.utility_other
            and (
                other.utility_self > candidate.utility_self
                or other.utility_other > candidate.utility_other
            )
            for other in allocations
        )
        if not dominated:
            pareto.append(candidate)
    self_shares = [
        (count * self_value) / scale_self
        for count, self_value in zip(item_counts, values_self)
    ]
    other_shares = [
        (count * other_value) / scale_other
        for count, other_value in zip(item_counts, values_other)
    ]
    conflict = 0.5 * math.fsum(abs(a - b) for a, b in zip(self_shares, other_shares))
    integrative = max(point.utility_self + point.utility_other for point in allocations) / 2.0
    return BargainingGeometry(
        tuple(allocations), nash, max_self, conflict, len(pareto) / len(allocations), integrative
    )


def _quantity_for_item(text: str, singular: str, total: int) -> int | None:
    plural = singular + "s"
    pattern = rf"\b(\d+|a|an|one|two|both|three|four|five|all)\s+(?:of\s+the\s+)?(?:{singular}|{plural})\b"
    match = re.search(pattern, text)
    if match:
        token = match.group(1)
        if token == "all":
            return total
        if token.isdigit():
            return min(total, int(token))
        return min(total, NUMBER_WORDS[token])
    if re.search(rf"\bthe\s+{singular}\b", text):
        return min(total, 1)
    if total == 1 and re.search(rf"\b(?:{singular}|{plural})\b", text):
        return 1
    return None


def infer_offer_to_listener(text: str, item_counts: Sequence[int]) -> tuple[int, int, int] | None:
    """Conservatively parse explicit allocations offered to the next speaker."""

    value = " ".join(text.lower().split())
    if not re.search(r"\b(book|books|hat|hats|ball|balls|rest)\b", value):
        return None
    listener = [None, None, None]
    speaker = [None, None, None]
    value = re.sub(r"\band\s+(?=you\b)", " | ", value)
    clauses = re.split(r"[,;|]|\bbut\b", value)
    for clause in clauses:
        direct_listener = bool(re.search(r"\b(you|your)\b", clause))
        direct_speaker = bool(re.search(r"\b(i|me|my|i'll|im|i'm)\b", clause))
        for index, item in enumerate(ITEMS):
            quantity = _quantity_for_item(clause, item, item_counts[index])
            if quantity is None:
                continue
            if direct_listener and not direct_speaker:
                listener[index] = quantity
            elif direct_speaker:
                speaker[index] = quantity
    if "you can have the rest" in value or "you get the rest" in value:
        for index in range(3):
            if listener[index] is None and speaker[index] is not None:
                listener[index] = item_counts[index] - speaker[index]
    for index in range(3):
        if listener[index] is None and speaker[index] is not None:
            listener[index] = item_counts[index] - speaker[index]
    if all(value is None for value in listener):
        return None
    # Do not fabricate unmentioned components; require a fully recoverable offer.
    if any(value is None for value in listener):
        return None
    return tuple(int(max(0, min(item_counts[i], value))) for i, value in enumerate(listener))


def _allocation_point(
    allocation: Sequence[int], geometry: BargainingGeometry
) -> AllocationPoint | None:
    target = tuple(allocation)
    return next((point for point in geometry.allocations if point.allocation_self == target), None)


def dialogue_states(dialogue: NegotiationDialogue) -> tuple[MoveState, ...]:
    states = []
    history: list[int] = []
    last_text: str | None = None
    for turn, move in enumerate(dialogue.moves):
        if move.actor == 0:
            values_self, values_other = dialogue.values_you, dialogue.values_them
        else:
            values_self, values_other = dialogue.values_them, dialogue.values_you
        geometry = bargaining_geometry(dialogue.item_counts, values_self, values_other)
        offer = None
        if last_text is not None:
            allocation = infer_offer_to_listener(last_text, dialogue.item_counts)
            if allocation is not None:
                offer = _allocation_point(allocation, geometry)
        states.append(MoveState(
            dialogue.identifier,
            move.actor,
            turn,
            dialogue.item_counts,
            values_self,
            values_other,
            tuple(history),
            last_text,
            geometry,
            offer,
        ))
        history.append(move.action)
        last_text = move.text
    return tuple(states)


class ConversationalPrior:
    """Smoothed empirical base measure conditioned on recent dialogue acts."""

    def __init__(self, smoothing: float = 1.0) -> None:
        if smoothing <= 0.0:
            raise ValueError("smoothing must be positive")
        self.smoothing = smoothing
        self.global_counts = [0] * len(ACTIONS)
        self.context_counts: dict[tuple[int, int, int], list[int]] = {}
        self.count_ranges: tuple[tuple[int, int], ...] | None = None
        self.value_ranges: tuple[tuple[int, int], ...] | None = None

    @staticmethod
    def context(state: MoveState) -> tuple[int, int, int]:
        previous = state.history[-1] if state.history else START_ACTION
        return previous, min(state.turn, 5), state.actor

    def fit(self, dialogues: Sequence[NegotiationDialogue]) -> "ConversationalPrior":
        counts_values = [[] for _ in range(3)]
        payoff_values = [[] for _ in range(3)]
        for dialogue in dialogues:
            for index, count in enumerate(dialogue.item_counts):
                counts_values[index].append(count)
            for values in (dialogue.values_you, dialogue.values_them):
                for index, value in enumerate(values):
                    payoff_values[index].append(value)
            for state, move in zip(dialogue_states(dialogue), dialogue.moves):
                self.global_counts[move.action] += 1
                key = self.context(state)
                bucket = self.context_counts.setdefault(key, [0] * len(ACTIONS))
                bucket[move.action] += 1
        self.count_ranges = tuple((min(values), max(values)) for values in counts_values)
        self.value_ranges = tuple((min(values), max(values)) for values in payoff_values)
        return self

    def probabilities(self, state: MoveState) -> Vector:
        if sum(self.global_counts) == 0:
            raise ValueError("fit must be called before probabilities")
        local = self.context_counts.get(self.context(state), [0] * len(ACTIONS))
        global_total = sum(self.global_counts)
        global_probability = [
            (count + self.smoothing) / (global_total + self.smoothing * len(ACTIONS))
            for count in self.global_counts
        ]
        effective = [
            count + self.smoothing * len(ACTIONS) * global_probability[index]
            for index, count in enumerate(local)
        ]
        total = math.fsum(effective)
        return tuple(value / total for value in effective)

    def in_support(self, state: MoveState) -> bool:
        if self.count_ranges is None or self.value_ranges is None:
            raise ValueError("fit must be called before in_support")
        return all(low <= value <= high for value, (low, high) in zip(state.item_counts, self.count_ranges)) and all(
            low <= value <= high
            for values in (state.values_self, state.values_other)
            for value, (low, high) in zip(values, self.value_ranges)
        )


def default_strategic_types() -> tuple[StrategicType, ...]:
    return tuple(
        StrategicType(beta, discount, social, toughness)
        for beta in (0.5, 1.5, 4.0)
        for discount in (0.65, 0.85, 0.97)
        for social in (-0.25, 0.0, 0.25)
        for toughness in (-0.75, 0.0, 0.75)
    )


def _best_offer(
    geometry: BargainingGeometry, reservation_other: float, social: float
) -> AllocationPoint:
    feasible = [point for point in geometry.allocations if point.utility_other + 1e-12 >= reservation_other]
    if not feasible:
        feasible = list(geometry.allocations)
    return max(
        feasible,
        key=lambda point: (
            point.utility_self + social * point.utility_other,
            point.utility_self * point.utility_other,
            point.allocation_self,
        ),
    )


def counterfactual_action_values(state: MoveState, strategic_type: StrategicType) -> Vector:
    """Utility of each dialogue act under a one-step continuation abstraction."""

    geometry = state.geometry
    progress = min(1.0, state.turn / 10.0)
    nash = geometry.nash
    offer = state.last_offer or nash
    reservation = max(
        0.0,
        min(1.0, nash.utility_other * (0.85 - 0.25 * strategic_type.toughness)),
    )
    proposal = _best_offer(geometry, reservation, strategic_type.social)
    insist_reservation = max(0.0, reservation * 0.45)
    insist_offer = _best_offer(geometry, insist_reservation, strategic_type.social)

    def utility(point: AllocationPoint) -> float:
        return point.utility_self + strategic_type.social * point.utility_other

    break_risk = min(
        0.80,
        max(0.02, 0.08 + 0.30 * geometry.conflict + 0.08 * strategic_type.toughness),
    )
    information_bonus = 0.08 * geometry.conflict * (1.0 - progress)
    values = {
        "propose": strategic_type.discount * utility(proposal) - 0.025 * (1.0 + progress),
        "accept": utility(offer) + 0.05 * progress,
        "reject": 0.0,
        "inquire": strategic_type.discount * utility(nash) + information_bonus - 0.035,
        "insist": strategic_type.discount * (1.0 - break_risk) * utility(insist_offer) - 0.055,
        "other": strategic_type.discount * 0.82 * utility(nash) - 0.065,
    }
    return tuple(values[action] for action in ACTIONS)


def entropy_regularized_response(
    state: MoveState, strategic_type: StrategicType, base: Sequence[float]
) -> Vector:
    """Unique maximizer of E[Q] - KL(pi || base)/beta."""

    if len(base) != len(ACTIONS) or any(value <= 0.0 for value in base):
        raise ValueError("base measure must be strictly positive over every action")
    values = counterfactual_action_values(state, strategic_type)
    logits = [math.log(probability) + strategic_type.beta * value for probability, value in zip(base, values)]
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    total = math.fsum(weights)
    return tuple(value / total for value in weights)


def _effective_count(weights: Sequence[float]) -> float:
    return 1.0 / math.fsum(value * value for value in weights)


def _total_variation(left: Sequence[float], right: Sequence[float]) -> float:
    return 0.5 * math.fsum(abs(a - b) for a, b in zip(left, right))


class StrategicMoveForecaster:
    """Bayesian latent-type filter over entropy-regularized game responses."""

    def __init__(
        self,
        prior: ConversationalPrior,
        strategic_types: Sequence[StrategicType] | None = None,
    ) -> None:
        self.prior = prior
        self.types = tuple(strategic_types or default_strategic_types())
        if not self.types:
            raise ValueError("at least one strategic type is required")
        self.population_weights: Vector = tuple([1.0 / len(self.types)] * len(self.types))
        self.optimization: dict[str, object] = {}

    def _dialogue_type_log_likelihoods(self, dialogue: NegotiationDialogue) -> array:
        likelihoods = array("d", [0.0] * len(self.types))
        for state, move in zip(dialogue_states(dialogue), dialogue.moves):
            base = self.prior.probabilities(state)
            for index, strategic_type in enumerate(self.types):
                probability = entropy_regularized_response(state, strategic_type, base)[move.action]
                likelihoods[index] += math.log(max(1e-15, probability))
        return likelihoods

    def fit_population(
        self,
        dialogues: Sequence[NegotiationDialogue],
        *,
        dirichlet: float = 0.25,
        max_iterations: int = 100,
        tolerance: float = 1e-9,
    ) -> "StrategicMoveForecaster":
        if not dialogues:
            raise ValueError("dialogues must be nonempty")
        log_likelihoods = [self._dialogue_type_log_likelihoods(dialogue) for dialogue in dialogues]
        weights = [1.0 / len(self.types)] * len(self.types)
        objective = -math.inf
        for iteration in range(max_iterations):
            responsibilities = [dirichlet] * len(self.types)
            next_objective = 0.0
            for values in log_likelihoods:
                joint = [math.log(max(weight, 1e-300)) + value for weight, value in zip(weights, values)]
                maximum = max(joint)
                denominator = math.fsum(math.exp(value - maximum) for value in joint)
                log_normalizer = maximum + math.log(denominator)
                next_objective += log_normalizer
                for index, value in enumerate(joint):
                    responsibilities[index] += math.exp(value - log_normalizer)
            total = math.fsum(responsibilities)
            updated = [value / total for value in responsibilities]
            change = max(abs(a - b) for a, b in zip(weights, updated))
            weights = updated
            objective = next_objective
            if change <= tolerance:
                break
        self.population_weights = tuple(weights)
        self.optimization = {
            "optimizer": "convex_fixed_component_mixture_em",
            "iterations": iteration + 1,
            "converged": change <= tolerance,
            "maximum_weight_change": change,
            "training_dialogues": len(dialogues),
            "types": len(self.types),
            "mixture_log_likelihood": objective,
            "effective_population_types": _effective_count(weights),
        }
        return self

    def start_dialogue(self) -> "StrategicDialogueFilter":
        return StrategicDialogueFilter(self)


class StrategicDialogueFilter:
    def __init__(self, model: StrategicMoveForecaster) -> None:
        self.model = model
        self.weights = list(model.population_weights)
        self._predictions: tuple[Vector, ...] | None = None

    def predict(self, state: MoveState) -> MoveForecast:
        base = self.model.prior.probabilities(state)
        predictions = tuple(
            entropy_regularized_response(state, strategic_type, base)
            for strategic_type in self.model.types
        )
        mixture = tuple(
            math.fsum(weight * prediction[action] for weight, prediction in zip(self.weights, predictions))
            for action in range(len(ACTIONS))
        )
        disagreement = max(_total_variation(prediction, mixture) for prediction in predictions)
        self._predictions = predictions
        return MoveForecast(mixture, _effective_count(self.weights), disagreement)

    def update(self, observed_action: int) -> None:
        if self._predictions is None:
            raise RuntimeError("call predict before update")
        if not 0 <= observed_action < len(ACTIONS):
            raise ValueError("observed action is invalid")
        updated = [
            weight * prediction[observed_action]
            for weight, prediction in zip(self.weights, self._predictions)
        ]
        total = math.fsum(updated)
        self.weights = [value / total for value in updated]
        self._predictions = None


def selected_move_index(dialogue: NegotiationDialogue) -> int:
    if not dialogue.moves:
        raise ValueError("dialogue contains no modeled moves")
    return int(dialogue.identifier, 16) % len(dialogue.moves)


def selected_move_forecast(
    model: StrategicMoveForecaster, dialogue: NegotiationDialogue
) -> tuple[MoveForecast, int, MoveState]:
    target = selected_move_index(dialogue)
    sequence = model.start_dialogue()
    for index, (state, move) in enumerate(zip(dialogue_states(dialogue), dialogue.moves)):
        forecast = sequence.predict(state)
        if index == target:
            return forecast, move.action, state
        sequence.update(move.action)
    raise AssertionError("selected move was not reached")


def first_move_forecast(
    model: StrategicMoveForecaster, dialogue: NegotiationDialogue
) -> tuple[MoveForecast, int, MoveState]:
    """Forecast the pre-outcome first move of one dialogue."""

    if not dialogue.moves:
        raise ValueError("dialogue contains no modeled moves")
    state = dialogue_states(dialogue)[0]
    return model.start_dialogue().predict(state), dialogue.moves[0].action, state


def aps_score(probabilities: Sequence[float], outcome: int) -> float:
    target = probabilities[outcome]
    return min(1.0, math.fsum(value for value in probabilities if value >= target - 1e-15))


class FailClosedMoveWrapper:
    """First-move dialogue-grouped conformal set with explicit abstention."""

    def __init__(
        self,
        model: StrategicMoveForecaster,
        *,
        alpha: float = 0.10,
        maximum_set_size: int = 3,
        minimum_top_probability: float = 0.30,
        maximum_type_disagreement: float = 0.35,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if not 1 <= maximum_set_size <= len(ACTIONS):
            raise ValueError("maximum_set_size is invalid")
        self.model = model
        self.alpha = alpha
        self.maximum_set_size = maximum_set_size
        self.minimum_top_probability = minimum_top_probability
        self.maximum_type_disagreement = maximum_type_disagreement
        self.quantile: float | None = None

    def calibrate(self, dialogues: Sequence[NegotiationDialogue]) -> "FailClosedMoveWrapper":
        scores = sorted(
            aps_score(forecast.probabilities, outcome)
            for dialogue in dialogues
            for forecast, outcome, _ in [first_move_forecast(self.model, dialogue)]
        )
        if not scores:
            raise ValueError("calibration dialogues must be nonempty")
        rank = math.ceil((len(scores) + 1) * (1.0 - self.alpha))
        self.quantile = math.inf if rank > len(scores) else scores[rank - 1]
        return self

    def wrap(self, state: MoveState, diagnostics: MoveForecast) -> FailClosedForecast:
        if self.quantile is None:
            raise ValueError("calibrate must be called before wrap")
        ranked = sorted(range(len(ACTIONS)), key=lambda index: (-diagnostics.probabilities[index], index))
        prediction_set = tuple(
            action for action in range(len(ACTIONS))
            if aps_score(diagnostics.probabilities, action) <= self.quantile
        )
        if not self.model.prior.in_support(state):
            status = "ABSTAIN_OUT_OF_SUPPORT"
        elif not prediction_set:
            status = "ABSTAIN_EMPTY_SET"
        elif len(prediction_set) > self.maximum_set_size:
            status = "ABSTAIN_SET_TOO_WIDE"
        elif max(diagnostics.probabilities) < self.minimum_top_probability:
            status = "ABSTAIN_LOW_POINT_CONFIDENCE"
        elif diagnostics.maximum_type_disagreement > self.maximum_type_disagreement:
            status = "ABSTAIN_TYPE_DISAGREEMENT"
        elif sum(
            abs(value - max(diagnostics.probabilities)) <= 1e-15
            for value in diagnostics.probabilities
        ) > 1:
            status = "ABSTAIN_POINT_TIE"
        else:
            status = "FORECAST"
        point = max(range(len(ACTIONS)), key=lambda index: (diagnostics.probabilities[index], -index)) if status == "FORECAST" else None
        return FailClosedForecast(status, diagnostics.probabilities, prediction_set, point, diagnostics)


class SoftmaxMoveBaseline:
    """Deterministic dependency-free multinomial logistic baseline."""

    def __init__(self, *, include_game: bool, learning_rate: float = 0.12, l2: float = 1e-4, epochs: int = 18) -> None:
        self.include_game = include_game
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs
        self.weights: list[list[float]] = []
        self.optimization: dict[str, object] = {}

    @staticmethod
    def features(state: MoveState, *, include_game: bool) -> Vector:
        previous = state.history[-1] if state.history else START_ACTION
        second = state.history[-2] if len(state.history) >= 2 else START_ACTION
        result = [1.0, min(state.turn, 10) / 10.0, float(state.actor)]
        result.extend(float(previous == index) for index in range(len(ACTIONS) + 1))
        result.extend(float(second == index) for index in range(len(ACTIONS) + 1))
        if include_game:
            geometry = state.geometry
            offer = state.last_offer
            total_items = sum(state.item_counts)
            result.extend((
                geometry.nash.utility_self,
                geometry.nash.utility_other,
                geometry.conflict,
                geometry.pareto_fraction,
                geometry.integrative_potential,
                geometry.max_self.utility_self - geometry.nash.utility_self,
                float(offer is not None),
                offer.utility_self if offer is not None else geometry.nash.utility_self,
                offer.utility_other if offer is not None else geometry.nash.utility_other,
                min(total_items, 10) / 10.0,
            ))
        return tuple(result)

    def fit(self, dialogues: Sequence[NegotiationDialogue], seed: int = 41_901) -> "SoftmaxMoveBaseline":
        examples = [
            (self.features(state, include_game=self.include_game), move.action)
            for dialogue in dialogues
            for state, move in zip(dialogue_states(dialogue), dialogue.moves)
        ]
        if not examples:
            raise ValueError("training dialogues must be nonempty")
        dimensions = len(examples[0][0])
        self.weights = [[0.0] * dimensions for _ in ACTIONS]
        accumulators = [[1e-8] * dimensions for _ in ACTIONS]
        rng = random.Random(seed)
        order = list(range(len(examples)))
        objective = math.inf
        for epoch in range(self.epochs):
            rng.shuffle(order)
            loss = 0.0
            for example_index in order:
                features, outcome = examples[example_index]
                logits = [math.fsum(w * x for w, x in zip(row, features)) for row in self.weights]
                maximum = max(logits)
                exponents = [math.exp(value - maximum) for value in logits]
                total = math.fsum(exponents)
                probabilities = [value / total for value in exponents]
                loss -= math.log(max(1e-15, probabilities[outcome]))
                for action in range(len(ACTIONS)):
                    error = probabilities[action] - float(action == outcome)
                    for index, feature in enumerate(features):
                        gradient = error * feature + (self.l2 * self.weights[action][index] if index else 0.0)
                        accumulators[action][index] += gradient * gradient
                        self.weights[action][index] -= self.learning_rate * gradient / math.sqrt(accumulators[action][index])
            objective = loss / len(examples)
        self.optimization = {
            "optimizer": "deterministic_shuffled_adagrad",
            "epochs": self.epochs,
            "examples": len(examples),
            "features": dimensions,
            "final_online_epoch_nll": objective,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
        }
        return self

    def predict(self, state: MoveState) -> Vector:
        if not self.weights:
            raise ValueError("fit must be called before predict")
        features = self.features(state, include_game=self.include_game)
        logits = [math.fsum(w * x for w, x in zip(row, features)) for row in self.weights]
        maximum = max(logits)
        exponents = [math.exp(value - maximum) for value in logits]
        total = math.fsum(exponents)
        return tuple(value / total for value in exponents)
