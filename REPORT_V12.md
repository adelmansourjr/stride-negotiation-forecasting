# V12 — Game-conditioned next-move forecasting

## Executive verdict

V12 produced a **useful research prototype, not a breakthrough**.

The surviving result is **Nash Geometry Softmax (NGS)**: a probabilistic
classifier for the next coarse bargaining move, conditioned on dialogue history
and explicit payoff geometry. On a held-out human negotiation split, NGS
improved negative log likelihood (NLL) over an otherwise identical history-only
model by `0.03060` nats per dialogue, after averaging all turns within each
dialogue. A dialogue-cluster bootstrap interval was
`[-0.03824, -0.02300]`; lower NLL is better. Across all 2,650 held-out moves,
accuracy was `44.30%` over six actions, versus `42.19%` for the history-only
model.

This is evidence that game structure adds predictive information in this one
dataset. It is not evidence of a new general learning principle, individual
intent inference, war prediction, or operational readiness. The test set has
now been used and must not be treated as untouched confirmation in later work.

The more ambitious **Strategic Entropic Response Filter (SERF)** was rejected.
Its pure strategic forecast was indistinguishable from a conversational prior,
and its validation-selected hybrid was worse on test than NGS.

## Question and prediction target

The empirical question was deliberately narrower than predicting arbitrary
world events:

> Given the legal move set, the public dialogue history, and both parties'
> payoff tables, can game geometry improve the probability forecast of the next
> observable move?

For the human benchmark the outcome space was frozen to six auditable dialogue
acts:

\[
\mathcal A=\{\text{propose, accept, reject, inquire, insist, other}\}.
\]

The model predicts a distribution on this set. It does not generate text and
does not claim to recover hidden motives.

## Human data and protocol

The benchmark uses the human-human object-allocation negotiations released with
[Deal or No Deal?](https://aclanthology.org/D17-1259/). Each game has three item
types, a common inventory, and private value vectors for both participants.
The source paper describes a multi-issue bargaining task in which each player
cannot observe the other's reward function. The local corpus includes both
perspectives of many dialogues, so V12 canonicalized the dialogue text and kept
one perspective per unique conversation to prevent mirror leakage.

**This is an oracle-information benchmark:** NGS receives both assigned payoff
vectors even though neither human participant observed the other's vector. The
result applies to an experimenter or platform that legitimately knows the
assignment; it is not a public-information or negotiator-facing result.

After de-duplication:

| Split | Canonical dialogues | Modeled moves |
|---|---:|---:|
| train | 5,211 | 25,939 |
| validation | 562 | 2,778 |
| test | 545 | 2,650 |

There were zero train/validation, train/test, and validation/test overlaps under
the corrected identity key. The primary test statistic first
averages the paired loss difference over all moves in each dialogue and then
bootstraps the 545 dialogue averages. This uses every chronological target
without treating multiple turns from the same pair as independent.
Participant identifiers are unavailable, so independence between dialogues is
an unverified corpus assumption.

The action labels are frozen regular-expression rules. That makes the
abstraction inspectable, but it also means classification error is possible.
Only `3.70%` of held-out moves had a fully recoverable explicit allocation in
the previous utterance. Most of the measured game signal therefore comes from
the static payoff geometry, not a rich semantic representation of offers.

## Model 1: Nash Geometry Softmax

For state \(s\), NGS uses the feature vector

\[
x(s)=\big[1,t,i,a_{t-1},a_{t-2},G(s)\big],
\]

where \(t\) is capped normalized turn index, \(i\) is actor role, the two recent
actions are one-hot encoded, and

\[
G(s)=\left[
u_i(a^N),u_{-i}(a^N),C,F_P,I,
u_i(a_i^{\max})-u_i(a^N),O,u_i(O),u_{-i}(O),N_{items}
\right].
\]

Here \(a^N\) maximizes the normalized Nash product, \(C\) is value conflict,
\(F_P\) is the fraction of discrete allocations on the Pareto frontier, \(I\)
is integrative potential, and \(O\) is the last offer when the conservative
parser can recover it. Otherwise the Nash allocation is used as the offer
fallback and a separate bit marks it as unparsed.

The predictive law is multinomial softmax:

\[
p_\theta(a\mid s)=
\frac{\exp(w_a^\top x(s))}
{\sum_{b\in\mathcal A}\exp(w_b^\top x(s))}.
\]

Training minimizes regularized cross entropy

\[
L(W)=-\frac1n\sum_{r=1}^n \log p_W(y_r\mid s_r)
+\frac\lambda2\lVert W\rVert_F^2.
\]

For one example its Hessian has the form

\[
(x x^\top)\otimes\left[\operatorname{diag}(p)-pp^\top\right],
\]

and the bracketed matrix is a categorical covariance matrix. Consequently
\(\nabla^2L\succeq0\). Without regularization, the usual common-logit direction
is redundant; with \(\lambda>0\), the full objective is strongly convex and the
coefficient minimizer is unique. There are no spurious local minima in this
model. The present dependency-free implementation uses deterministic
shuffled AdaGrad, so convexity characterizes the objective but does not by
itself prove that the finite training schedule reached the exact minimizer.

## Model 2: generic explicit-payoff API

`CounterfactualPayoffLogit` generalizes the same idea to a caller-supplied
finite strategic state. For actions \(a\), response contingencies \(r\), actor
and other-player payoff matrices \(U_i,U_{-i}\), a response belief \(q(r)\),
and an empirical base measure \(\mu(a\mid h)>0\), it computes actionwise
features

\[
\phi(s,a)=\left[
E_{q_a}U_i,\min U_i,\max U_i,\operatorname{sd}_{q_a}U_i,
P_{q_a}(a\in BR),E_{q_a}U_{-i},\mathbf 1(a=a_{t-1})
\right]
\]

and predicts

\[
\boxed{
p_\theta(a\mid s)=
\frac{\mu(a\mid h)\exp\{\theta^\top\phi(s,a)\}}
{\sum_b\mu(b\mid h)\exp\{\theta^\top\phi(s,b)\}}
}.
\]

Callers may supply \(q_a(r)=P(r\mid do(a))\). If omitted, the API explicitly
assumes an exogenous, action-invariant contingency law \(q_a=q\). The same
coefficient vector is shared across actions. Therefore relabeling actions
relabels the probability distribution. A tied point optimum causes abstention
rather than an index-dependent choice. Each player's payoff matrix is
independently range-normalized, so
the features and predictions are invariant under positive affine changes
\(U_i'=c_i+d_iU_i\), \(d_i>0\). These properties are covered by tests.

This is a conditional-logit synthesis of game features, not a new equilibrium
concept. Entropy-regularized or noisy best responses are longstanding ideas in
quantal-response models, and inferring utilities or behavior from observed play
is an established inverse-game problem; see the original
[quantal-response equilibrium formulation](https://doi.org/10.1006/game.1995.1023),
[Computational Rationalization](https://arxiv.org/abs/1103.5254), and a recent
[inverse Stackelberg-game analysis](https://arxiv.org/abs/2210.01380).
The V12 feature bundle and fail-closed software interface may be practically
useful, but the research performed here cannot establish algorithmic novelty.

The generic objective is convex and is fitted with deterministic L-BFGS plus
Armijo backtracking. Across all eight synthetic mechanism tests, the optimizer
reported convergence with final gradient norm below `1e-7` in 16–22
iterations.

## Rejected model: Strategic Entropic Response Filter

SERF attempted a more structural predictor:

\[
\pi_z(a\mid s)\propto \mu(a\mid h)
\exp\{\beta_z Q_z(s,a)\},
\]

with 81 latent types spanning response precision, discounting, social
preference, and toughness. A population mixture was fitted on training
dialogues, then updated within each conversation by Bayes' rule.

This attractive story did not survive testing:

| All held-out moves (2,650; raw-move metric) | NLL | accuracy |
|---|---:|---:|
| NGS | **1.407525** | **0.443019** |
| SERF + NGS, weight selected on validation | 1.408622 | 0.443019 |
| history-only softmax | 1.430691 | 0.421887 |
| conversational prior | 1.450519 | 0.419623 |
| SERF | 1.452518 | 0.419245 |

Validation selected a SERF weight of `0.22`; on held-out moves the hybrid was
slightly worse than NGS. The SERF population optimizer also hit its
100-iteration cap rather than the stated convergence tolerance. SERF is
therefore retained as a falsified candidate, not presented as the answer.

For a two-predictor mixture

\[
p_\gamma=(1-\gamma)p_0+\gamma p_1,
\qquad
L(\gamma)=-\sum_r\log p_\gamma(y_r),
\]

we have

\[
L''(\gamma)=\sum_r
\frac{(p_1(y_r)-p_0(y_r))^2}{p_\gamma(y_r)^2}\ge0.
\]

Thus the observed validation landscape has a global minimum and no spurious
local minima. Its interior optimum did not transfer to test: a concrete example
of validation selection overfitting despite a benign optimization landscape.

![Validation mixture landscape](/Users/adelmansour/Desktop/game%20theory/artifacts_v12/strategic_weight_landscape.svg)

## Held-out results and where the gain comes from

Primary dialogue-cluster bootstrap comparisons, candidate minus baseline NLL
after equal-weighting each dialogue's average loss:

| Baseline | mean difference | 95% paired bootstrap interval |
|---|---:|---:|
| history-only softmax | **-0.03060** | **[-0.03824, -0.02300]** |
| conversational prior | -0.04196 | [-0.05339, -0.03072] |
| SERF | -0.03507 | [-0.04621, -0.02411] |

The bootstrap resamples dialogue averages, not individual turns. These are
exploratory post-selection intervals, not preregistered confirmatory evidence.

Per-action differences `history NLL - NGS NLL` were:

| true action | improvement |
|---|---:|
| accept | **0.07382** |
| reject | 0.02605 |
| propose | 0.02105 |
| insist | 0.01620 |
| inquire | **-0.01962** |
| other | **-0.04534** |

The result is therefore not “game theory predicts language.” Payoff geometry
mainly helps identify when agreement or rejection becomes plausible. It also
helps proposals/insistence, but hurts inquiry and the heterogeneous “other”
class.

![Held-out NLL comparison](/Users/adelmansour/Desktop/game%20theory/artifacts_v12/human_next_move_nll.svg)

## Fail-closed uncertainty wrapper

The calibration layer uses a separate validation dialogue from each pair and an
adaptive prediction-set score

\[
S(p,y)=\sum_{a:p(a)\ge p(y)}p(a).
\]

With calibration scores \(S_1,\ldots,S_m\), level \(\alpha\), and

\[
k=\lceil(m+1)(1-\alpha)\rceil,
\qquad
\widehat q=\begin{cases}S_{(k)},&k\le m\\+\infty,&k=m+1.\end{cases}
\]

the deterministic APS set is
\(C(s)=\{a:S(p(s),a)\le\widehat q\}\). It emits a point forecast only when the
set has at most three actions and the top probability is unique; otherwise it
abstains. If the requested order-statistic rank exceeds the calibration size,
the implementation uses the full action set. Exchangeability between
calibration and future examples—not strategic rationality—is the assumption
behind marginal prediction-set coverage.

Calibrating on the first move of each of 562 validation dialogues and evaluating
the same pre-outcome first-move target in 545 test dialogues gave:

- observed marginal set coverage: `88.26%`;
- mean set size: `2.27` of six actions;
- issuance: `545/545` under the frozen three-action cap;
- point accuracy: `67.34%`.

The target coverage level was 90%; a realized finite test sample can fall below
the marginal guarantee, whose validity is over exchangeable calibration and
test sampling. The observed 88.26% therefore does not establish empirical 90%
coverage. The wrapper also did not abstain on this easy, proposal-heavy
first-turn target, so V12 does not establish useful selective behavior on human
data. Marginal set coverage is not a guarantee of point accuracy, conditional
coverage, or accuracy after issuance.

## Synthetic mechanism and impossibility suite

The generic API was tested on eight different finite-game generators, with
1,200 training, 400 calibration, and 800 test states per family. NLL improvement
over the supplied base measure ranged from `0.218` to `0.500` in seven
observable mechanism families. In the hidden-signal family, accuracy was only
`34.75%` and the wrapper issued no forecasts.

| family | NLL improvement | accuracy | issuance |
|---|---:|---:|---:|
| expected utility | 0.21804 | 0.61750 | 0.00000 |
| maximin | 0.22072 | 0.62250 | 0.00000 |
| regret averse | 0.31371 | 0.65625 | 0.85375 |
| prosocial welfare | 0.32165 | 0.66500 | 1.00000 |
| inertial | 0.50009 | 0.95250 | 0.98250 |
| mixed strategic | 0.24569 | 0.67250 | 0.99875 |
| nonlinear threshold | 0.31494 | 0.67000 | 0.99625 |
| hidden signal impossibility | 0.04901 | 0.34750 | 0.00000 |

These are mechanism checks, not external validation. The data-generating rules
were intentionally built from related payoff features, so strong matched-family
performance is partly circular. The hidden-signal result illustrates the basic
identifiability limit: if the next move depends on information absent from
\(s\), no optimizer can reconstruct it from “game theory” alone.

## What it can and cannot predict

The model is suitable for controlled research where the following are supplied
explicitly and consistently:

1. a small, predeclared action ontology;
2. response contingencies and a defensible response belief;
3. payoff or preference tables;
4. chronological histories available before the target move;
5. enough domain-matched examples for fitting and untouched calibration;
6. a willingness to accept frequent abstention.

Plausible low-stakes applications are next-act prediction in bargaining games,
aggregate coordination experiments, public-notice experiments, randomized
disclosure studies, and experimental mechanism design. It can also be used for
**retrospective or sandboxed historical conflict scenario analysis** after a
researcher defines coarse public actions and explicit payoff assumptions.

It should not be used for war prediction, negotiator profiling, surveillance,
targeting, military or coercive decisions, or autonomous high-stakes decisions.
The human evidence is one low-stakes bargaining corpus, not a conflict corpus;
private values were provided by the game; the action labels were coarse; and
the system saw no free-form world state. Supplying “everything” is not a
well-defined statistical condition: the model can only condition on encoded,
timestamped variables, and hidden information remains non-identifiable.

## Final scientific conclusion

The genuine result is narrower and more useful than a breakthrough claim:

> In this object-allocation benchmark, adding normalized Nash/Pareto payoff
> geometry to recent-action history materially improves held-out probabilistic
> prediction of coarse next moves, especially acceptance and rejection. A more
> structural latent-type game model fails. The corrected conformal wrapper does
> not establish its nominal coverage or useful selective abstention on the human
> benchmark and therefore remains a research component rather than a validated
> safety guarantee.

The next credible step is a new untouched dataset or a preregistered prospective
low-stakes bargaining experiment. For a conflict research benchmark, the next
step is not deployment: it is to freeze a public-action ontology, construct
timestamped information vintages, declare payoff scenarios rather than inferred
personal profiles, and evaluate strictly forward in time against simple
history-only and non-game baselines.

## Reproduction

```bash
python3 -m unittest discover -s tests -v
python3 -m stride.strategic_move_experiments
python3 -m stride.game_move_experiments
python3 -m stride.strategic_move_plots
```

Machine-readable results are in
[`results/strategic_move_results.json`](results/strategic_move_results.json)
and [`results/explicit_game_suite.json`](results/explicit_game_suite.json).
