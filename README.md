# STRIDE Negotiation Forecasting

A payoff-conditioned next-move classifier evaluated on the public Deal or No Deal negotiation corpus. The archived model obtained 44.30% action accuracy against 42.19% for the matched history-only model on 2,650 held-out moves.

The evaluation gives the model both parties' assigned payoff tables. This information was private to each human negotiator. The result concerns an experimenter with that information. [Detailed study and limitations](REPORT_V12.md) · [Aggregate outcomes](results/strategic_move_results.json)

The public release includes the actual forecasting and evaluation code, requiring Python 3.10+ and the standard library. Provision the original public corpus at `external/end-to-end-negotiator/src/data/negotiate/` with train.txt, val.txt and test.txt. Then run:

```sh
python3 -c "from pathlib import Path; from stride.strategic_move_experiments import run; run(Path('/tmp/stride-replay'))"
```

The code and aggregate outputs were extracted from the local STRIDE V12 project. Training and corpus provision are separate from the archived result. The full report records rejected alternatives and evaluation assumptions.
