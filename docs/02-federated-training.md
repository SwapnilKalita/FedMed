# Federated Training

## Why FedAvg

Each hospital trains locally, starting from the current global model, then sends
back only its updated `(weight, bias)` plus how many cases it used. The
coordinator (`server/federated_averaging.py`) combines updates weighted by
sample count -- Federated Averaging (FedAvg), the same algorithm Flower's
`FedAvg` strategy implements.

Weighting by sample count matters because hospitals rarely have equal amounts
of data (see `partition_across_hospitals`, which deliberately creates uneven
splits). Without weighting, a hospital with 5 cases would have the same say
as one with 50.

## The training loop

`server/coordinator.py` runs the full round-based loop:

1. Broadcast the current global `(weight, bias)` to every hospital node.
2. Each node calls `HospitalNode.local_train`, which warm-starts from the
   global parameters and trains only on its own private cases.
3. The coordinator aggregates the returned updates (see
   `03-secure-aggregation.md` for how it does this without seeing any single
   hospital's raw update).
4. The new global model is evaluated on a held-out test set that no hospital
   trained on, and its Dice score is recorded.

## Node resilience

Real hospital nodes go offline -- a maintenance window, a network blip. The
coordinator accepts an `offline_rounds` map (round number -> unavailable
hospital names) and simply aggregates whichever nodes are still online that
round, rather than failing the whole training run. `tests/test_coordinator.py`
proves training still converges when a hospital drops out for a couple of
rounds and comes back.

## Run it

```bash
python run_experiment.py
python -m unittest discover -s tests
```
