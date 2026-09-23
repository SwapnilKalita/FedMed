# Differential Privacy

## Why this is a separate protection from secure aggregation

Secure aggregation (previous doc) hides individual hospital updates from the
coordinator. It does **not** protect against someone inspecting the *final*
model and inferring facts about training examples through memorization --
the basis of model-inversion and membership-inference attacks the PDF names
as the threat to defend against.

`model/differential_privacy.py` adds that second layer: calibrated Gaussian
noise added to the aggregated update every round, following the classic
Gaussian mechanism for `(epsilon, delta)`-differential privacy.

## The two knobs

- **`clip_norm`** bounds how much any single round's update is allowed to move
  the global model, so one hospital's aggregated update (even after secure
  aggregation) can't dominate the result or leak an outsized signal.
- **`epsilon`** controls the privacy budget: smaller epsilon means more noise
  and stronger formal privacy, at the cost of slower or noisier convergence.
  `PrivacyBudget.noise_multiplier()` implements this trade-off directly --
  `tests/test_differential_privacy.py::test_smaller_epsilon_adds_more_noise_on_average`
  checks the relationship holds.

## What `run_experiment.py` shows

Comparing `federated` (secure aggregation only) against `federated_with_dp`
(secure aggregation + noise) on the dashboard makes the privacy/utility
trade-off visible: the DP run's Dice score is slightly lower and noisier
round-to-round than the non-private federated run, which itself tracks the
centralized reference closely.

## Honesty about scope

This is a simplified, illustrative Gaussian mechanism for a 2-parameter toy
model -- useful for seeing the trade-off clearly, but **not** a formally
audited `(epsilon, delta)` accountant across multiple rounds. A production
system training a real 3D U-Net would track a cumulative privacy budget
across every round with a library such as Opacus or TensorFlow Privacy, and
noise would be added to per-example gradients (DP-SGD), not to a single
aggregated scalar pair as done here.
