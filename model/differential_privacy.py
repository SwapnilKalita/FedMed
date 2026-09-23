"""Calibrated noise added to the aggregated global update.

Secure aggregation (server/secure_aggregation.py) stops the coordinator from
reading any single hospital's update. Differential privacy is a separate,
complementary protection: even the final *aggregated* model can leak
information about individual training examples through memorization, which
is what model-inversion / membership-inference attacks exploit.

This module adds Gaussian noise to the aggregated (weight, bias) update,
scaled by a privacy budget epsilon and a clipping bound on how much any one
round is allowed to move the model. Smaller epsilon means more noise and
stronger privacy, at the cost of slower/noisier convergence.

This is a simplified, illustrative implementation of the Gaussian mechanism
for teaching purposes -- not a formally audited (epsilon, delta)-DP
accountant. A production system would track a cumulative privacy budget
across rounds with a library such as Opacus or TensorFlow Privacy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PrivacyBudget:
    """Parameters controlling how much noise protects each aggregation round."""

    epsilon: float = 1.0
    clip_norm: float = 5.0
    delta: float = 1e-5

    def noise_multiplier(self) -> float:
        """Gaussian mechanism: noise scale grows as epsilon shrinks."""
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        return self.clip_norm * float(np.sqrt(2.0 * np.log(1.25 / self.delta))) / self.epsilon


def clip_update(weight: float, bias: float, clip_norm: float) -> tuple[float, float]:
    """Bound one round's update so a single outlier hospital can't dominate it."""
    vector = np.array([weight, bias], dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= clip_norm or norm == 0.0:
        return weight, bias
    scaled = vector * (clip_norm / norm)
    return float(scaled[0]), float(scaled[1])


def add_privacy_noise(
    weight: float, bias: float, budget: PrivacyBudget, seed: int | None = None
) -> tuple[float, float]:
    """Clip then add calibrated Gaussian noise to the aggregated update."""
    clipped_weight, clipped_bias = clip_update(weight, bias, budget.clip_norm)
    generator = np.random.default_rng(seed)
    noise = generator.normal(loc=0.0, scale=budget.noise_multiplier(), size=2)
    return clipped_weight + float(noise[0]), clipped_bias + float(noise[1])
