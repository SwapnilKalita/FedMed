"""A simulated hospital participating in federated training.

Each node owns a private slice of scans and never exposes them. It only ever
sends out a *model update* (weight/bias deltas plus a sample count), which is
the same contract a real Flower `NumPyClient` would follow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.baseline import IntensitySegmentationBaseline


@dataclass
class ClientUpdate:
    """What a hospital node is allowed to send to the coordinator."""

    weight: float
    bias: float
    sample_count: int
    local_loss: float


@dataclass
class HospitalNode:
    """One simulated hospital with a private partition of scans/masks."""

    name: str
    scans: np.ndarray
    masks: np.ndarray

    def local_train(
        self, global_weight: float, global_bias: float, epochs: int = 20, learning_rate: float = 0.8
    ) -> ClientUpdate:
        """Warm-start from the current global model and train on local data only.

        Raw `self.scans` / `self.masks` never leave this method's scope.
        """
        model = IntensitySegmentationBaseline(weight=global_weight, bias=global_bias)
        losses = model.fit(self.scans, self.masks, epochs=epochs, learning_rate=learning_rate)
        return ClientUpdate(
            weight=model.weight,
            bias=model.bias,
            sample_count=len(self.scans),
            local_loss=losses[-1],
        )
