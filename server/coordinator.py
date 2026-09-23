"""Orchestrates federated training rounds across hospital nodes.

This is what a Flower `Strategy` + server loop does in the real stack: hold
the current global model, broadcast it, collect updates, aggregate them
(securely), optionally add differential privacy, and evaluate progress --
all without a hospital's raw scans ever reaching this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from model.baseline import IntensitySegmentationBaseline, dice_score
from model.differential_privacy import PrivacyBudget, add_privacy_noise
from model.hospital_node import HospitalNode
from server.secure_aggregation import secure_federated_average


@dataclass
class RoundMetrics:
    round_number: int
    mean_local_loss: float
    held_out_dice: float


@dataclass
class FederatedTrainingResult:
    final_weight: float
    final_bias: float
    history: list[RoundMetrics] = field(default_factory=list)


def run_federated_training(
    nodes: list[HospitalNode],
    held_out_scans: np.ndarray,
    held_out_masks: np.ndarray,
    rounds: int = 15,
    local_epochs: int = 20,
    privacy_budget: PrivacyBudget | None = None,
    seed: int = 0,
    offline_rounds: dict[int, set[str]] | None = None,
) -> FederatedTrainingResult:
    """Run multiple rounds of secure federated averaging and track convergence.

    If `privacy_budget` is provided, Gaussian noise calibrated to it is added
    to the aggregated model after every round (see model/differential_privacy.py).

    `offline_rounds` simulates a hospital node dropping offline mid-training:
    map a round number to the set of hospital names unavailable that round.
    The coordinator must keep making progress with whichever nodes remain.
    """
    global_weight, global_bias = 0.0, 0.0
    history: list[RoundMetrics] = []
    offline_rounds = offline_rounds or {}

    for round_number in range(1, rounds + 1):
        offline_this_round = offline_rounds.get(round_number, set())
        active_nodes = [node for node in nodes if node.name not in offline_this_round]
        if not active_nodes:
            raise RuntimeError(f"round {round_number} has no available hospital nodes")

        named_updates = {
            node.name: node.local_train(global_weight, global_bias, epochs=local_epochs)
            for node in active_nodes
        }

        global_weight, global_bias = secure_federated_average(named_updates, seed=seed + round_number)

        if privacy_budget is not None:
            global_weight, global_bias = add_privacy_noise(
                global_weight, global_bias, privacy_budget, seed=seed + round_number
            )

        evaluation_model = IntensitySegmentationBaseline(weight=global_weight, bias=global_bias)
        held_out_score = dice_score(evaluation_model.predict(held_out_scans), held_out_masks)
        mean_local_loss = float(np.mean([update.local_loss for update in named_updates.values()]))

        history.append(
            RoundMetrics(round_number=round_number, mean_local_loss=mean_local_loss, held_out_dice=held_out_score)
        )

    return FederatedTrainingResult(final_weight=global_weight, final_bias=global_bias, history=history)
