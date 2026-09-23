"""Federated Averaging (FedAvg): the aggregation strategy Flower would run.

The coordinator never sees raw data, only each hospital's `ClientUpdate`. It
combines them into one global model, weighting each hospital by how many
cases it trained on so a small clinic doesn't get outsized influence noise,
and a large one isn't unfairly diluted either.
"""

from __future__ import annotations

from model.hospital_node import ClientUpdate


def federated_average(updates: list[ClientUpdate]) -> tuple[float, float]:
    """Combine client updates into new global (weight, bias) parameters."""
    if not updates:
        raise ValueError("federated_average requires at least one client update")

    total_samples = sum(update.sample_count for update in updates)
    if total_samples == 0:
        raise ValueError("client updates must include at least one sample in total")

    global_weight = sum(update.weight * update.sample_count for update in updates) / total_samples
    global_bias = sum(update.bias * update.sample_count for update in updates) / total_samples
    return global_weight, global_bias
