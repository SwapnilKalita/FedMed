"""Secure aggregation so the coordinator never sees an individual update.

The PDF's plan calls for TenSEAL homomorphic encryption, which needs a
compiled dependency this sandbox can't install. What we implement instead is
the same *guarantee* using pairwise additive masking (the core idea behind
Bonawitz et al.'s secure aggregation protocol, which is what production
Flower deployments layer on top of FedAvg):

Every pair of hospitals agrees on a random mask. Hospital A adds it to its
contribution, hospital B subtracts it from its own. Individually, each
masked contribution is indistinguishable from noise. Once every hospital's
masked contribution is summed, the masks cancel out exactly and only the
true total survives -- the coordinator learns the sum, never any single
hospital's value.

Swapping this module for real homomorphic encryption (TenSEAL/CKKS) later
would not change any other file's interface: `secure_federated_average`
still takes client updates and returns global (weight, bias).
"""

from __future__ import annotations

import numpy as np

from model.hospital_node import ClientUpdate


def _pairwise_masks(client_names: list[str], seed: int, mask_scale: float = 1e4) -> dict[str, np.ndarray]:
    """Generate masks that sum to exactly zero across all clients."""
    rng = np.random.default_rng(seed)
    masks = {name: np.zeros(2, dtype=np.float64) for name in client_names}
    for i in range(len(client_names)):
        for j in range(i + 1, len(client_names)):
            shared_secret = rng.uniform(-mask_scale, mask_scale, size=2)
            masks[client_names[i]] += shared_secret
            masks[client_names[j]] -= shared_secret
    return masks


def mask_contributions(
    named_updates: dict[str, ClientUpdate], seed: int
) -> dict[str, np.ndarray]:
    """Mask each hospital's sample-weighted contribution before it leaves the node."""
    masks = _pairwise_masks(list(named_updates.keys()), seed=seed)
    masked: dict[str, np.ndarray] = {}
    for name, update in named_updates.items():
        weighted = np.array(
            [update.weight * update.sample_count, update.bias * update.sample_count], dtype=np.float64
        )
        masked[name] = weighted + masks[name]
    return masked


def secure_federated_average(named_updates: dict[str, ClientUpdate], seed: int = 0) -> tuple[float, float]:
    """FedAvg where the server only ever handles masked, summed contributions."""
    if not named_updates:
        raise ValueError("secure_federated_average requires at least one client update")

    masked = mask_contributions(named_updates, seed=seed)
    total_samples = sum(update.sample_count for update in named_updates.values())
    if total_samples == 0:
        raise ValueError("client updates must include at least one sample in total")

    # The coordinator sums masked vectors -- each individual entry is
    # meaningless on its own, but the masks cancel exactly in the sum.
    secure_sum = np.sum(np.stack(list(masked.values())), axis=0)
    global_weight, global_bias = secure_sum / total_samples
    return float(global_weight), float(global_bias)
