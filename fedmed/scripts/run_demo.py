"""Simple local federated demo using the repository stubs.

This demo runs a 3-client simulation using LocalTrainer.get_weights() and
MockSecureAggregator to demonstrate the end-to-end wiring without external
dependencies.
"""
# When executed as a script, ensure the repository root is on sys.path so
# the `fedmed` package can be imported without installing the package.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fedmed.fl.client import LocalTrainer
from fedmed.encryption.mock import MockSecureAggregator
from fedmed.model.unet3d import UNet3D
import numpy as np


def run_demo():
    # Prepare a model summary and three synthetic updates
    model_stub = UNet3D()
    print(model_stub.summary())

    # Create three local trainers (they would normally hold local datasets)
    trainers = [LocalTrainer(model_stub) for _ in range(3)]

    # Each trainer produces a small weight vector (stubbed)
    updates = [t.get_weights() + (i * 0.1) for i, t in enumerate(trainers)]
    print("Local updates:")
    for u in updates:
        print(u)

    # Secure aggregation simulation
    agg = MockSecureAggregator(rng_seed=1)
    masked = []
    masks = []
    for u in updates:
        m_u, m = agg.mask_update(u)
        masked.append(m_u)
        masks.append(m)

    averaged = agg.aggregate(masked, masks)
    print("Aggregated (averaged) update:", averaged)


if __name__ == "__main__":
    run_demo()
