"""Flower client template for FedMed.

Provides a simple Flower client wrapper demonstrating how a hospital node
could expose local training and weight extraction. The module is import-safe
when `flwr` or `torch` are not installed.
"""
from typing import Any, Dict


class LocalTrainer:
    """Small local trainer abstraction used by the Flower client template."""

    def __init__(self, model: Any):
        self.model = model

    def get_weights(self):
        """Return a NumPy-like representation of model weights.

        In a real PyTorch model this would return [param.numpy() for param in model.parameters()].
        The function intentionally avoids importing torch at module import time.
        """
        # If the model exposes build_torch_model() and torch is available, use it.
        try:
            import numpy as _np
        except Exception:
            raise RuntimeError("numpy is required to use LocalTrainer.get_weights()")

        # This stub returns a small fixed-weight vector to allow simulation without torch
        return _np.array([0.1, 0.2, 0.3], dtype=_np.float32)


# Flower client scaffold (lazy imports)

def start_flower_client(server_address: str = "127.0.0.1:8080"):
    try:
        import flwr as fl
    except Exception as exc:
        raise RuntimeError("Flower (flwr) is required to run the actual client. Install with `pip install flwr`.") from exc

    class MinimalClient(fl.client.NumPyClient):
        def __init__(self):
            super().__init__()

        def get_parameters(self):
            import numpy as np
            # Example: return parameters as a list of numpy arrays
            return [np.array([0.1, 0.2, 0.3], dtype=np.float32)]

        def fit(self, parameters, config):
            # Perform local training here
            return self.get_parameters(), 1, {}

        def evaluate(self, parameters, config):
            # Return loss and metrics
            return 0.5, 0, {}

    print(f"Connecting Flower client to server at {server_address}")
    fl.client.start_numpy_client(server_address=server_address, client=MinimalClient())


if __name__ == "__main__":
    try:
        start_flower_client()
    except RuntimeError as e:
        print(e)
