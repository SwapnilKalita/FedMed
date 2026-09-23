"""Flower server template for FedMed.

This module provides a lightweight template showing how to start a Flower
server using a FedAvg-like strategy. The functions are safe to import even
when `flwr` is not installed; they import Flower lazily inside the start
function.
"""
from typing import Optional


def start_flower_server(num_rounds: int = 1, port: int = 8080) -> None:
    """Start a Flower server for `num_rounds` rounds.

    If `flwr` is not available the function raises a RuntimeError with
    instructions to install it.
    """
    try:
        import flwr as fl
    except Exception as exc:
        raise RuntimeError("Flower (flwr) is required to run the actual server. Install with `pip install flwr`.") from exc

    # Minimal strategy: FedAvg
    strategy = fl.server.strategy.FedAvg()

    print(f"Starting Flower server on port {port} for {num_rounds} rounds (this will block)")
    fl.server.start_server(server_address=f"0.0.0.0:{port}", config=fl.server.ServerConfig(num_rounds=num_rounds), strategy=strategy)


if __name__ == "__main__":
    # Convenience: allow running the server template directly
    try:
        start_flower_server(num_rounds=1)
    except RuntimeError as e:
        print(e)
