"""Run the full FedMed comparison and write results for the dashboard.

Trains three ways on the same synthetic data split:
  1. Centralized baseline  -- all data on one "machine" (the reference score).
  2. Federated + secure agg -- three hospitals, FedAvg, masked aggregation.
  3. Federated + secure agg + differential privacy -- adds calibrated noise.

Usage:
    python run_experiment.py
"""

from __future__ import annotations

import json
from pathlib import Path

from model.baseline import IntensitySegmentationBaseline, dice_score
from model.differential_privacy import PrivacyBudget
from model.hospital_node import HospitalNode
from model.synthetic_data import make_synthetic_mri_cases, partition_across_hospitals, split_cases
from server.coordinator import run_federated_training

ROUNDS = 15
LOCAL_EPOCHS = 20
HOSPITAL_NAMES = ("Hospital A", "Hospital B", "Hospital C")


def main() -> None:
    scans, masks = make_synthetic_mri_cases(case_count=48, seed=7)
    train_scans, train_masks, test_scans, test_masks = split_cases(scans, masks)

    # 1. Centralized baseline.
    centralized_model = IntensitySegmentationBaseline()
    centralized_losses = centralized_model.fit(train_scans, train_masks, epochs=LOCAL_EPOCHS * ROUNDS)
    centralized_dice = dice_score(centralized_model.predict(test_scans), test_masks)

    partition = partition_across_hospitals(train_scans, train_masks, HOSPITAL_NAMES)
    nodes = [HospitalNode(name=name, scans=s, masks=m) for name, (s, m) in partition.items()]

    # 2. Federated + secure aggregation, no differential privacy.
    federated_result = run_federated_training(
        [HospitalNode(name=n.name, scans=n.scans, masks=n.masks) for n in nodes],
        test_scans,
        test_masks,
        rounds=ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        offline_rounds={5: {"Hospital B"}},  # simulate one hospital dropping out briefly
        seed=1,
    )

    # 3. Federated + secure aggregation + differential privacy.
    private_result = run_federated_training(
        [HospitalNode(name=n.name, scans=n.scans, masks=n.masks) for n in nodes],
        test_scans,
        test_masks,
        rounds=ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        privacy_budget=PrivacyBudget(epsilon=20.0, clip_norm=12.0),
        seed=1,
    )

    results = {
        "hospital_sample_counts": {name: len(s) for name, (s, _) in partition.items()},
        "centralized": {
            "final_loss": centralized_losses[-1],
            "loss_curve": centralized_losses[:: max(1, len(centralized_losses) // ROUNDS)][:ROUNDS],
            "final_dice": centralized_dice,
        },
        "federated": {
            "rounds": [r.round_number for r in federated_result.history],
            "loss_curve": [r.mean_local_loss for r in federated_result.history],
            "dice_curve": [r.held_out_dice for r in federated_result.history],
            "final_dice": federated_result.history[-1].held_out_dice,
        },
        "federated_with_dp": {
            "rounds": [r.round_number for r in private_result.history],
            "loss_curve": [r.mean_local_loss for r in private_result.history],
            "dice_curve": [r.held_out_dice for r in private_result.history],
            "final_dice": private_result.history[-1].held_out_dice,
        },
        "sample_mask": {
            "ground_truth": test_masks[0].astype(int).sum(axis=2).tolist(),
        },
    }

    output_path = Path(__file__).parent / "dashboard" / "results.json"
    output_path.write_text(json.dumps(results, indent=2))

    print(f"Centralized Dice:              {centralized_dice:.4f}")
    print(f"Federated Dice (secure agg):   {federated_result.history[-1].held_out_dice:.4f}")
    print(f"Federated Dice (+DP, eps=20):  {private_result.history[-1].held_out_dice:.4f}")
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
