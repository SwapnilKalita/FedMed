"""A small centralized segmentation baseline used before federated training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.synthetic_data import make_synthetic_mri_cases, split_cases


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Convert unrestricted scores to probabilities between zero and one."""
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def dice_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Measure segmentation overlap; one means the masks match perfectly."""
    predicted_mask = predictions.astype(bool)
    target_mask = targets.astype(bool)
    intersection = np.logical_and(predicted_mask, target_mask).sum()
    total = predicted_mask.sum() + target_mask.sum()
    return 1.0 if total == 0 else float((2 * intersection) / total)


@dataclass
class IntensitySegmentationBaseline:
    """One-feature logistic classifier for verifying the segmentation workflow."""

    weight: float = 0.0
    bias: float = 0.0

    def probabilities(self, scans: np.ndarray) -> np.ndarray:
        return sigmoid(self.weight * scans + self.bias)

    def predict(self, scans: np.ndarray) -> np.ndarray:
        return self.probabilities(scans) >= 0.5

    def fit(self, scans: np.ndarray, masks: np.ndarray, epochs: int = 100, learning_rate: float = 0.8) -> list[float]:
        """Learn intensity-to-mask mapping with full-batch gradient descent."""
        features = scans.reshape(-1)
        targets = masks.reshape(-1).astype(np.float32)
        losses: list[float] = []

        # Tumor voxels are uncommon. Without this weight, the model can minimize
        # loss by predicting background everywhere and still appear successful.
        positive_weight = float((1.0 - targets.mean()) / targets.mean())

        for _ in range(epochs):
            probabilities = sigmoid(self.weight * features + self.bias)
            sample_weights = np.where(targets == 1.0, positive_weight, 1.0)
            error = (probabilities - targets) * sample_weights
            self.weight -= learning_rate * float(np.mean(error * features))
            self.bias -= learning_rate * float(np.mean(error))
            clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
            losses.append(
                float(
                    -np.mean(
                        sample_weights
                        * (targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped))
                    )
                )
            )

        return losses


def run_baseline() -> tuple[float, list[float]]:
    """Train and evaluate the first reproducible centralized experiment."""
    scans, masks = make_synthetic_mri_cases(case_count=24)
    train_scans, train_masks, test_scans, test_masks = split_cases(scans, masks)
    model = IntensitySegmentationBaseline()
    losses = model.fit(train_scans, train_masks)
    return dice_score(model.predict(test_scans), test_masks), losses


if __name__ == "__main__":
    score, losses = run_baseline()
    print(f"Final training loss: {losses[-1]:.4f}")
    print(f"Test Dice score: {score:.4f}")
