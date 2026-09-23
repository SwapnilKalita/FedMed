"""Deterministic MRI-like volumes for testing the training pipeline safely."""

from __future__ import annotations

import numpy as np


def make_synthetic_mri_cases(
    case_count: int, volume_size: int = 16, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """Create noisy 3D scans with bright spherical tumor masks.

    These are synthetic scans only. They let us validate the full learning flow
    without downloading or handling patient data.
    """
    if case_count < 1 or volume_size < 8:
        raise ValueError("case_count must be positive and volume_size must be at least 8")

    generator = np.random.default_rng(seed)
    axis = np.arange(volume_size, dtype=np.float32)
    grid = np.meshgrid(axis, axis, axis, indexing="ij")
    scans = np.empty((case_count, volume_size, volume_size, volume_size), dtype=np.float32)
    masks = np.empty_like(scans, dtype=bool)

    for index in range(case_count):
        center = generator.uniform(4, volume_size - 4, size=3)
        radius = generator.uniform(2.0, 3.0)
        distance_squared = sum((coordinate - origin) ** 2 for coordinate, origin in zip(grid, center))
        mask = distance_squared <= radius**2
        background = generator.normal(loc=0.15, scale=0.05, size=mask.shape)
        tumor_signal = generator.normal(loc=0.75, scale=0.05, size=mask.shape)
        scans[index] = np.where(mask, tumor_signal, background)
        masks[index] = mask

    return scans, masks


def split_cases(
    scans: np.ndarray, masks: np.ndarray, training_fraction: float = 0.75
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split whole scans, never individual voxels, to avoid data leakage."""
    if scans.shape != masks.shape:
        raise ValueError("scans and masks must have matching shapes")
    split_index = max(1, min(len(scans) - 1, round(len(scans) * training_fraction)))
    return scans[:split_index], masks[:split_index], scans[split_index:], masks[split_index:]
