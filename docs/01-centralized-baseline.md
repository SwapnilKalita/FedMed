# Centralized Baseline

## Why this comes first

Before splitting data among hospitals, we need a trustworthy single-machine result. It gives us a reference point: later, federated training should approach this quality without centralizing the data.

## What the first baseline does

The prototype generates MRI-like 3D volumes with synthetic bright tumors. A logistic classifier learns one relationship: bright voxels are more likely to be tumor voxels. This is intentionally much smaller than the eventual 3D U-Net, but it verifies the important wiring:

- A scan and its segmentation mask remain paired.
- Train/test splitting happens by scan, preventing voxel-level leakage.
- Training minimizes class-balanced binary cross-entropy loss, so rare tumor voxels matter.
- Dice score measures how much predicted and true tumor masks overlap.

## Reading the metric

Dice is `2 * overlap / (predicted tumor voxels + true tumor voxels)`. A value near `1.0` is strong overlap; `0.0` means no overlap. We will record this centralized score before comparing it against the federated result.

## Run it

```powershell
python -m model.baseline
python -m unittest discover -s tests
```
