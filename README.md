# FedMed

Privacy-preserving federated learning for MRI brain tumor segmentation.

FedMed simulates three hospital nodes training a shared 3D segmentation model without transferring raw patient data. Each node trains locally, sends protected model updates to a central coordinator, and receives an improved global model for the next round.

## Project goals

- Establish a centralized 3D U-Net baseline using a public MRI dataset such as BraTS.
- Partition data across three simulated hospitals and coordinate rounds with Flower.
- Aggregate client updates with Federated Averaging (FedAvg).
- Protect updates with homomorphic encryption and differential privacy.
- Expose training loss, accuracy, and segmentation results in a monitoring dashboard.

## Planned architecture

```text
Hospital A ----\
Hospital B ----- > Flower coordinator -> global PyTorch/MONAI model
Hospital C ----/          |
                         +-> metrics API / dashboard
```

Raw MRI scans remain within each hospital node. The coordinator receives model updates only.

## Milestones

1. Build a centralized 3D U-Net baseline and three local hospital-node stubs.
2. Implement Flower-based federated training and secure node communication.
3. Add encrypted aggregation with TenSEAL and live training metrics.
4. Add differential privacy and a dashboard for convergence and segmentation masks.

## Repository layout

```text
fedmed-federated-learning/
  server/       # Federated coordinator and aggregation strategy
  clients/      # Isolated hospital training nodes
  model/        # PyTorch/MONAI segmentation model and training utilities
  dashboard/    # Future React metrics dashboard
  docs/         # Architecture decisions and experiment notes
  tests/        # Unit and integration tests
```

## Safety and privacy

This repository will use public or synthetic data only. It is a portfolio prototype, not a clinical system; it must not be used for medical decisions or real patient records.

## Getting started

The initial commit defines the project boundary and directory structure. The next implementation step is to create the centralized baseline and a reproducible public-data pipeline.
