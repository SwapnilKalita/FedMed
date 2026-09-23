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

1. ✅ Centralized baseline and three simulated hospital nodes (`model/hospital_node.py`).
2. ✅ Federated training loop with FedAvg, uneven data partitions, and
   resilience to a hospital dropping offline mid-training (`server/coordinator.py`).
3. ✅ Secure aggregation so the coordinator never sees an individual hospital's
   update, plus per-round metrics feeding the dashboard (`server/secure_aggregation.py`).
4. ✅ Differential privacy noise and a dashboard comparing centralized vs.
   federated vs. federated+DP convergence (`model/differential_privacy.py`, `dashboard/`).

This sandbox has no network access and cannot install `torch`, `flwr`, or
`tenseal`. Every milestone above is implemented with the *same algorithms*
(FedAvg, a Bonawitz-style secure aggregation protocol, the Gaussian DP
mechanism) in dependency-free NumPy, so the wiring is fully verified end to
end. See `docs/03-secure-aggregation.md` and `docs/04-differential-privacy.md`
for exactly what's simplified and what a production port to
PyTorch/MONAI + Flower + TenSEAL would change.

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

```bash
pip install -e .          # numpy only, no other dependencies needed
python -m unittest discover -s tests   # 19 tests, all pure-Python/NumPy
python run_experiment.py               # trains all three variants, writes dashboard/results.json
python -m dashboard.build_dashboard    # bakes results.json into dashboard/index.html
```

Then open `dashboard/index.html` in a browser to see:
- final Dice score for centralized vs. federated vs. federated+DP,
- each hospital's (uneven) sample count,
- per-round Dice and training-loss curves,
- a sample ground-truth tumor mask.

## Porting to the real stack

Every module documents, in its docstring, exactly which real library it
stands in for and why (no network access here to install them):

| This repo | Production equivalent |
|---|---|
| `IntensitySegmentationBaseline` (1-parameter logistic model) | 3D U-Net (PyTorch/MONAI) |
| `server/federated_averaging.py` + `server/coordinator.py` | Flower `FedAvg` strategy + server loop |
| `server/secure_aggregation.py` (pairwise masking) | TenSEAL homomorphic encryption / Flower `SecAgg+` |
| `model/differential_privacy.py` (Gaussian mechanism) | Opacus / TensorFlow Privacy DP-SGD with an accountant |
| `dashboard/` (static HTML + Chart.js) | React/Recharts dashboard fed by a live WebSocket |

None of the module *interfaces* would need to change -- `run_federated_training`
still takes hospital nodes and returns a global model either way.


## Demo, dashboard, and CI

This repository includes a local interactive dashboard (dark theme) that visualizes the demo wiring and client updates.

Quick start — run the dashboard locally (recommended)

1. From the repository root open PowerShell and create/activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install the minimal dependencies (or full requirements):

```powershell
pip install -r requirements.txt
# or minimal: pip install flask flask-cors
```

3. Start the demo servers (helper script opens two terminals for you):

```powershell
.\start_demo.ps1
```

4. Open the dashboard in your browser:

- Local static URL: `http://localhost:8000/dashboard.html`
- (Alternative) open the file directly: `fedmed/docs/dashboard.html`

Notes

- The helper starts the Flask demo API on port 5000 and a static HTTP server serving `fedmed/docs` on port 8000. The dashboard fetches demo data from `http://localhost:5000/run_demo`.
- If you prefer to run servers manually:

```powershell
# start API (from repo root)
python -m fedmed.api.run_demo_api
# serve docs folder
python -m http.server 8000 --directory ".\fedmed\docs"
# then open: http://localhost:8000/dashboard.html
```

Assets and artifacts

- Dashboard page: `fedmed/docs/dashboard.html`
- One-page summary: `fedmed/docs/one_pager.html` and PDF `fedmed/docs/one_pager.pdf`
- Add screenshots or demo GIFs under `fedmed/docs/assets/` (create the folder) and reference them from the README if needed.

Continuous integration

- A lightweight smoke-test workflow runs on pushes and PRs to validate imports and the basic demo wiring: `.github/workflows/smoke-test.yml`.

![Smoke test status](https://github.com/SwapnilKalita/FedMed/actions/workflows/smoke-test.yml/badge.svg)

