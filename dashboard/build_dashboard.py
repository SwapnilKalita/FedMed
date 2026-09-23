"""Inline results.json into template.html so the dashboard is one self-contained file.

Usage:
    python -m dashboard.build_dashboard
"""

from __future__ import annotations

from pathlib import Path

DASHBOARD_DIR = Path(__file__).parent


def build_dashboard() -> Path:
    results_path = DASHBOARD_DIR / "results.json"
    template_path = DASHBOARD_DIR / "template.html"
    output_path = DASHBOARD_DIR / "index.html"

    if not results_path.exists():
        raise FileNotFoundError("dashboard/results.json not found. Run `python run_experiment.py` first.")

    results_json = results_path.read_text()
    template = template_path.read_text()
    output_path.write_text(template.replace("__RESULTS_JSON__", results_json))
    return output_path


if __name__ == "__main__":
    path = build_dashboard()
    print(f"Dashboard written to {path}")
