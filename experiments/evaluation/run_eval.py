"""
Unified evaluation entry point.

Loads trained models, runs all experiments, and produces result tables.
"""

import sys
import json
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.utils import get_device, load_checkpoint
from evaluation.condition_metrics import evaluate_condition_model
from evaluation.yield_metrics import evaluate_yield_field
from evaluation.route_metrics import scoring_method_comparison
from evaluation.active_metrics import compare_acquisition_methods


def print_results_table(results: dict, title: str = "Results"):
    """Pretty-print results as a table."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    # Flat dict
    if all(not isinstance(v, dict) for v in results.values()):
        max_key_len = max(len(k) for k in results.keys())
        for k, v in sorted(results.items()):
            if isinstance(v, float):
                print(f"  {k:<{max_key_len+2}} {v:.4f}")
            else:
                print(f"  {k:<{max_key_len+2}} {v}")

    # Nested dict (comparison table)
    else:
        methods = list(results.keys())
        if not methods:
            return

        metrics = list(results[methods[0]].keys())

        # Header
        col_width = max(max(len(m) for m in methods), 12)
        header = f"  {'Metric':<30}" + "".join(f"{m:>{col_width+2}}" for m in methods)
        print(header)
        print("  " + "-" * (len(header) - 2))

        for metric in metrics:
            row = f"  {metric:<30}"
            for method in methods:
                val = results[method].get(metric, "N/A")
                if isinstance(val, float):
                    row += f"{val:>{col_width+2}.4f}"
                else:
                    row += f"{str(val):>{col_width+2}}"
            print(row)

    print(f"{'='*60}\n")


def save_results(results: dict, path: Path):
    """Save results as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(convert(results), f, indent=2)
    print(f"Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Run evaluations")
    parser.add_argument("--experiment", choices=["condition", "yield", "route", "active", "all"], default="all")
    parser.add_argument("--results-dir", type=str, default="./results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment in ("condition", "all"):
        print("\n[Experiment 2] Condition Prediction Evaluation")
        print("Load trained models and run condition_metrics...")
        # Placeholder — actual model loading depends on trained checkpoints
        print("  (Run train_condition.py first)")

    if args.experiment in ("yield", "all"):
        print("\n[Experiment 3] Yield Field Evaluation")
        print("Load trained ensemble and run yield_metrics...")
        print("  (Run train_yield_field.py first)")

    if args.experiment in ("route", "all"):
        print("\n[Experiment 4] Route Scoring Evaluation")
        print("Load models and run route comparison...")
        print("  (Run train_condition.py and train_yield_field.py first)")

    if args.experiment in ("active", "all"):
        print("\n[Experiment 5] Active Learning Evaluation")
        print("Run active learning loops and compare...")
        print("  (Requires Buchwald HTE dataset)")

    print("\nTo run evaluations, ensure models are trained first.")
    print("See README.md for the full pipeline.")


if __name__ == "__main__":
    main()
