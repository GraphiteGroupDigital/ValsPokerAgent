"""
Batch aggregate utility for multiple seed folders.

Loads every log from the specified seed directories, prints the aggregate seat
and model statistics, and (optionally) saves bar charts for the average chips
per seat/model with ±1 std-dev error bars.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import List, Dict, Any

from analyze_log import (
    analyze_folder_aggregate,
    collect_model_statistics,
    collect_seat_statistics,
    gather_seed_logs,
    RESULTS_DIR,
)

DEFAULT_SEEDS: List[str] = [
    "seed_42",
    "seed_42_variation",
    "seed_1042",
    "seed_3042",
    "seed_4042",
    "seed_5042",
    "seed_6042",
    "seed_7042",
    "seed_8042",
    "seed_9042",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate poker log statistics across multiple seed folders.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "logs",
        help="Base logs directory (default: %(default)s)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Seed folder names or paths to include (default: common seed list)",
    )
    parser.add_argument(
        "--aggregate-by",
        choices=["seat", "model", "all"],
        default="all",
        help="Aggregation mode (default: all)",
    )
    parser.add_argument(
        "--plot-bars",
        action="store_true",
        help="Generate bar charts for average chips per seat/model with error bars.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=RESULTS_DIR / "aggregate_bar_charts",
        help="Directory where plots will be saved (default: %(default)s)",
    )

    args = parser.parse_args()

    folder_paths, logs = gather_seed_logs(args.seeds, args.logs_dir)

    print("=" * 70)
    print(
        f"Analyzing {len(logs)} logs from {len(folder_paths)} seed folders "
        f"({', '.join(path.name for path in folder_paths)})"
    )
    print("=" * 70)
    print()

    analyze_folder_aggregate(logs, args.aggregate_by)

    if args.plot_bars:
        seat_stats = collect_seat_statistics(logs)
        model_stats = collect_model_statistics(logs)
        plot_average_chips_bars(seat_stats, model_stats, args.plots_dir)


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise SystemExit(
            "matplotlib is required for plotting. Install it via `pip install matplotlib`."
        ) from exc

    return plt


def compute_mean_std(values):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def plot_average_chips_bars(
    seat_stats: Dict[int, Dict[str, Any]],
    model_stats: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Create bar charts (with error bars) for seats and models."""

    plt = require_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    if seat_stats:
        seats = sorted(seat_stats.items())
        labels = [f"Seat {seat}" for seat, _ in seats]
        means = []
        stds = []
        for _, data in seats:
            mean, std = compute_mean_std(data["final_chips"])
            means.append(mean)
            stds.append(std)

        plt.figure(figsize=(10, 6))
        positions = range(len(labels))
        plt.bar(positions, means, yerr=stds, capsize=6, color="#4C72B0", alpha=0.85)
        plt.xticks(positions, labels)
        plt.ylabel("Average final chips")
        plt.title("Average Final Chips per Seat")
        plt.tight_layout()
        seat_path = output_dir / "average_chips_per_seat.png"
        plt.savefig(seat_path, dpi=150)
        plt.close()
        print(f"Saved seat bar chart to {seat_path}")
    else:
        print("No seat data available for plotting.")

    model_items = []
    for model, data in model_stats.items():
        chips = data["final_chips"]
        if chips:
            mean, std = compute_mean_std(chips)
            model_items.append((model, mean, std))

    if model_items:
        model_items.sort(key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in model_items]
        means = [item[1] for item in model_items]
        stds = [item[2] for item in model_items]

        plt.figure(figsize=(max(10, len(labels) * 0.6), 6))
        positions = range(len(labels))
        plt.bar(positions, means, yerr=stds, capsize=4, color="#55A868", alpha=0.85)
        plt.xticks(positions, labels, rotation=45, ha="right")
        plt.ylabel("Average final chips")
        plt.title("Average Final Chips per Model")
        plt.tight_layout()
        model_path = output_dir / "average_chips_per_model.png"
        plt.savefig(model_path, dpi=150)
        plt.close()
        print(f"Saved model bar chart to {model_path}")
    else:
        print("No model data available for plotting.")


if __name__ == "__main__":
    main()
