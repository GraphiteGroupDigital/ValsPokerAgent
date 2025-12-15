"""
Batch chip stack plotting utility.

Iterates over every JSON log under the logs directory (recursing into the seed
folders) and generates a chips-over-time line plot for each file. The plots are
saved under analysis/results/chips_over_time mirroring the logs/ directory
structure, e.g. logs/seed_6042/seating_0.json -> analysis/results/chips_over_time/seed_6042/seating_0.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Any, List, Tuple, Sequence

from analyze_log import extract_round_chips_series, RESULTS_DIR


def load_log(path: Path) -> Dict[str, Any]:
    with path.open("r") as infile:
        return json.load(infile)


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise SystemExit(
            "matplotlib is required for plotting. Install it via `pip install matplotlib`."
        ) from exc

    return plt


def save_chips_plot(
    series: Dict[str, Tuple[List[int], List[int]]],
    title: str,
    output_path: Path,
    dpi: int = 150,
) -> None:
    plt = require_matplotlib()

    plt.figure(figsize=(10, 6))
    for player, (xs, ys) in sorted(series.items()):
        if xs:
            plt.plot(xs, ys, linewidth=2, label=player)

    plt.xlabel("Round index (timestep // num_players)")
    plt.ylabel("Chips (stack)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def sanitize_filename(value: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value)
    return safe.strip("_") or "player"


def _extend_series(xs: Sequence[int], ys: Sequence[int], upto_round: int) -> List[int]:
    """Return stack values for each round 0..upto_round, carrying forward last value."""
    if not xs:
        return []

    values: List[int] = []
    last_value = ys[0]
    idx = 0
    for round_idx in range(upto_round + 1):
        while idx < len(xs) and xs[idx] <= round_idx:
            last_value = ys[idx]
            idx += 1
        values.append(last_value)
    return values


def _compute_average_series(
    series_list: List[Tuple[str, Tuple[List[int], List[int]]]]
) -> Tuple[List[int], List[float], List[float]]:
    valid_points = [points for _, points in series_list if points[0]]
    if not valid_points:
        return [], [], []

    max_round = max(xs[-1] for xs, _ in valid_points)
    if max_round < 0:
        return [], [], []

    totals = [0.0] * (max_round + 1)
    squared_totals = [0.0] * (max_round + 1)
    contributors = 0

    for xs, ys in valid_points:
        extended = _extend_series(xs, ys, max_round)
        if not extended:
            continue
        contributors += 1
        for i, value in enumerate(extended):
            totals[i] += value
            squared_totals[i] += value * value

    if contributors == 0:
        return [], [], []

    averages = [total / contributors for total in totals]
    variances = []
    for i in range(max_round + 1):
        mean = averages[i]
        mean_sq = squared_totals[i] / contributors
        variance = max(mean_sq - mean * mean, 0.0)
        variances.append(variance)

    rounds = list(range(max_round + 1))
    std_devs = [variance ** 0.5 for variance in variances]
    return rounds, averages, std_devs


def save_overlay_plots(
    per_player_series: Dict[str, List[Tuple[str, Tuple[List[int], List[int]]]]],
    output_dir: Path,
    dpi: int = 150,
    legend_threshold: int = 12,
) -> None:
    """Plot overlays where each player's trajectories from every log share one chart."""

    plt = require_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    for player, series_list in sorted(per_player_series.items()):
        if not series_list:
            continue

        plt.figure(figsize=(10, 6))
        show_legend = len(series_list) <= legend_threshold
        for label, (xs, ys) in series_list:
            if xs:
                plt.plot(
                    xs,
                    ys,
                    linewidth=1.3,
                    alpha=0.65,
                    label=label if show_legend else None,
                )

        plt.xlabel("Round index (timestep // num_players)")
        plt.ylabel("Chips (stack)")
        plt.title(f"{player} Chip Trajectories ({len(series_list)} logs)")
        plt.grid(True, linestyle="--", alpha=0.25)
        avg_line = None
        avg_xs, avg_ys, avg_std = _compute_average_series(series_list)
        if avg_xs:
            (avg_line,) = plt.plot(
                avg_xs,
                avg_ys,
                color="black",
                linewidth=3,
                label="Average",
                zorder=5,
            )
            lower = [y - s for y, s in zip(avg_ys, avg_std)]
            upper = [y + s for y, s in zip(avg_ys, avg_std)]
            plt.fill_between(
                avg_xs,
                lower,
                upper,
                color="gray",
                alpha=0.25,
                linewidth=0,
                label="+/- 1 std dev" if show_legend else None,
                zorder=4,
            )

        if show_legend:
            handles, labels = plt.gca().get_legend_handles_labels()
            if avg_line is not None and "+/- 1 std dev" not in labels:
                handles.append(avg_line)
                labels.append("Average")
            plt.legend(handles, labels, fontsize=7, loc="best")
        elif avg_line is not None:
            plt.legend([avg_line], ["Average"], fontsize=7, loc="best")
        plt.tight_layout()

        filename = sanitize_filename(player) + ".png"
        plt.savefig(output_dir / filename, dpi=dpi)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate chips-over-time plots for every poker log."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "logs",
        help="Directory containing JSON logs (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "chips_over_time",
        help="Directory where plots are written (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after plotting this many files (useful for smoke tests)",
    )
    parser.add_argument(
        "--model-overlays",
        action="store_true",
        help="Also produce one overlay chart per model with trajectories from every log.",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=RESULTS_DIR / "chips_over_time_overlays",
        help="Directory for overlay plots (default: %(default)s)",
    )
    args = parser.parse_args()

    logs_dir = args.logs_dir
    output_dir = args.output_dir

    if not logs_dir.exists():
        raise SystemExit(f"Logs directory {logs_dir} does not exist")

    log_files = sorted(p for p in logs_dir.rglob("*.json") if p.is_file())
    if not log_files:
        raise SystemExit(f"No JSON logs found in {logs_dir}")

    print(f"Found {len(log_files)} log files under {logs_dir}")

    failures = 0
    per_player_series: DefaultDict[
        str, List[Tuple[str, Tuple[List[int], List[int]]]]
    ] = defaultdict(list)
    for idx, log_path in enumerate(log_files, start=1):
        if args.limit is not None and idx > args.limit:
            break

        rel = log_path.relative_to(logs_dir)
        output_path = (output_dir / rel).with_suffix(".png")

        try:
            log = load_log(log_path)
            series = extract_round_chips_series(log)
        except Exception as exc:
            failures += 1
            print(f"[SKIP] {rel}: {exc}")
            continue

        if args.model_overlays:
            for player, points in series.items():
                per_player_series[player].append((str(rel), points))

        title = f"Chip Stacks Over Time - {rel}"
        save_chips_plot(series, title, output_path)
        print(f"[OK] {rel} -> {output_path}")

    complete = min(len(log_files), args.limit or len(log_files))
    if failures:
        print(
            f"Generated {complete - failures} plot(s) with {failures} file(s) skipped due to errors."
        )
    else:
        print(f"Generated {complete} plot(s) successfully.")

    if args.model_overlays:
        save_overlay_plots(per_player_series, args.overlay_dir)
        print(
            f"Overlay plots saved to {args.overlay_dir} "
            f"({len(per_player_series)} model charts)."
        )


if __name__ == "__main__":
    main()
