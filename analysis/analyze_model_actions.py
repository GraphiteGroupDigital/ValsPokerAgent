"""
Aggregate per-model action metadata across many poker logs.

For every model that appears in the specified logs, compute average counts and
percentages for every action type encountered (folds, checks, calls, bets,
raises, etc.) plus the average raise size (chips).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import statistics

from analyze_log import gather_seed_logs


DEFAULT_SEEDS: List[str] = [
    "final/seed_42",
    # "seed_42_variation",
    "seed_1042",
    "seed_2042",
    "seed_3042",
    "seed_4042",
    "seed_5042",
    "seed_6042",
    "seed_7042",
    "seed_8042",
    "seed_9042",
]


PLAYER_ALIAS = {
    "ZAI-GLM-4.6": "fireworks/glm-4p6",
}


def normalize_model_name(response: Dict[str, Any], player: str) -> str:
    name = response.get("model_name") or player or "Unknown"
    provider = response.get("provider", "")
    if player in PLAYER_ALIAS:
        return PLAYER_ALIAS[player]
    if provider:
        name = f"{provider}/{name.split('/')[-1]}"
    return name or "Unknown"


def extract_player_models(log: Dict[str, Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for hand in log.get("hands", []):
        for action in hand.get("actions", []):
            player = action.get("player")
            response = action.get("model_response")
            if player and response and player not in mapping:
                mapping[player] = normalize_model_name(response, player)
        if len(mapping) == len(log["session_info"]["player_names"]):
            break
    # Default unknown for players without actions
    for player in log["session_info"]["player_names"]:
        mapping.setdefault(player, "Unknown")
    return mapping


def extract_player_action_counts(log: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    players = log["session_info"]["player_names"]

    def new_player():
        return {
            "actions": defaultdict(float),
            "raise_amount_sum": 0.0,
            "raise_actions": 0,
            "action_opportunities": 0,
            "hands_seen": 0,
            "hands_played": 0,
        }

    counts: Dict[str, Dict[str, Any]] = {player: new_player() for player in players}

    for hand in log.get("hands", []):
        starting_state = hand.get("starting_state", {})
        first_action: Dict[str, Optional[str]] = {}
        for player in starting_state.keys():
            if player not in counts:
                continue
            counts[player]["hands_seen"] += 1
            first_action[player] = None

        for action in hand.get("actions", []):
            player = action.get("player")
            if player not in counts:
                continue
            action_type = action.get("action", {}).get("type", "unknown")
            counts[player]["action_opportunities"] += 1
            counts[player]["actions"][action_type] += 1
            # record first decision after cards dealt
            if player in first_action and first_action[player] is None:
                first_action[player] = action_type
            if action_type == "raise":
                counts[player]["raise_actions"] += 1
                amount = action.get("action", {}).get("amount")
                if amount is not None:
                    counts[player]["raise_amount_sum"] += float(amount)

        for player, action_type in first_action.items():
            if action_type and action_type != "fold":
                counts[player]["hands_played"] += 1

    return counts


def aggregate_model_action_stats(
    logs: List[Tuple[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    def new_model():
        return {
            "games": 0,
            "hands_seen": 0,
            "hands_played": 0,
            "actions": defaultdict(float),
            "raise_amount_sum": 0.0,
            "raise_actions": 0,
            "action_opportunities": 0,
        }

    stats: Dict[str, Dict[str, Any]] = defaultdict(new_model)

    for _, log in logs:
        player_models = extract_player_models(log)
        player_counts = extract_player_action_counts(log)

        for player, model in player_models.items():
            data = player_counts.get(player)
            if data is None:
                continue
            model_stats = stats[model]
            model_stats["games"] += 1
            model_stats["hands_seen"] += data["hands_seen"]
            model_stats["hands_played"] += data["hands_played"]
            model_stats["raise_actions"] += data["raise_actions"]
            model_stats["raise_amount_sum"] += data["raise_amount_sum"]
            model_stats["action_opportunities"] += data["action_opportunities"]
            for action_type, count in data["actions"].items():
                model_stats["actions"][action_type] += count

    return stats


def print_model_action_report(model_stats: Dict[str, Dict[str, Any]]):
    action_types = sorted(
        {atype for stats in model_stats.values() for atype in stats["actions"].keys()}
    )

    header = f"{'Model':40s} {'Games':>6s}"
    for action in action_types:
        header += f" {('Avg ' + action.title()):>12s}"
    header += f" {'Avg Raise Size':>16s}"
    for action in action_types:
        header += f" {('%' + action.title()):>9s}"
    header += f" {'Hands Played %':>15s}"

    print("=" * len(header))
    print("MODEL ACTION METRICS")
    print(header)
    print("=" * len(header))

    def sort_key(item):
        data = item[1]
        raises = data["actions"].get("raise", 0.0)
        return (-raises, item[0])

    for model, data in sorted(model_stats.items(), key=sort_key):
        games = data["games"] or 1
        opportunities = data["action_opportunities"] or 1
        row = f"{model:40s} {data['games']:6.0f}"
        for action in action_types:
            avg = data["actions"].get(action, 0.0) / games
            row += f" {avg:12.2f}"
        avg_raise_size = (
            data["raise_amount_sum"] / data["raise_actions"]
            if data["raise_actions"]
            else 0.0
        )
        row += f" {avg_raise_size:16.2f}"
        for action in action_types:
            pct = (data["actions"].get(action, 0.0) / opportunities) * 100
            row += f" {pct:9.2f}"
        hands_pct = (
            (data["hands_played"] / data["hands_seen"]) * 100 if data["hands_seen"] else 0
        )
        row += f" {hands_pct:15.2f}"
        print(row)

    print("=" * len(header))


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise SystemExit(
            "matplotlib is required for plotting. Install it via `pip install matplotlib`."
        ) from exc

    return plt


def plot_play_styles(
    model_stats: Dict[str, Dict[str, Any]],
    plots_dir: Path,
    folders: List[Path],
) -> None:
    plt = require_matplotlib()
    plots_dir.mkdir(parents=True, exist_ok=True)

    x_vals = []
    y_vals = []
    labels = []

    for model, data in model_stats.items():
        hands_seen = data["hands_seen"]
        opportunities = data["action_opportunities"]
        if hands_seen == 0 or opportunities == 0:
            continue
        hands_pct = (data["hands_played"] / hands_seen) * 100
        aggressive_actions = (
            data["actions"].get("raise", 0.0) + data["actions"].get("bet", 0.0)
        )
        aggressive_rate = aggressive_actions / opportunities
        x_vals.append(aggressive_rate * 100)
        y_vals.append(hands_pct)
        labels.append(model)

    if not x_vals:
        print("No data available to plot play styles.")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(x_vals, y_vals, color="#4C72B0", alpha=0.8)
    for x, y, label in zip(x_vals, y_vals, labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)

    plt.xlabel("Aggressive action percentage (bets + raises)")
    plt.ylabel("Hands played (% of hands where model didn't auto-fold)")
    title_suffix = ", ".join(folder.name for folder in folders)
    plt.title(f"Model Play Styles ({title_suffix})")
    plt.grid(True, linestyle="--", alpha=0.3)

    avg_x = statistics.mean(x_vals)
    avg_y = statistics.mean(y_vals)
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.axvline(avg_x, color="gray", linestyle="--", alpha=0.5)
    plt.axhline(avg_y, color="gray", linestyle="--", alpha=0.5)
    plt.text(avg_x * 1.01, ylim[1], "Loose", color="gray", va="top")
    plt.text(avg_x * 1.01, ylim[0], "Tight", color="gray", va="bottom")
    plt.text(xlim[1], avg_y * 1.01, "Aggressive", color="gray", ha="right")
    plt.text(xlim[0], avg_y * 1.01, "Passive", color="gray", ha="left")

    output_path = plots_dir / "model_play_styles.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved play style plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-model action metadata across multiple logs."
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
        help="Seed folders to include (default: %(default)s)",
    )
    parser.add_argument(
        "--plot-style",
        action="store_true",
        help="Generate play style scatter plot (hands played %% vs aggressive actions).",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory for saving plots (default: %(default)s)",
    )

    args = parser.parse_args()

    folder_paths, logs = gather_seed_logs(args.seeds, args.logs_dir)
    model_stats = aggregate_model_action_stats(logs)
    print_model_action_report(model_stats)
    if args.plot_style:
        plot_play_styles(model_stats, args.plots_dir, folder_paths)


if __name__ == "__main__":
    main()
