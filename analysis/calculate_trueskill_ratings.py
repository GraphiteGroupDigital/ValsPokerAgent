#!/usr/bin/env python3

"""
Calculate TrueSkill ratings for poker models based on game results from log files.

Uses TrueSkill for multi-player poker games, ranking by chip count with elimination order tiebreaker.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

try:
    import trueskill
except ImportError:
    print("Error: trueskill package not found.")
    print("Install it with: pip install trueskill")
    raise SystemExit(1)

import matplotlib.pyplot as plt
import numpy as np

from analyze_log import calculate_cost


def load_log_file(log_path: Path) -> Dict[str, Any]:
    """Load a single JSON log file."""
    try:
        with open(log_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {log_path}: {e}")
        return None


def gather_all_logs(
    logs_dir: Path, seed_folders: List[str] = None, exclude_dirs: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Gather all log files from specified seed folders or recursively from logs directory.

    Args:
        logs_dir: Base logs directory
        seed_folders: List of seed folder names to include (if None, searches recursively)
        exclude_dirs: List of directory names to exclude (default: ['archived'])

    Returns:
        List of parsed log dictionaries
    """
    all_logs = []

    if exclude_dirs is None:
        exclude_dirs = ["archived"]

    if seed_folders is not None:
        # Original behavior: search specific seed folders
        for seed_name in seed_folders:
            seed_path = logs_dir / seed_name
            if not seed_path.exists():
                print(f"Warning: Seed folder not found: {seed_path}")
                continue

            # Find all JSON files in this seed folder
            json_files = sorted(seed_path.glob("*.json"))
            for json_file in json_files:
                log = load_log_file(json_file)
                if log:
                    # Attach path metadata for debugging/analysis
                    log["_log_path"] = str(json_file.relative_to(logs_dir))
                    all_logs.append(log)
    else:
        # New behavior: recursively find all JSON files, excluding specified directories
        print(f"Searching recursively in {logs_dir} for .json files...")
        print(f"Excluding directories: {', '.join(exclude_dirs)}")

        for json_file in sorted(logs_dir.rglob("*.json")):
            # Check if any excluded directory is in the path
            if any(excluded in json_file.parts for excluded in exclude_dirs):
                continue

            log = load_log_file(json_file)
            if log:
                # Attach path metadata for debugging/analysis
                rel_path = json_file.relative_to(logs_dir)
                log["_log_path"] = str(rel_path)
                all_logs.append(log)
                # Print relative path for better readability
                print(f"  Loaded: {rel_path}")

    return all_logs


def extract_game_rankings(log: Dict[str, Any]) -> List[Tuple[str, int, int]]:
    """Extract player rankings from game log. Returns (model_name, final_chips, rank) tuples."""
    if "hands" not in log or not log["hands"]:
        return []

    players = log["session_info"]["player_names"]
    last_hand = log["hands"][-1]
    ending_stacks = last_hand.get("ending_stacks", {})

    # Build player data with elimination info
    player_data = []
    for player in players:
        final_stack = ending_stacks.get(player, 0)

        # Find elimination hand (first hand where player has 0 chips)
        elimination_hand = None
        elimination_round = None

        if final_stack == 0:
            for hand in log["hands"]:
                hand_ending_stacks = hand.get("ending_stacks", {})
                if hand_ending_stacks.get(player, 0) == 0:
                    elimination_hand = hand.get("hand_number", 0)

                    # Find the round within the hand where they were eliminated
                    last_action_with_chips = None
                    for action in hand.get("actions", []):
                        if (
                            action.get("player") == player
                            and action.get("stack_after", 0) > 0
                        ):
                            last_action_with_chips = action

                    if last_action_with_chips:
                        elimination_round = last_action_with_chips.get("timestep", 0)
                    break

        player_data.append(
            {
                "player": player,
                "final_stack": final_stack,
                "elimination_hand": elimination_hand or float("inf"),
                "elimination_round": elimination_round or 0,
            }
        )

    # Sort by: final_stack (desc), elimination_hand (desc), elimination_round (desc)
    player_data.sort(
        key=lambda x: (
            -x["final_stack"],
            -(x["elimination_hand"] if x["final_stack"] == 0 else 0),
            -(x["elimination_round"] if x["final_stack"] == 0 else 0),
        )
    )

    # Assign ranks
    rankings = []
    for rank, data in enumerate(player_data):
        rankings.append((data["player"], data["final_stack"], rank))

    return rankings


def calculate_trueskill_ratings(
    all_logs: List[Dict[str, Any]],
    mu: float = 25.0,
    sigma: float = 8.333,
    beta: float = 4.166,
    tau: float = 0.0833,
    draw_probability: float = 0.0,
) -> Dict[str, trueskill.Rating]:
    """Calculate TrueSkill ratings for all models across games."""
    # Configure TrueSkill environment
    env = trueskill.TrueSkill(
        mu=mu,
        sigma=sigma,
        beta=beta,
        tau=tau,
        draw_probability=draw_probability,
    )
    trueskill.setup(
        mu=mu, sigma=sigma, beta=beta, tau=tau, draw_probability=draw_probability
    )

    # Initialize ratings dictionary
    ratings: Dict[str, trueskill.Rating] = {}

    # Track game count per model
    game_counts: Dict[str, int] = defaultdict(int)

    # Process each game
    for game_num, log in enumerate(all_logs, 1):
        rankings = extract_game_rankings(log)
        if not rankings:
            continue

        # TEMP DEBUG: track where Deepseek-V3.2 appears
        deepseek_entries = [
            (player, chips, rank)
            for (player, chips, rank) in rankings
            if player == "Deepseek-V3.2"
        ]
        if deepseek_entries:
            log_id = log.get("_log_path", f"game_{game_num}")
            # print(
            #     f"[DEBUG Deepseek-V3.2] Appears {len(deepseek_entries)} time(s) in log: {log_id}"
            # )

        # Build rating groups for this game
        # Each player is their own team (FFA format)
        rating_groups = []
        ranks = []

        for player, chips, rank in rankings:
            # Initialize rating if this is player's first game
            if player not in ratings:
                ratings[player] = env.create_rating(mu=mu, sigma=sigma)

            rating_groups.append((ratings[player],))
            ranks.append(rank)
            game_counts[player] += 1

        # Update ratings based on game outcome
        # rate() returns list of rating tuples (one per team)
        updated_rating_groups = env.rate(rating_groups, ranks=ranks)

        # Extract updated ratings
        for i, (player, chips, rank) in enumerate(rankings):
            ratings[player] = updated_rating_groups[i][0]

    return ratings, game_counts


def scale_rating_to_1000(
    mu: float, sigma: float, scale: bool = True
) -> Tuple[float, float]:
    """Scale TrueSkill rating to 1000-baseline (ELO-like) system."""
    if not scale:
        return mu, sigma

    # Scale factor: 40 points per TrueSkill unit
    # Baseline: TrueSkill 25 = 1000
    scaled_mu = (mu - 25.0) * 40.0 + 1000.0
    scaled_sigma = sigma * 40.0  # Scale uncertainty proportionally

    return scaled_mu, scaled_sigma


def calculate_model_costs(
    all_logs: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Calculate total costs and hands played for each model."""
    if calculate_cost is None:
        print(
            "Warning: Cost calculation not available (analyze_log.calculate_cost not imported)"
        )
        return {}

    model_costs: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"total_cost": 0.0, "total_hands": 0}
    )

    for log in all_logs:
        players = log["session_info"]["player_names"]

        # Track which hands each player participated in
        player_hands: Dict[str, int] = {p: 0 for p in players}

        for hand in log.get("hands", []):
            # Count participation in this hand (player was dealt in)
            for player in hand.get("starting_state", {}).keys():
                player_hands[player] += 1

            # Calculate costs for each action using imported function
            for action in hand.get("actions", []):
                player = action.get("player")
                model_response = action.get("model_response", {})

                if model_response:
                    cost = calculate_cost(model_response, "total")
                    if cost > 0:
                        model_costs[player]["total_cost"] += cost

        # Update total hands played for each player
        for player, hands in player_hands.items():
            model_costs[player]["total_hands"] += hands

    # Calculate cost per hand
    for model in model_costs:
        total_cost = model_costs[model]["total_cost"]
        total_hands = model_costs[model]["total_hands"]
        if total_hands > 0:
            model_costs[model]["cost_per_hand"] = total_cost / total_hands
        else:
            model_costs[model]["cost_per_hand"] = 0.0

    return dict(model_costs)


def get_model_statistics(
    ratings: Dict[str, trueskill.Rating],
    game_counts: Dict[str, int],
    all_logs: List[Dict[str, Any]],
    scale_ratings: bool = True,
    include_costs: bool = False,
) -> List[Dict[str, Any]]:
    """Compute comprehensive statistics for each model."""
    # Gather chip statistics per model
    model_chips: Dict[str, List[int]] = defaultdict(list)
    model_ranks: Dict[str, List[int]] = defaultdict(list)

    for log in all_logs:
        rankings = extract_game_rankings(log)
        for player, chips, rank in rankings:
            model_chips[player].append(chips)
            model_ranks[player].append(rank + 1)  # Convert to 1-indexed

    # Calculate costs if requested
    model_costs = {}
    if include_costs:
        model_costs = calculate_model_costs(all_logs)

    # Build statistics list
    stats = []
    for model, rating in ratings.items():
        chips = model_chips[model]
        ranks = model_ranks[model]

        # Apply scaling if requested
        scaled_mu, scaled_sigma = scale_rating_to_1000(
            rating.mu, rating.sigma, scale_ratings
        )

        model_stat = {
            "model": model,
            "rating_mu": scaled_mu,
            "rating_sigma": scaled_sigma,
            "conservative_rating": scaled_mu - 1 * scaled_sigma,
            "games_played": game_counts[model],
            "avg_chips": statistics.mean(chips) if chips else 0,
            "std_chips": statistics.stdev(chips) if len(chips) > 1 else 0,
            "avg_rank": statistics.mean(ranks) if ranks else 0,
            "median_rank": statistics.median(ranks) if ranks else 0,
            "best_rank": min(ranks) if ranks else 0,
            "worst_rank": max(ranks) if ranks else 0,
        }

        # Add cost statistics if available
        if model in model_costs:
            model_stat["total_cost"] = model_costs[model]["total_cost"]
            model_stat["total_hands"] = model_costs[model]["total_hands"]
            model_stat["cost_per_hand"] = model_costs[model]["cost_per_hand"]

        stats.append(model_stat)

    # Sort by conservative rating (mu - 1*sigma) for more reliable ranking
    stats.sort(key=lambda x: x["conservative_rating"], reverse=True)

    return stats


def print_rankings_table(
    stats: List[Dict[str, Any]], scaled: bool = True, total_games: int | None = None
):
    """Print formatted table of model rankings."""
    print("\n" + "=" * 120)
    if scaled:
        print("TRUESKILL RANKINGS (1000-Scale, Poker Multi-Player)")
    else:
        print("TRUESKILL RANKINGS (Poker Multi-Player)")
    print("=" * 120)
    print(
        f"{'Rank':<6} {'Model':<35} {'Rating':<12} {'±σ':<8} {'Games':<7} "
        f"{'Avg Chips':<11} {'Avg Rank':<10} {'Best/Worst':<12}"
    )
    print("-" * 120)

    for rank, model_stats in enumerate(stats, 1):
        model = model_stats["model"]
        mu = model_stats["rating_mu"]
        sigma = model_stats["rating_sigma"]
        games = model_stats["games_played"]
        avg_chips = model_stats["avg_chips"]
        avg_rank = model_stats["avg_rank"]
        best = model_stats["best_rank"]
        worst = model_stats["worst_rank"]

        print(
            f"{rank:<6} {model:<35} {mu:>7.2f} {sigma:>10.2f} {games:>7} "
            f"{avg_chips:>11.1f} {avg_rank:>10.2f} {best:>4}/{worst:<6}"
        )

    print("=" * 120)

    # If total_games not provided, fall back to the maximum games_played across models
    if total_games is None:
        total_games = max((s["games_played"] for s in stats), default=0)

    print(f"\nTotal models ranked: {len(stats)}")
    print(f"Total games analyzed: {total_games}")
    if scaled:
        print("\nNote: Rating = Scaled TrueSkill (1000 = average, scale: TrueSkill×40)")
        print("      ±σ = uncertainty (lower is more confident)")
        print("      Conservative rating = μ - 1σ (used for ranking)")
        print("      Original TrueSkill 25 → 1000, 27 → 1080, 30 → 1200")
    else:
        print("\nNote: Rating = TrueSkill μ (mean skill estimate)")
        print("      ±σ = uncertainty (lower is more confident)")
        print("      Conservative rating = μ - 1σ (used for ranking)")
    print()


def save_rankings_json(
    stats: List[Dict[str, Any]],
    output_path: Path,
    total_games: int,
    scaled: bool = True,
    ratings: Dict[str, trueskill.Rating] = None,
):
    """Save rankings to JSON file."""
    output_data = {
        "rankings": [
            {
                "rank": rank,
                "model": model_stats["model"],
                "rating": float(model_stats["rating_mu"]),
                "rating_sigma": float(model_stats["rating_sigma"]),
                "conservative_rating": float(model_stats["conservative_rating"]),
                "games_played": model_stats["games_played"],
                "average_chips": float(model_stats["avg_chips"]),
                "std_chips": float(model_stats["std_chips"]),
                "average_rank": float(model_stats["avg_rank"]),
                "median_rank": float(model_stats["median_rank"]),
                "best_rank": model_stats["best_rank"],
                "worst_rank": model_stats["worst_rank"],
            }
            for rank, model_stats in enumerate(stats, 1)
        ],
        "total_games": total_games,
        "total_models": len(stats),
        "rating_system": {
            "type": "TrueSkill",
            "scaled": scaled,
            "scale_formula": "(mu - 25) * 40 + 1000" if scaled else "raw TrueSkill",
            "baseline": 1000 if scaled else 25,
        },
        "trueskill_params": {
            "mu": 25.0,
            "sigma": 8.333,
            "beta": 4.166,
            "tau": 0.0833,
            "draw_probability": 0.0,
        },
    }

    # Add raw TrueSkill ratings if provided (for matchmaking)
    if ratings is not None:
        raw_ratings = {}
        for model, rating in ratings.items():
            raw_ratings[model] = {
                "mu": float(rating.mu),
                "sigma": float(rating.sigma),
            }
        output_data["raw_trueskill_ratings"] = raw_ratings

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Rankings saved to {output_path}")


def plot_ratings(stats: List[Dict[str, Any]], output_path: Path, scaled: bool = True):
    """Create visualization of TrueSkill ratings with error bars."""
    # Sort by conservative rating for plotting
    sorted_stats = sorted(stats, key=lambda x: x["conservative_rating"], reverse=True)

    models = [s["model"] for s in sorted_stats]
    mus = [s["rating_mu"] for s in sorted_stats]
    sigmas = [s["rating_sigma"] for s in sorted_stats]

    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(models) * 0.6), 8))

    positions = range(len(models))

    # Determine baseline for horizontal line
    baseline = 1000.0 if scaled else 25.0

    # Plot ratings with error bars (±1σ for conservative rating)
    ax.errorbar(
        positions,
        mus,
        yerr=[1 * s for s in sigmas],
        fmt="o",
        capsize=5,
        capthick=2,
        markersize=8,
        linewidth=2,
        color="#2E86AB",
        ecolor="#A23B72",
        label="Rating (μ ± 1σ)",
    )

    # Customize plot
    ax.set_xticks(positions)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ylabel = "Rating (1000-scale)" if scaled else "TrueSkill Rating"
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    title = (
        "Poker Model Ratings with Uncertainty"
        if scaled
        else "Poker Model TrueSkill Ratings with Uncertainty"
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=10)

    # Add horizontal line at initial rating
    ax.axhline(
        y=baseline,
        color="gray",
        linestyle=":",
        alpha=0.5,
        label=f"Average ({baseline:.0f})",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Rating plot saved to {output_path}")


def plot_rating_vs_chips(stats: List[Dict[str, Any]], output_path: Path):
    """Create scatter plot of TrueSkill rating vs average chips."""
    sorted_stats = sorted(stats, key=lambda x: x["conservative_rating"], reverse=True)

    models = [s["model"] for s in sorted_stats]
    mus = [s["rating_mu"] for s in sorted_stats]
    avg_chips = [s["avg_chips"] for s in sorted_stats]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

    ax.scatter(
        mus, avg_chips, c=colors, s=150, alpha=0.7, edgecolors="black", linewidth=2
    )

    # Add trend line
    z = np.polyfit(mus, avg_chips, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(mus), max(mus), 100)
    ax.plot(
        x_trend,
        p(x_trend),
        "r--",
        alpha=0.8,
        linewidth=2.5,
        label=f"Trend: y={z[0]:.1f}x{z[1]:+.1f}",
    )

    # Calculate correlation
    correlation = np.corrcoef(mus, avg_chips)[0, 1]
    ax.text(
        0.05,
        0.95,
        f"Correlation: {correlation:.3f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        fontweight="bold",
    )

    ax.set_xlabel("TrueSkill Rating (μ)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Average Final Chips", fontsize=13, fontweight="bold")
    ax.set_title(
        "TrueSkill Rating vs Chip Performance", fontsize=15, fontweight="bold", pad=20
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=11, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Rating vs Chips plot saved to {output_path}")


def plot_rating_vs_rank(stats: List[Dict[str, Any]], output_path: Path):
    """Create scatter plot of TrueSkill rating vs average rank."""
    sorted_stats = sorted(stats, key=lambda x: x["conservative_rating"], reverse=True)

    models = [s["model"] for s in sorted_stats]
    mus = [s["rating_mu"] for s in sorted_stats]
    avg_ranks = [s["avg_rank"] for s in sorted_stats]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

    ax.scatter(
        mus, avg_ranks, c=colors, s=150, alpha=0.7, edgecolors="black", linewidth=2
    )

    # Add trend line
    z_rank = np.polyfit(mus, avg_ranks, 1)
    p_rank = np.poly1d(z_rank)
    x_trend = np.linspace(min(mus), max(mus), 100)
    ax.plot(
        x_trend,
        p_rank(x_trend),
        "r--",
        alpha=0.8,
        linewidth=2.5,
        label=f"Trend: y={z_rank[0]:.2f}x{z_rank[1]:+.1f}",
    )

    correlation_rank = np.corrcoef(mus, avg_ranks)[0, 1]
    ax.text(
        0.05,
        0.95,
        f"Correlation: {correlation_rank:.3f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        fontweight="bold",
    )

    ax.set_xlabel("TrueSkill Rating (μ)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Average Final Rank", fontsize=13, fontweight="bold")
    ax.set_title(
        "TrueSkill Rating vs Average Rank (Lower is Better)",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    ax.invert_yaxis()  # Lower rank is better
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=11, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Rating vs Rank plot saved to {output_path}")


def plot_confidence_vs_games(stats: List[Dict[str, Any]], output_path: Path):
    """Create scatter plot of uncertainty vs games played."""
    sorted_stats = sorted(stats, key=lambda x: x["conservative_rating"], reverse=True)

    games_played = [s["games_played"] for s in sorted_stats]
    sigmas = [s["rating_sigma"] for s in sorted_stats]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.plasma(np.linspace(0, 1, len(games_played)))

    ax.scatter(
        games_played,
        sigmas,
        c=colors,
        s=150,
        alpha=0.7,
        edgecolors="black",
        linewidth=2,
    )

    ax.set_xlabel("Games Played", fontsize=13, fontweight="bold")
    ax.set_ylabel("Rating Uncertainty (σ)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Confidence Increases with More Games", fontsize=15, fontweight="bold", pad=20
    )
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add text explanation
    ax.text(
        0.95,
        0.95,
        "Lower σ = Higher Confidence",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Confidence vs Games plot saved to {output_path}")


def plot_top_models_comparison(
    stats: List[Dict[str, Any]], output_path: Path, top_n: int = 8
):
    """Create horizontal bar chart of top models."""
    sorted_stats = sorted(stats, key=lambda x: x["conservative_rating"], reverse=True)

    top_n = min(top_n, len(sorted_stats))
    top_models = [s["model"] for s in sorted_stats[:top_n]]
    top_mus = [s["rating_mu"] for s in sorted_stats[:top_n]]
    top_sigmas = [s["rating_sigma"] for s in sorted_stats[:top_n]]

    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.6)))

    bar_positions = range(top_n)
    ax.barh(
        bar_positions,
        top_mus,
        xerr=[1 * s for s in top_sigmas],
        capsize=5,
        color="#55A868",
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
    )

    ax.axvline(
        x=25.0,
        color="gray",
        linestyle=":",
        alpha=0.6,
        linewidth=2.5,
        label="Initial Rating (25.0)",
    )

    ax.set_yticks(bar_positions)
    ax.set_yticklabels(top_models, fontsize=11)
    ax.set_xlabel("TrueSkill Rating (μ ± 1σ)", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Top {top_n} Models by Conservative Rating",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.legend(fontsize=11, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Top models comparison plot saved to {output_path}")


def plot_pareto_frontier(stats: List[Dict[str, Any]], output_path: Path):
    """Create Pareto frontier plot comparing cost efficiency vs TrueSkill rating."""
    # Filter models that have cost data
    models_with_costs = [
        s for s in stats if "cost_per_hand" in s and s["cost_per_hand"] > 0
    ]

    if not models_with_costs:
        print("Warning: No models with cost data available for Pareto plot")
        return

    models = [s["model"] for s in models_with_costs]
    ratings = [s["conservative_rating"] for s in models_with_costs]
    costs_per_hand = [s["cost_per_hand"] for s in models_with_costs]

    # Calculate Pareto frontier
    # A point is on the Pareto frontier if no other point is better in both dimensions
    # (higher rating AND lower cost)
    pareto_indices = []
    for i in range(len(models)):
        is_pareto = True
        for j in range(len(models)):
            if i != j:
                # Check if point j dominates point i
                if ratings[j] >= ratings[i] and costs_per_hand[j] <= costs_per_hand[i]:
                    if ratings[j] > ratings[i] or costs_per_hand[j] < costs_per_hand[i]:
                        is_pareto = False
                        break
        if is_pareto:
            pareto_indices.append(i)

    # Sort Pareto points by rating for drawing the frontier line
    pareto_indices_sorted = sorted(pareto_indices, key=lambda i: ratings[i])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot all models
    scatter = ax.scatter(
        costs_per_hand,
        ratings,
        s=150,
        alpha=0.6,
        edgecolors="black",
        linewidth=1.5,
        c=ratings,
        cmap="RdYlGn",
        zorder=2,
    )

    # Draw Pareto frontier line
    if len(pareto_indices_sorted) > 1:
        frontier_costs = [costs_per_hand[i] for i in pareto_indices_sorted]
        frontier_ratings = [ratings[i] for i in pareto_indices_sorted]
        ax.plot(
            frontier_costs,
            frontier_ratings,
            "b-",
            alpha=0.8,
            linewidth=3,
            label="Pareto Frontier",
            zorder=3,
        )

    # Annotate all models
    for i, (model, cost, rating) in enumerate(zip(models, costs_per_hand, ratings)):
        # Shorten model name if too long
        display_name = model if len(model) <= 20 else model[:17] + "..."

        # Use yellow background for Pareto frontier models
        bgcolor = "yellow" if i in pareto_indices else "white"

        ax.annotate(
            display_name,
            (cost, rating),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=bgcolor,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            ),
            zorder=4,
        )

    # Formatting
    ax.set_xlabel("Cost per Hand ($)", fontsize=13, fontweight="bold")
    ax.set_ylabel("TrueSkill Rating (Conservative)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Pareto Frontier: Model Performance vs Cost Efficiency",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=11, loc="lower right")

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("TrueSkill Rating", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Pareto frontier plot saved to {output_path}")
    print(f"\nPareto-optimal models ({len(pareto_indices)}):")
    for i in pareto_indices_sorted:
        print(
            f"  • {models[i]}: Rating={ratings[i]:.1f}, Cost/Hand=${costs_per_hand[i]:.6f}"
        )


def plot_comprehensive_analysis(stats: List[Dict[str, Any]], output_path: Path):
    """Create comprehensive multi-panel visualization of TrueSkill analysis."""
    sorted_stats = sorted(stats, key=lambda x: x["conservative_rating"], reverse=True)

    models = [s["model"] for s in sorted_stats]
    mus = [s["rating_mu"] for s in sorted_stats]
    sigmas = [s["rating_sigma"] for s in sorted_stats]
    avg_chips = [s["avg_chips"] for s in sorted_stats]
    avg_ranks = [s["avg_rank"] for s in sorted_stats]
    games_played = [s["games_played"] for s in sorted_stats]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # TrueSkill Ratings with Error Bars
    ax1 = fig.add_subplot(gs[0, :])
    positions = range(len(models))
    ax1.errorbar(
        positions,
        mus,
        yerr=[1 * s for s in sigmas],
        fmt="o",
        capsize=5,
        capthick=2,
        markersize=8,
        linewidth=2,
        color="#2E86AB",
        ecolor="#A23B72",
    )
    ax1.axhline(y=25.0, color="gray", linestyle=":", alpha=0.5, linewidth=2)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("TrueSkill Rating (μ)", fontsize=11)
    ax1.set_title(
        "TrueSkill Ratings with Uncertainty (±1σ)", fontsize=13, fontweight="bold"
    )
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Rating vs Average Chips
    ax2 = fig.add_subplot(gs[1, 0])
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    ax2.scatter(
        mus, avg_chips, c=colors, s=100, alpha=0.7, edgecolors="black", linewidth=1.5
    )

    # Add trend line
    z = np.polyfit(mus, avg_chips, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(mus), max(mus), 100)
    ax2.plot(
        x_trend,
        p(x_trend),
        "r--",
        alpha=0.8,
        linewidth=2,
        label=f"Trend: y={z[0]:.1f}x{z[1]:+.1f}",
    )

    # Calculate correlation
    correlation = np.corrcoef(mus, avg_chips)[0, 1]
    ax2.text(
        0.05,
        0.95,
        f"Correlation: {correlation:.3f}",
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax2.set_xlabel("TrueSkill Rating (μ)", fontsize=11)
    ax2.set_ylabel("Average Final Chips", fontsize=11)
    ax2.set_title("Rating vs Performance (Chip Count)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    # Rating vs Average Rank
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(
        mus, avg_ranks, c=colors, s=100, alpha=0.7, edgecolors="black", linewidth=1.5
    )

    z_rank = np.polyfit(mus, avg_ranks, 1)
    p_rank = np.poly1d(z_rank)
    ax3.plot(
        x_trend,
        p_rank(x_trend),
        "r--",
        alpha=0.8,
        linewidth=2,
        label=f"Trend: y={z_rank[0]:.2f}x{z_rank[1]:+.1f}",
    )

    correlation_rank = np.corrcoef(mus, avg_ranks)[0, 1]
    ax3.text(
        0.05,
        0.95,
        f"Correlation: {correlation_rank:.3f}",
        transform=ax3.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax3.set_xlabel("TrueSkill Rating (μ)", fontsize=11)
    ax3.set_ylabel("Average Final Rank", fontsize=11)
    ax3.set_title(
        "Rating vs Average Rank (Lower is Better)", fontsize=12, fontweight="bold"
    )
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)

    # Uncertainty by Games Played
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(
        games_played,
        sigmas,
        c=colors,
        s=100,
        alpha=0.7,
        edgecolors="black",
        linewidth=1.5,
    )
    ax4.set_xlabel("Games Played", fontsize=11)
    ax4.set_ylabel("Rating Uncertainty (σ)", fontsize=11)
    ax4.set_title(
        "Confidence Increases with More Games", fontsize=12, fontweight="bold"
    )
    ax4.grid(True, alpha=0.3)

    # Top Models Comparison
    ax5 = fig.add_subplot(gs[2, 1])
    top_n = min(8, len(models))
    top_models = models[:top_n]
    top_mus = mus[:top_n]
    top_sigmas = sigmas[:top_n]

    bar_positions = range(top_n)
    ax5.barh(
        bar_positions,
        top_mus,
        xerr=[1 * s for s in top_sigmas],
        capsize=4,
        color="#55A868",
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )
    ax5.axvline(
        x=25.0,
        color="gray",
        linestyle=":",
        alpha=0.5,
        linewidth=2,
        label="Initial Rating",
    )
    ax5.set_yticks(bar_positions)
    ax5.set_yticklabels(top_models, fontsize=9)
    ax5.set_xlabel("TrueSkill Rating (μ ± 1σ)", fontsize=11)
    ax5.set_title(
        f"Top {top_n} Models by Conservative Rating", fontsize=12, fontweight="bold"
    )
    ax5.invert_yaxis()
    ax5.grid(axis="x", alpha=0.3)
    ax5.legend(fontsize=9)

    plt.suptitle(
        "Comprehensive TrueSkill Analysis for Poker Models",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Comprehensive analysis plot saved to {output_path}")


def main():
    """Main function to calculate and display TrueSkill ratings."""
    parser = argparse.ArgumentParser(
        description="Calculate TrueSkill ratings for poker models from game logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all logs recursively (default, excludes archived/)
  %(prog)s
  
  # Analyze all logs with plots
  %(prog)s --all-plots
  
  # Analyze specific seed folders only
  %(prog)s --seeds seed_42 seed_1042 seed_3042
  
  # Exclude additional directories
  %(prog)s --exclude-dirs archived old_runs test_data
  
  # Specify custom logs directory
  %(prog)s --logs-dir logs/experimental
  
  # Custom TrueSkill parameters
  %(prog)s --mu 30.0 --sigma 10.0
        """,
    )

    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "logs",
        help="Base logs directory (default: ../logs)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        help="Seed folder names to include. If not specified, searches recursively for all .json files (excluding archived/)",
    )
    parser.add_argument(
        "--exclude-dirs",
        nargs="+",
        default=["archived"],
        help="Directory names to exclude when searching recursively (default: archived)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search recursively for all JSON files (default: True). Use --no-recursive to disable.",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Disable recursive search (requires --seeds to be specified)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "analysis"
        / "results"
        / "trueskill_rankings.json",
        help="Output JSON file path (default: analysis/results/trueskill_rankings.json)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "analysis"
        / "results"
        / "trueskill_ratings.png",
        help="Output plot file path (default: analysis/results/trueskill_ratings.png)",
    )
    parser.add_argument(
        "--comprehensive-plot",
        action="store_true",
        help="Generate comprehensive multi-panel analysis plot",
    )
    parser.add_argument(
        "--comprehensive-plot-output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "analysis"
        / "results"
        / "trueskill_comprehensive.png",
        help="Output comprehensive plot file path (default: analysis/results/trueskill_comprehensive.png)",
    )
    parser.add_argument(
        "--separate-plots",
        action="store_true",
        help="Generate separate individual plots for each analysis",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "analysis" / "results",
        help="Directory for individual plots (default: analysis/results/)",
    )
    parser.add_argument(
        "--pareto-plot",
        action="store_true",
        help="Generate Pareto frontier plot comparing cost efficiency vs performance",
    )
    parser.add_argument(
        "--pareto-plot-output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "analysis"
        / "results"
        / "trueskill_pareto_frontier.png",
        help="Output Pareto plot file path (default: analysis/results/trueskill_pareto_frontier.png)",
    )
    parser.add_argument(
        "--all-plots",
        action="store_true",
        help="Generate ALL available plots (equivalent to --plot --comprehensive-plot --separate-plots --pareto-plot)",
    )

    # TrueSkill parameters
    parser.add_argument(
        "--mu",
        type=float,
        default=25.0,
        help="Initial mean skill (default: 25.0)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=8.333,
        help="Initial skill uncertainty (default: 8.333)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=4.166,
        help="Skill class width (default: 4.166)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.0833,
        help="Dynamics factor (default: 0.0833)",
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        default=True,
        help="Scale ratings to 1000-baseline (ELO-like). Default: True. Use --no-scale for raw TrueSkill.",
    )
    parser.add_argument(
        "--no-scale",
        dest="scale",
        action="store_false",
        help="Disable scaling, show raw TrueSkill ratings (baseline 25)",
    )

    args = parser.parse_args()

    # If --all-plots is specified, enable all plotting options
    if args.all_plots:
        args.plot = True
        args.comprehensive_plot = True
        args.separate_plots = True
        args.pareto_plot = True

    print("=" * 70)
    print("Loading game logs...")
    print("=" * 70)

    # Determine search mode
    if args.seeds is not None:
        # Specific seed folders mode
        all_logs = gather_all_logs(args.logs_dir, seed_folders=args.seeds)
        print(f"\nLoaded {len(all_logs)} game logs from {len(args.seeds)} seed folders")
        print(f"Seeds: {', '.join(args.seeds)}")
    else:
        # Recursive search mode (default)
        all_logs = gather_all_logs(
            args.logs_dir, seed_folders=None, exclude_dirs=args.exclude_dirs
        )
        print(f"\nLoaded {len(all_logs)} game logs recursively from {args.logs_dir}")
        print(f"Excluded directories: {', '.join(args.exclude_dirs)}")

    if not all_logs:
        print("Error: No valid log files found.")
        return 1

    print()

    print("Calculating TrueSkill ratings...")
    ratings, game_counts = calculate_trueskill_ratings(
        all_logs,
        mu=args.mu,
        sigma=args.sigma,
        beta=args.beta,
        tau=args.tau,
    )

    include_costs = args.pareto_plot
    stats = get_model_statistics(
        ratings,
        game_counts,
        all_logs,
        scale_ratings=args.scale,
        include_costs=include_costs,
    )

    print_rankings_table(stats, scaled=args.scale, total_games=len(all_logs))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_rankings_json(
        stats, args.output, len(all_logs), scaled=args.scale, ratings=ratings
    )
    if args.plot:
        plot_ratings(stats, args.plot_output, scaled=args.scale)

    if args.comprehensive_plot:
        plot_comprehensive_analysis(stats, args.comprehensive_plot_output)

    if args.separate_plots:
        args.plots_dir.mkdir(parents=True, exist_ok=True)
        plot_rating_vs_chips(stats, args.plots_dir / "trueskill_rating_vs_chips.png")
        plot_rating_vs_rank(stats, args.plots_dir / "trueskill_rating_vs_rank.png")
        plot_confidence_vs_games(
            stats, args.plots_dir / "trueskill_confidence_vs_games.png"
        )
        plot_top_models_comparison(stats, args.plots_dir / "trueskill_top_models.png")

    if args.pareto_plot:
        args.pareto_plot_output.parent.mkdir(parents=True, exist_ok=True)
        plot_pareto_frontier(stats, args.pareto_plot_output)

    return 0


if __name__ == "__main__":
    exit(main())
