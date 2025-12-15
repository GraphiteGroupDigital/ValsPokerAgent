"""
Utility script to analyze and display poker game logs, including token usage, chip usage, and costs.

"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Sequence
from collections import defaultdict
import statistics

# Default output directory for all plots
RESULTS_DIR = Path(__file__).parent / "results"

# Matplotlib imported lazily inside plot function to avoid hard dependency during non-plot runs


def load_model_costs() -> Dict[str, Dict[str, float]]:
    """Load model costs from vals model proxy registry.

    Returns:
        Dictionary mapping model names to their cost information
    """
    try:
        from model_library.registry_utils import get_model_registry

        registry = get_model_registry()

        # Extract cost information from the registry
        model_costs = {}
        for model_name, model_config in registry.items():
            if hasattr(model_config, "costs_per_million_token"):
                costs = model_config.costs_per_million_token
                model_costs[model_name] = {
                    "input": float(costs.input or 0.0),
                    "output": float(costs.output or 0.0),
                }

        return model_costs
    except ImportError:
        print("Warning: model_library not available, using default cost lookup")
        # Return empty dict to trigger default cost fallback
        return {}


def calculate_tokens(model_response: Dict[str, Any], token_mode: str) -> int:
    """Calculate tokens based on the specified mode.

    Args:
        model_response: The model response dictionary
        token_mode: One of "input", "output", or "total"

    Returns:
        Number of tokens based on the mode
    """
    # New format: calculate tokens (reasoning_tokens optional for non-reasoning models)
    if "out_tokens" in model_response or "in_tokens" in model_response:
        out_tokens = int(model_response.get("out_tokens", 0))
        reasoning_tokens = int(model_response.get("reasoning_tokens", 0))
        in_tokens = int(model_response.get("in_tokens", 0))

        if token_mode == "input":
            return in_tokens
        elif token_mode == "output":
            return out_tokens + reasoning_tokens
        elif token_mode == "total":
            return in_tokens + out_tokens + reasoning_tokens
    # Old format: tokens in usage object
    elif "usage" in model_response:
        usage = model_response.get("usage", {})
        if token_mode == "input":
            return int(usage.get("prompt_tokens", 0))
        elif token_mode == "output":
            return int(usage.get("completion_tokens", 0))
        elif token_mode == "total":
            return int(usage.get("total_tokens", 0))

    return 0


def get_token_type_label(token_mode: str) -> str:
    """Get a human-readable label for the token mode."""
    if token_mode == "input":
        return "input tokens"
    elif token_mode == "output":
        return "output tokens"
    elif token_mode == "total":
        return "total tokens"
    else:
        return "tokens"


def normalize_model_name(model_name: str, provider: str = None) -> str:
    """Normalize model name using the same logic as resume_game.py

    Args:
        model_name: The model name from the log
        provider: The provider name (optional)

    Returns:
        Normalized model name for registry lookup
    """
    if not model_name:
        return model_name

    # Strip any provider-specific path like accounts/.../models/<name>
    if "/models/" in model_name:
        model_name = model_name.split("/models/", 1)[1]

    # Fallback: take last path segment if any slashes remain
    if "/" in model_name:
        model_name = model_name.rsplit("/", 1)[-1]

    # Create registry key in the same format as resume_game.py
    if provider:
        provider = provider.lower()
        if provider == "fireworks":
            return f"fireworks/{model_name}"
        else:
            return f"{provider}/{model_name}"

    return model_name


def get_model_cost_info(model_name: str, provider: str = None) -> Dict[str, float]:
    """Get cost information for a model.

    Args:
        model_name: The model name from the log
        provider: The provider name (optional)

    Returns:
        Dictionary with input and output costs per million tokens
    """
    model_costs = load_model_costs()

    # Try exact match first
    if model_name in model_costs:
        return model_costs[model_name]

    # Normalize the model name using the same logic as resume_game.py
    normalized_name = normalize_model_name(model_name, provider)

    # Try normalized name
    if normalized_name != model_name and normalized_name in model_costs:
        return model_costs[normalized_name]

    # Try partial matches for different naming conventions
    for key, costs in model_costs.items():
        if model_name in key or key in model_name:
            return costs
        # Also try partial matches with normalized model name
        if normalized_name != model_name and (
            normalized_name in key or key in normalized_name
        ):
            return costs

    # Default cost if model not found
    print(
        f"Warning: No cost information found for model '{model_name}', using default costs"
    )
    return {"input": 1.0, "output": 1.0}


def calculate_cost(model_response: Dict[str, Any], token_mode: str = "total") -> float:
    """Calculate cost based on token usage and model pricing.

    Args:
        model_response: The model response dictionary
        token_mode: One of "input", "output", or "total"

    Returns:
        Cost in dollars
    """
    model_name = model_response.get("model_name", "")
    provider = model_response.get("provider", "")
    cost_info = get_model_cost_info(model_name, provider)

    # Get token counts
    in_tokens = int(model_response.get("in_tokens", 0))
    out_tokens = int(model_response.get("out_tokens", 0))
    reasoning_tokens = int(model_response.get("reasoning_tokens", 0))

    # Calculate costs based on token mode
    if token_mode == "input":
        tokens = in_tokens
        cost_per_million = cost_info["input"]
    elif token_mode == "output":
        # For output mode, include both output and reasoning tokens at output pricing
        tokens = out_tokens + reasoning_tokens
        cost_per_million = cost_info["output"]
    elif token_mode == "total":
        # For total cost, calculate input and output separately
        input_cost = (in_tokens / 1_000_000) * cost_info["input"]
        output_cost = ((out_tokens + reasoning_tokens) / 1_000_000) * cost_info[
            "output"
        ]
        return input_cost + output_cost
    else:
        return 0.0

    return (tokens / 1_000_000) * cost_per_million


def print_session_summary(log: Dict[str, Any]):
    """Print session metadata."""
    info = log["session_info"]
    print("=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    print(f"Players: {', '.join(info['player_names'])}")
    print(f"Starting stacks: {info['starting_stacks']}")
    print(f"Blinds: {info['small_blind']}/{info['big_blind']}")
    print(f"Seed: {info.get('seed', 'Random')}")
    print(f"Start time: {info.get('start_time', 'N/A')}")
    print(f"End time: {info.get('end_time', 'N/A')}")
    print(f"Total hands: {info.get('total_hands', len(log['hands']))}")
    print()


def print_hand_summary(
    hand: Dict[str, Any], log: Dict[str, Any], verbose: bool = False
):
    """Print summary of a single hand."""
    print("-" * 70)
    print(f"HAND {hand['hand_number']}")
    print("-" * 70)
    print(f"Button seat: {hand['button_seat']}")
    print(f"Total actions: {len(hand['actions'])}")

    # Starting stacks
    print("\nStarting state:")
    for player, state in hand["starting_state"].items():
        cards = ", ".join(state["hole_cards"])
        print(
            f"  {player}: [{cards}] - Position {state['position']} - Stack: {state['starting_stack']}"
        )

    # Actions
    if verbose:
        print("\nActions:")
        for action in hand["actions"]:
            board_str = (
                f"[{', '.join(action['board'])}]" if action["board"] else "[Preflop]"
            )
            action_str = action["action"]["type"]
            if "amount" in action["action"]:
                action_str += f" to {action['action']['amount']}"

            print(f"  [{action['timestep']}] {action['player']}: {action_str}")
            print(
                f"      Board: {board_str} | Chip change: {action['chip_change']:+d} | Stack after: {action['stack_after']}"
            )

            # Show model reasoning if available
            if "model_response" in action and "reasoning" in action["model_response"]:
                reasoning = action["model_response"]["reasoning"]
                if reasoning:
                    # Truncate long reasoning
                    if len(reasoning) > 100:
                        reasoning = reasoning[:97] + "..."
                    print(f"      Reasoning: {reasoning}")

            # Show errors if any
            if "model_response" in action and "error" in action["model_response"]:
                print(f"      ⚠️  Error: {action['model_response']['error']}")

    # Final results
    print(
        "\nFinal board:",
        ", ".join(hand["final_board"]) if hand["final_board"] else "N/A",
    )
    print("Ending stacks:")
    # Use session initial stacks for consistent profit calculation
    player_names = log["session_info"]["player_names"]
    initial_stacks = log["session_info"]["starting_stacks"]
    initial_stack_map = dict(zip(player_names, initial_stacks))

    for player, stack in hand["ending_stacks"].items():
        if player in initial_stack_map:
            starting = initial_stack_map[player]
            profit = stack - starting
            print(f"  {player}: {stack} ({profit:+d})")
        else:
            print(f"  {player}: {stack} (starting stack unknown)")
    print()


def print_player_stats(
    log: Dict[str, Any],
    token_mode: str = "output",
    show_costs: bool = True,
    cost_mode: str = "total",
):
    """Print aggregate statistics per player, including token usage averages and costs."""
    players = log["session_info"]["player_names"]
    stats = {
        player: {
            "hands_played": 0,
            "total_actions": 0,
            "folds": 0,
            "calls": 0,
            "checks": 0,
            "bets": 0,
            "raises": 0,
            "total_profit": 0,
            # token aggregates
            "token_sum": 0,
            "token_count": 0,  # count only actions with a usage.total_tokens entry > 0
            "action_tokens": {},  # action_type -> {sum, count}
            # cost aggregates
            "total_cost": 0.0,
            "cost_count": 0,  # count only actions with cost > 0
            "action_costs": {},  # action_type -> {sum, count}
            "model_name": None,  # track which model this player uses
        }
        for player in players
    }

    starting_stacks = {
        player: stack
        for player, stack in zip(players, log["session_info"]["starting_stacks"])
    }

    for hand in log["hands"]:
        # Track participation
        for player in hand["starting_state"].keys():
            stats[player]["hands_played"] += 1

        # Track actions
        for action in hand["actions"]:
            player = action["player"]
            stats[player]["total_actions"] += 1
            action_type = action["action"]["type"]
            # Map action types to stat keys
            stat_key = action_type + "s"
            if stat_key in stats[player]:
                stats[player][stat_key] += 1

            # Token usage and cost calculation - check for both old and new log formats
            model_response = action.get("model_response", {})
            tokens = calculate_tokens(model_response, token_mode)
            cost = calculate_cost(model_response, cost_mode) if show_costs else 0.0

            # Track model name for this player
            if stats[player]["model_name"] is None and "model_name" in model_response:
                stats[player]["model_name"] = model_response["model_name"]

            if tokens > 0:
                stats[player]["token_sum"] += tokens
                stats[player]["token_count"] += 1
                per_action = stats[player]["action_tokens"].setdefault(
                    action_type, {"sum": 0, "count": 0}
                )
                per_action["sum"] += tokens
                per_action["count"] += 1

            if cost > 0:
                stats[player]["total_cost"] += cost
                stats[player]["cost_count"] += 1
                per_action_cost = stats[player]["action_costs"].setdefault(
                    action_type, {"sum": 0.0, "count": 0}
                )
                per_action_cost["sum"] += cost
                per_action_cost["count"] += 1

        # Track profit for this hand
        for player, ending_stack in hand["ending_stacks"].items():
            starting_stack = starting_stacks[player]
            stats[player]["total_profit"] += ending_stack - starting_stack

    print("=" * 70)
    print("PLAYER STATISTICS")
    print("=" * 70)
    total_cost_all_players = 0.0

    for player in players:
        s = stats[player]
        print(f"\n{player}:")
        if s["model_name"]:
            print(f"  Model: {s['model_name']}")
        print(f"  Hands played: {s['hands_played']}")
        print(f"  Actions taken: {s['total_actions']}")
        print(f"  Folds: {s['folds']} | Checks: {s['checks']} | Calls: {s['calls']}")
        print(f"  Bets: {s['bets']} | Raises: {s['raises']}")
        print(f"  Total profit/loss: {s['total_profit']:+d} chips")

        # Token averages
        if s["token_count"] > 0:
            overall_avg = s["token_sum"] / s["token_count"]
            token_type = get_token_type_label(token_mode)
            print(
                f"  Avg {token_type} per action: {overall_avg:.2f} over {s['token_count']} actions"
            )
        else:
            token_type = get_token_type_label(token_mode)
            print(f"  Avg {token_type} per action: N/A")

        if s["action_tokens"]:
            print(f"  Avg {token_type} by action type:")
            for action_type in sorted(s["action_tokens"].keys()):
                agg = s["action_tokens"][action_type]
                avg = agg["sum"] / agg["count"] if agg["count"] else 0.0
                print(
                    f"    - {action_type}: {avg:.2f} over {agg['count']} actions (sum={agg['sum']})"
                )

        # Cost information
        print(f"  Total cost: ${s['total_cost']:.4f}")
        total_cost_all_players += s["total_cost"]

        if s["cost_count"] > 0:
            avg_cost = s["total_cost"] / s["cost_count"]
            print("  Avg cost per action: N/A")

        if s["action_costs"]:
            print("  Cost by action type:")
        if s["action_costs"]:
            print("  Cost by action type:")
            for action_type in sorted(s["action_costs"].keys()):
                agg = s["action_costs"][action_type]
                avg = agg["sum"] / agg["count"] if agg["count"] else 0.0
                print(
                    f"    - {action_type}: ${avg:.6f} over {agg['count']} actions (total=${agg['sum']:.4f})"
                )

    print(f"\n{'=' * 70}")
    print(f"TOTAL COST ACROSS ALL MODELS: ${total_cost_all_players:.4f}")
    print(f"{'=' * 70}")

    print_final_ranking(log, stats)
    print()


def print_final_ranking(log: Dict[str, Any], stats: Dict[str, Any]):
    """Print final ranking of models by chip count, with tiebreaking for eliminated players."""
    players = log["session_info"]["player_names"]
    initial_stacks = log["session_info"]["starting_stacks"]
    initial_stack_map = dict(zip(players, initial_stacks))

    # Get final chip counts and elimination info
    player_data = []
    for player in players:
        # Get final stack from the last hand
        last_hand = log["hands"][-1] if log["hands"] else {}
        final_stack = last_hand.get("ending_stacks", {}).get(player, 0)

        # Find elimination hand (first hand where player has 0 chips)
        elimination_hand = None
        elimination_round = None
        for hand in log["hands"]:
            ending_stack = hand.get("ending_stacks", {}).get(player, 0)
            if ending_stack == 0:
                elimination_hand = hand.get("hand_number", 0)
                # Find the round within the hand where they were eliminated
                # Look for the last action where they had chips > 0
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
                "elimination_hand": elimination_hand,
                "elimination_round": elimination_round,
                "model_name": stats[player]["model_name"],
            }
        )

    player_data.sort(
        key=lambda x: (
            -x["final_stack"],  # Higher stacks first
            -(
                x["elimination_hand"]
                if x["final_stack"] == 0 and x["elimination_hand"] is not None
                else 0
            ),  # Later elimination first for 0-stack
            -(
                x["elimination_round"]
                if x["final_stack"] == 0 and x["elimination_round"] is not None
                else 0
            ),  # Later round first for same hand
        )
    )

    print(f"\n{'=' * 70}")
    print("FINAL RANKING BY CHIPS")
    print(f"{'=' * 70}")

    for i, data in enumerate(player_data, 1):
        player = data["player"]
        final_stack = data["final_stack"]
        model_name = data["model_name"] or "Unknown"

        if final_stack > 0:
            print(f"{i:2d}. {player:<25} | {final_stack:>4d} chips")
        else:
            elimination_info = f"Eliminated Hand {data['elimination_hand']}"
            if data["elimination_round"] is not None:
                elimination_info += f", Round {data['elimination_round']}"
            print(f"{i:2d}. {player:<25} | {elimination_info}")


def extract_round_token_points(
    log: Dict[str, Any], token_mode: str = "output"
) -> Dict[str, Tuple[List[int], List[int]]]:
    """Compute per-player (x, y) points of per-turn tokens by 3-timestep rounds.

    - Groups by hand, buckets timesteps into rounds of size num_players (0..n-1, n..2n-1, ...).
    - For each round, if a player has no tokens used (no action or tokens==0), we do NOT
      add a point for that player (skip instead of plotting zero).
    - Returns a dict player -> (x_indices, y_tokens) where x_indices are global round indices.
    """
    players: List[str] = log["session_info"]["player_names"]
    num_players = max(1, len(players))
    points: Dict[str, Tuple[List[int], List[int]]] = {p: ([], []) for p in players}

    hands = sorted(log.get("hands", []), key=lambda h: h.get("hand_number", 0))
    global_round_index = 0
    for hand in hands:
        actions = sorted(hand.get("actions", []), key=lambda a: a.get("timestep", 0))
        # round_idx -> player -> tokens
        rounds: Dict[int, Dict[str, int]] = {}
        for action in actions:
            timestep = int(action.get("timestep", 0))
            round_idx = timestep // num_players
            player = action.get("player")
            # Token usage
            model_response = action.get("model_response", {})
            tokens = calculate_tokens(model_response, token_mode)
            if round_idx not in rounds:
                rounds[round_idx] = {}
            rounds[round_idx][player] = tokens

        for _, per_player in sorted(rounds.items()):
            # append point only if tokens > 0 for that player in this round
            for p in players:
                tokens = per_player.get(p, 0)
                if tokens > 0:
                    px, py = points[p]
                    px.append(global_round_index)
                    py.append(tokens)
            global_round_index += 1

    return points


def extract_hand_token_totals(
    log: Dict[str, Any], token_mode: str = "output"
) -> Dict[str, Tuple[List[int], List[int]]]:
    """Compute per-player total tokens used per hand.

    Returns a dict player -> (hand_numbers, token_totals) where each entry
    is the sum of all tokens used by that player in that hand.
    """
    players: List[str] = log["session_info"]["player_names"]
    series: Dict[str, Tuple[List[int], List[int]]] = {p: ([], []) for p in players}

    for hand in log.get("hands", []):
        hand_num = hand.get("hand_number", 0)

        # Track tokens per player for this hand
        hand_tokens: Dict[str, int] = {p: 0 for p in players}

        for action in hand.get("actions", []):
            player = action.get("player")
            if player not in hand_tokens:
                continue

            # Token usage
            model_response = action.get("model_response", {})
            tokens = calculate_tokens(model_response, token_mode)

            hand_tokens[player] += tokens

        # Add hand data point for each player
        for player in players:
            xs, ys = series[player]
            xs.append(hand_num)
            ys.append(hand_tokens[player])

    return series


def maybe_plot_hand_tokens(
    log: Dict[str, Any], save_path: Optional[str], token_mode: str = "output"
):
    """Plot total token usage per hand for each player."""
    series = extract_hand_token_totals(log, token_mode)

    # Lazy import to avoid requiring matplotlib when not plotting
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    for player, (xs, ys) in series.items():
        if xs:
            plt.plot(xs, ys, marker="o", linewidth=2, markersize=8, label=player)

    plt.xlabel("Hand Number")
    token_type = get_token_type_label(token_mode)
    title = f"{token_type.title()} Usage Per Hand"
    plt.ylabel(f"{token_type.title()} used in hand")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    if save_path:
        # Ensure results directory exists and save there
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RESULTS_DIR / save_path
        plt.savefig(output_path, dpi=150)
        print(f"Saved hand tokens plot to {output_path}")
    else:
        plt.show()


def maybe_plot_round_tokens(
    log: Dict[str, Any], save_path: Optional[str], token_mode: str = "output"
):
    points = extract_round_token_points(log, token_mode)

    # Lazy import to avoid requiring matplotlib when not plotting
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for player, (xs, ys) in points.items():
        if xs:
            plt.plot(xs, ys, marker="o", linewidth=2, label=player)

    plt.xlabel("Round index (every 3 actions)")
    token_type = get_token_type_label(token_mode)
    title = f"Per-Turn {token_type.title()} Usage (3-timestep rounds)"
    plt.ylabel(f"{token_type.title()} used on turn")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    if save_path:
        # Ensure results directory exists and save there
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RESULTS_DIR / save_path
        plt.savefig(output_path, dpi=150)
        print(f"Saved round tokens plot to {output_path}")
    else:
        plt.show()


def extract_round_chips_series(
    log: Dict[str, Any],
) -> Dict[str, Tuple[List[int], List[int]]]:
    """Compute per-player (x, y) stacks by round index.

    - Round index per hand is timestep // num_players; indexed globally across hands.
    - At the start of each hand, initialize current stacks from starting_state.
    - After processing all actions in a round, append a point for each player with their
      current stack (carried forward if they did not act that round).
    """
    players: List[str] = log["session_info"]["player_names"]
    num_players = max(1, len(players))
    series: Dict[str, Tuple[List[int], List[int]]] = {p: ([], []) for p in players}

    hands = sorted(log.get("hands", []), key=lambda h: h.get("hand_number", 0))
    global_round_index = 0
    for hand in hands:
        # Initialize stacks from starting_state for this hand
        current_stacks: Dict[str, int] = {
            p: int(hand["starting_state"][p]["starting_stack"]) for p in players
        }

        actions = sorted(hand.get("actions", []), key=lambda a: a.get("timestep", 0))
        # Build mapping of round -> list of actions in that round
        rounds: Dict[int, List[Dict[str, Any]]] = {}
        for action in actions:
            timestep = int(action.get("timestep", 0))
            round_idx = timestep // num_players
            rounds.setdefault(round_idx, []).append(action)

        for _, actions_in_round in sorted(rounds.items()):
            # Apply actions in round to update stacks
            for action in actions_in_round:
                p = action.get("player")
                stack_after = int(action.get("stack_after", current_stacks.get(p, 0)))
                current_stacks[p] = stack_after

            # Append a point for each player with current stack
            for p in players:
                px, py = series[p]
                px.append(global_round_index)
                py.append(current_stacks[p])

            global_round_index += 1

        # After all rounds for this hand, append a final point reflecting ending_stacks
        # to capture the payout distribution at showdown.
        if "ending_stacks" in hand:
            for p in players:
                current_stacks[p] = int(hand["ending_stacks"].get(p, current_stacks[p]))
            for p in players:
                px, py = series[p]
                px.append(global_round_index)
                py.append(current_stacks[p])
            global_round_index += 1

    return series


def maybe_plot_round_chips(log: Dict[str, Any], save_path: Optional[str]):
    series = extract_round_chips_series(log)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for player, (xs, ys) in series.items():
        if xs:
            plt.plot(xs, ys, linewidth=2, label=player)

    plt.xlabel("Round index (every num_players actions)")
    plt.ylabel("Chips (stack)")
    plt.title("Player Stacks Over Rounds")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    if save_path:
        # Ensure results directory exists and save there
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RESULTS_DIR / save_path
        plt.savefig(output_path, dpi=150)
        print(f"Saved chips plot to {output_path}")
    else:
        plt.show()


def load_logs_from_folder(folder_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """Load all JSON log files from a folder.

    Returns:
        List of (filename, log_data) tuples
    """
    logs = []
    json_files = sorted(folder_path.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return logs

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                log = json.load(f)
                logs.append((json_file.name, log))
        except Exception as e:
            print(f"Warning: Failed to load {json_file.name}: {e}")

    return logs


def load_logs_from_folders(
    folder_paths: Sequence[Path],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Load logs from multiple folders, prefixing filenames with folder names."""

    all_logs: List[Tuple[str, Dict[str, Any]]] = []
    for folder in folder_paths:
        folder_logs = load_logs_from_folder(folder)
        for filename, log in folder_logs:
            label = f"{folder.name}/{filename}"
            all_logs.append((label, log))

    return all_logs


def gather_seed_logs(
    seed_names: Sequence[str], base_logs_dir: Optional[Path] = None
) -> Tuple[List[Path], List[Tuple[str, Dict[str, Any]]]]:
    """Resolve seed folders, load logs, and return both paths and logs."""

    base_dir = base_logs_dir or (Path(__file__).resolve().parents[1] / "logs")

    folder_paths: List[Path] = []
    missing: List[Path] = []
    for seed_name in seed_names:
        seed_path = Path(seed_name)
        if not seed_path.is_absolute():
            seed_path = base_dir / seed_path

        if seed_path.is_dir():
            folder_paths.append(seed_path)
        else:
            missing.append(seed_path)

    if missing:
        for missing_path in missing:
            print(f"Warning: seed folder not found: {missing_path}")

    if not folder_paths:
        raise ValueError("No valid seed folders provided.")

    logs = load_logs_from_folders(folder_paths)
    if not logs:
        raise ValueError("No JSON logs found in the provided seed folders.")

    return folder_paths, logs


def collect_seat_statistics(
    logs: Sequence[Tuple[str, Dict[str, Any]]],
) -> Dict[int, Dict[str, Any]]:
    """Return aggregated placement/final-chip data per seat."""

    seat_data: Dict[int, Dict[str, Any]] = defaultdict(
        lambda: {"placements": [], "final_chips": [], "models_used": set()}
    )

    for _, log in logs:
        player_names = log["session_info"]["player_names"]
        last_hand = log["hands"][-1] if log["hands"] else {}

        player_data = []
        for seat, player_name in enumerate(player_names):
            final_stack = last_hand.get("ending_stacks", {}).get(player_name, 0)

            model_name = None
            for hand in log["hands"]:
                for action in hand.get("actions", []):
                    if (
                        action.get("player") == player_name
                        and "model_response" in action
                    ):
                        model_name = action["model_response"].get(
                            "model_name", "Unknown"
                        )
                        break
                if model_name:
                    break

            player_data.append(
                {
                    "seat": seat,
                    "player": player_name,
                    "final_stack": final_stack,
                    "model": model_name,
                }
            )

        player_data.sort(key=lambda x: x["final_stack"], reverse=True)

        for placement, data in enumerate(player_data, 1):
            seat = data["seat"]
            seat_data[seat]["placements"].append(placement)
            seat_data[seat]["final_chips"].append(data["final_stack"])
            if data["model"]:
                seat_data[seat]["models_used"].add(data["model"])

    return seat_data


def collect_model_statistics(
    logs: Sequence[Tuple[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Return aggregated placement/final-chip data per model."""

    model_data: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"placements": [], "final_chips": [], "seats_played": set()}
    )

    for _, log in logs:
        player_names = log["session_info"]["player_names"]
        last_hand = log["hands"][-1] if log["hands"] else {}

        player_data = []
        for seat, player_name in enumerate(player_names):
            final_stack = last_hand.get("ending_stacks", {}).get(player_name, 0)

            model_name = "Unknown"
            for hand in log["hands"]:
                for action in hand.get("actions", []):
                    if (
                        action.get("player") == player_name
                        and "model_response" in action
                    ):
                        response = action["model_response"]
                        model_name = response.get("model_name", "Unknown")
                        provider = response.get("provider", "")
                        if provider:
                            model_name = f"{provider}/{model_name.split('/')[-1]}"
                        break
                if model_name != "Unknown":
                    break

            player_data.append(
                {
                    "seat": seat,
                    "player": player_name,
                    "final_stack": final_stack,
                    "model": model_name,
                }
            )

        player_data.sort(key=lambda x: x["final_stack"], reverse=True)

        for placement, data in enumerate(player_data, 1):
            model = data["model"]
            model_data[model]["placements"].append(placement)
            model_data[model]["final_chips"].append(data["final_stack"])
            model_data[model]["seats_played"].add(data["seat"])

    return model_data


def analyze_seed_directories(
    seed_names: Sequence[str],
    base_logs_dir: Optional[Path] = None,
    aggregate_by: str = "all",
) -> None:
    """Analyze multiple seed folders and print aggregate stats."""

    if aggregate_by not in {"seat", "model", "all"}:
        raise ValueError("aggregate_by must be 'seat', 'model', or 'all'")

    folder_paths, logs = gather_seed_logs(seed_names, base_logs_dir)

    print("=" * 70)
    print(
        f"Analyzing {len(logs)} logs from {len(folder_paths)} seed folders "
        f"({', '.join(path.name for path in folder_paths)})"
    )
    print("=" * 70)
    print()

    analyze_folder_aggregate(logs, aggregate_by)


def analyze_folder_aggregate(
    logs: List[Tuple[str, Dict[str, Any]]], aggregate_by: str = "all"
):
    """Analyze multiple logs and compute aggregate statistics.

    Args:
        logs: List of (filename, log_data) tuples
        aggregate_by: Either "seat" or "model"
    """
    print("=" * 70)
    print(f"AGGREGATE ANALYSIS ({len(logs)} games) - BY {aggregate_by.upper()}")
    print("=" * 70)
    print()

    # Collect placement and chip data
    if aggregate_by in ["seat", "all"]:
        seat_data = collect_seat_statistics(logs)

        # Print seat statistics
        for seat in sorted(seat_data.keys()):
            data = seat_data[seat]
            placements = data["placements"]
            chips = data["final_chips"]

            print(f"Seat {seat}:")
            print(f"  Games played: {len(placements)}")
            print(f"  Average placement: {statistics.mean(placements):.2f}")
            print(
                f"  Placement std dev: {statistics.stdev(placements) if len(placements) > 1 else 0:.2f}"
            )
            print(f"  Best: {min(placements)}, Worst: {max(placements)}")
            print(f"  Average final chips: {statistics.mean(chips):.1f}")
            print(
                f"  Chips std dev: {statistics.stdev(chips) if len(chips) > 1 else 0:.1f}"
            )
            print(f"  Models used: {', '.join(sorted(data['models_used']))}")
            print()

    if aggregate_by in ["model", "all"]:
        model_data = collect_model_statistics(logs)

        # Print model statistics sorted by average placement
        model_stats = []
        for model, data in model_data.items():
            if data["placements"]:
                model_stats.append(
                    {
                        "model": model,
                        "avg_placement": statistics.mean(data["placements"]),
                        "data": data,
                    }
                )

        model_stats.sort(key=lambda x: x["avg_placement"])

        for item in model_stats:
            model = item["model"]
            data = item["data"]
            placements = data["placements"]
            chips = data["final_chips"]

            print(f"{model}:")
            print(f"  Games played: {len(placements)}")
            print(f"  Average placement: {statistics.mean(placements):.2f}")
            print(
                f"  Placement std dev: {statistics.stdev(placements) if len(placements) > 1 else 0:.2f}"
            )
            print(f"  Best: {min(placements)}, Worst: {max(placements)}")
            print(f"  Average final chips: {statistics.mean(chips):.1f}")
            print(
                f"  Chips std dev: {statistics.stdev(chips) if len(chips) > 1 else 0:.1f}"
            )
            print(f"  Seats played: {sorted(data['seats_played'])}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze poker game logs (single file or entire folder)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single log file
  %(prog)s logs/seed_42/seating_0.json --verbose
  
  # Analyze entire folder (aggregate by seat)
  %(prog)s logs/seed_42 --aggregate-by seat
  
  # Analyze folder (aggregate by model)
  %(prog)s logs/seed_42 --aggregate-by model
        """,
    )

    # Required argument (file or folder)
    parser.add_argument(
        "path",
        type=str,
        help="Path to a game log JSON file or folder containing multiple logs",
    )

    # Folder analysis option
    parser.add_argument(
        "--aggregate-by",
        choices=["seat", "model", "all"],
        help="Aggregate statistics across multiple log files by seat position or model name. Requires path to be a folder.",
    )

    # Optional arguments
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed action-by-action breakdown (single file mode only)",
    )

    parser.add_argument(
        "--plot-hands",
        action="store_true",
        help="Show token usage per hand (recommended)",
    )

    parser.add_argument(
        "--save-hands",
        metavar="FILE",
        help="Save the hand tokens plot to FILE instead of showing it",
    )

    parser.add_argument(
        "--plot-round-tokens",
        action="store_true",
        help="Show per-round token usage plot (every N actions, deprecated)",
    )

    parser.add_argument(
        "--save-round-tokens",
        metavar="FILE",
        help="Save the per-round token plot to FILE instead of showing it",
    )

    parser.add_argument(
        "--plot-chips",
        action="store_true",
        help="Show per-round chip stacks for each player",
    )

    parser.add_argument(
        "--save-chips",
        metavar="FILE",
        help="Save the chips plot to FILE instead of showing it",
    )

    parser.add_argument(
        "--token-mode",
        choices=["input", "output", "total"],
        default="output",
        help="Token calculation mode: 'input' (input tokens only), 'output' (output + reasoning tokens), or 'total' (input + output + reasoning). Default: output",
    )

    parser.add_argument(
        "--show-costs",
        action="store_true",
        help="Show cost analysis in addition to token analysis",
    )

    parser.add_argument(
        "--cost-mode",
        choices=["input", "output", "total"],
        default="total",
        help="Cost calculation mode: 'input' (input token costs only), 'output' (output + reasoning token costs), or 'total' (input + output + reasoning costs). Default: total",
    )

    args = parser.parse_args()

    # Check if path is file or folder
    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path '{args.path}' not found")
        sys.exit(1)

    # Folder analysis mode
    if path.is_dir():
        if not args.aggregate_by:
            print("Error: --aggregate-by required when analyzing a folder")
            print("Use: --aggregate-by seat  or  --aggregate-by model")
            sys.exit(1)

        logs = load_logs_from_folder(path)
        if not logs:
            print(f"No valid log files found in {path}")
            sys.exit(1)

        print(f"Loaded {len(logs)} log files from {path}")
        print()

        analyze_folder_aggregate(logs, args.aggregate_by)
        return

    # Single file analysis mode
    try:
        with open(path, "r") as f:
            log = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{args.path}' not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: '{args.path}' is not valid JSON")
        sys.exit(1)

    # Print analysis
    print_session_summary(log)

    for hand in log["hands"]:
        print_hand_summary(hand, log, verbose=args.verbose)

    print_player_stats(
        log, args.token_mode, show_costs=args.show_costs, cost_mode=args.cost_mode
    )

    # Generate plots if requested
    if args.plot_hands or args.save_hands:
        maybe_plot_hand_tokens(log, args.save_hands, args.token_mode)

    if args.plot_round_tokens or args.save_round_tokens:
        maybe_plot_round_tokens(log, args.save_round_tokens, args.token_mode)

    if args.plot_chips or args.save_chips:
        maybe_plot_round_chips(log, args.save_chips)


if __name__ == "__main__":
    main()
