"""utils for poker game management and resumption."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from gpt_agent import PokerAgent
from poker_table import PokerTable

logger = logging.getLogger(__name__)


def build_game_context_from_log(game_log: dict) -> List[dict]:
    """reconstruct game_context from complete game log for resuming."""
    context: List[dict] = []
    for hand in game_log.get("hands", []):
        hand_num = hand.get("hand_number", 0)
        for idx, action_entry in enumerate(hand.get("actions", [])):
            context.append(
                {
                    "hand_num": hand_num,
                    "action_num": idx,
                    "player": action_entry.get("player", "Unknown"),
                    "action": action_entry.get("action", {}),
                    "board": action_entry.get("board", []),
                }
            )
    return context


def load_game_log(log_file_path: str) -> dict:
    """load game log from json file."""
    log_path = Path(log_file_path)

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_file_path}")

    with open(log_path, "r") as f:
        game_log = json.load(f)

    required_keys = ["hands", "session_info"]
    for key in required_keys:
        if key not in game_log:
            raise ValueError(f"Invalid log structure: missing '{key}' field")

    if not game_log["hands"]:
        raise ValueError("Log file contains no hands")

    logger.info(f"Loaded game log from {log_file_path}")

    return game_log


def extract_agent_history(game_log: dict, player_name: str, seat: int) -> List[dict]:
    """extract agent's reasoning history from game log."""
    reasoning_history: List[dict] = []

    for hand in game_log["hands"]:
        hand_num = hand["hand_number"]

        for action_entry in hand.get("actions", []):
            if action_entry.get("player") == player_name:
                action_num = action_entry.get("timestep", 0)

                reasoning = ""
                if "model_response" in action_entry:
                    reasoning = action_entry["model_response"].get("reasoning", "")

                history_entry = {
                    "hand_num": hand_num,
                    "action_num": action_num,
                    "reasoning": reasoning,
                    "action": action_entry["action"],
                }
                reasoning_history.append(history_entry)

    logger.debug(f"Extracted {len(reasoning_history)} actions for {player_name}")
    return reasoning_history


def reconstruct_game_state(game_log: dict) -> Tuple[dict, dict]:
    """extract current game state from log."""

    session_info = game_log["session_info"]
    last_hand = game_log["hands"][-1]

    player_names = session_info["player_names"]
    current_stacks = [last_hand["ending_stacks"].get(name, 0) for name in player_names]

    last_button = last_hand["button_seat"]
    num_players = len(player_names)
    ending_stacks = last_hand["ending_stacks"]

    if num_players > 1:
        probe = (last_button + 1) % num_players
        for _ in range(num_players):
            name_at_seat = player_names[probe]
            if ending_stacks.get(name_at_seat, 0) > 0:
                next_button = probe
                break
            probe = (probe + 1) % num_players
        else:
            next_button = last_button
    else:
        next_button = last_button

    hands_played = len(game_log["hands"])

    table_state = {
        "current_stacks": current_stacks,
        "button_seat": next_button,
        "hands_played": hands_played,
    }

    logger.info(f"Game state after {hands_played} hands:")
    for name, stack in zip(player_names, current_stacks):
        logger.info(f"  {name}: {stack} chips")
    logger.info(f"  Next button seat: {next_button}")

    return table_state, session_info


def resume_game_from_log(
    log_file_path: str,
    model_configs: Dict[str, str],
) -> Tuple[PokerTable, Dict[int, PokerAgent], str]:
    """Resume poker game from log with full agent context."""
    game_log = load_game_log(log_file_path)

    table_state, session_info = reconstruct_game_state(game_log)

    player_names = session_info["player_names"]

    for name in player_names:
        if name not in model_configs:
            raise ValueError(f"No model config provided for player '{name}'")

    table = PokerTable(
        player_names=player_names,
        starting_stacks=table_state["current_stacks"],
        small_blind=session_info["small_blind"],
        big_blind=session_info["big_blind"],
        seed=session_info["seed"],
    )

    table.stacks = list(table_state["current_stacks"])
    table.button_seat = table_state["button_seat"]
    table.hand_count = table_state["hands_played"]

    table.game_log = game_log
    table.game_context = build_game_context_from_log(game_log)

    agents = {}
    for seat, name in enumerate(player_names):
        reasoning_history = extract_agent_history(game_log, name, seat)

        agent = PokerAgent(
            seat=seat,
            player_name=name,
            model_str=model_configs[name],
        )

        agent.reasoning_history = reasoning_history

        agents[seat] = agent
        logger.info(
            f"restored agent {name} (seat {seat}) with {len(reasoning_history)} past actions"
        )

    return table, agents, log_file_path


def print_game_summary(game_log: dict):
    """print game summary."""
    session = game_log["session_info"]

    print("GAME LOG SUMMARY")
    print("=" * 70)
    print(f"Players: {', '.join(session['player_names'])}")
    print(f"Starting stacks: {session['starting_stacks']}")

    if game_log["hands"]:
        last_hand = game_log["hands"][-1]
        print("\nCurrent stacks:")
        for name in session["player_names"]:
            stack = last_hand["ending_stacks"].get(name, 0)
            starting = session["starting_stacks"][session["player_names"].index(name)]
            profit = stack - starting
            print(f"  {name}: {stack} ({profit:+d})")

    print("=" * 70 + "\n")


def validate_log_for_resume(log_file_path: str) -> bool:
    try:
        game_log = load_game_log(log_file_path)

        last_hand = game_log["hands"][-1]
        required_hand_fields = ["ending_stacks", "button_seat", "hand_number"]
        for field in required_hand_fields:
            if field not in last_hand:
                logger.error(f"Last hand missing field: {field}")
                return False

        player_names = game_log["session_info"]["player_names"]
        ending_stacks = last_hand["ending_stacks"]
        for name in player_names:
            if name not in ending_stacks:
                logger.error(f"Missing ending stack for player: {name}")
                return False

        logger.info("Log file is valid for resuming")
        return True

    except Exception as e:
        logger.error(f"Log validation failed: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python utils.py <log_file_path>")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    log_path = sys.argv[1]

    print(f"Validating log file: {log_path}")
    if validate_log_for_resume(log_path):
        game_log = load_game_log(log_path)
        print_game_summary(game_log)
    else:
        print("Log file is not valid for resuming")
        sys.exit(1)
