"""resumes poker game from log file and continues playing."""

import asyncio
import logging
import sys
import argparse
from pathlib import Path

from utils import resume_game_from_log, print_game_summary, load_game_log

logger = logging.getLogger(__name__)


def create_model_router(agents):
    def model_router(seat: int, observation: dict, game_context: list):
        agent = agents.get(seat)
        if agent is None:
            raise ValueError(f"No agent configured for seat {seat}")

        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            agent.get_action(observation, game_context, return_response=True)
        )

    return model_router


def main():
    parser = argparse.ArgumentParser(description="Resume a poker game from a log file")
    parser.add_argument("log_file", help="Path to the JSON log file to resume from")
    parser.add_argument(
        "--to-total",
        type=int,
        default=100,
        help="Play until reaching this total number of hands (default: 100)",
    )
    parser.add_argument(
        "--additional-hands",
        type=int,
        default=None,
        help="Play an explicit number of additional hands (overrides --to-total)",
    )
    args = parser.parse_args()

    log_file = args.log_file

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not Path(log_file).exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)

    print("\nCurrent Game State:")
    game_log = load_game_log(log_file)
    print_game_summary(game_log)
    model_configs = {}
    for hand in game_log["hands"]:
        for action in hand.get("actions", []):
            if "model_response" in action:
                player_name = action["player"]
                provider = action["model_response"].get("provider", "").lower()
                model_name = action["model_response"].get("model_name", "")
                if provider and model_name:
                    if provider == "fireworks":
                        if "/models/" in model_name:
                            model_name = model_name.split("/models/", 1)[1]
                        if "/" in model_name:
                            model_name = model_name.rsplit("/", 1)[-1]
                        registry_key = f"fireworks/{model_name}"
                    else:
                        registry_key = f"{provider}/{model_name}"
                    model_configs[player_name] = registry_key

    if not model_configs:
        print("Error: Could not extract model configurations from log")
        sys.exit(1)

    table, agents, log_path = resume_game_from_log(log_file, model_configs)
    model_router = create_model_router(agents)
    initial_stacks = list(table.stacks)
    starting_hand = table.hand_count + 1
    hands_already_played = table.hand_count
    if args.additional_hands is not None:
        additional_hands = max(0, args.additional_hands)
        target_hand = hands_already_played + additional_hands
        mode_msg = f"Playing {additional_hands} more hands"
    else:
        target_total = max(0, args.to_total)
        additional_hands = max(0, target_total - hands_already_played)
        target_hand = target_total
        mode_msg = f"Playing until total {target_total} hands"

    logger.info("=" * 70)
    logger.info(f"Continuing game: {mode_msg}")
    logger.info(f"Hands {starting_hand} to {target_hand}")
    logger.info("=" * 70)

    for hand_num in range(additional_hands):
        current_hand = table.hand_count + 1
        logger.info(f"\n{'-' * 20}")
        logger.info(
            f"Hand {current_hand} (Additional hand {hand_num + 1}/{additional_hands})"
        )
        logger.info(f"{'-' * 20}")

        table.play_hand(model_router)
        table.save_game_log(log_path)

        if not table.can_continue():
            logger.info("\nGAME OVER - Not enough chips to continue")
            break

    summary = table.get_session_summary(initial_stacks)
    results = sorted(summary["final_stacks"].items(), key=lambda x: x[1], reverse=True)

    logger.info("Current Standings:")
    for rank, (name, stack) in enumerate(results, 1):
        profit = summary["total_profit_loss"].get(name, 0)
        logger.info(f"{rank}. {name}: {stack} chips ({profit:+d} this session)")



if __name__ == "__main__":
    main()
