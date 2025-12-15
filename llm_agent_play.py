"""Multi-model LLM poker with continuous logging."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, List

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from gpt_agent import PokerAgent
from poker_table import PokerTable

logger = logging.getLogger(__name__)

# creates model router to map models to poker agents
def create_model_router(agents: Dict[int, PokerAgent]):
    def model_router(seat: int, observation: dict, game_context: List[dict]):
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

"""run game with models and store full logs to json."""
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.getLogger().setLevel(getattr(logging, cfg.logging.level))

    players_sorted = sorted(cfg.players, key=lambda p: p.seat)
    player_names = [p.name for p in players_sorted]
    starting_stacks = [p.starting_stack for p in players_sorted]

    # store PokerAgent instances for each player
    agents = {}
    for player in cfg.players:
        agent = PokerAgent(
            seat=player.seat,
            player_name=player.name,
            model_str=player.model,
        )
        agents[player.seat] = agent
        logger.info(f"  - {player.name} (Seat {player.seat}): {player.model}")

    logger.info("Make sure you have the required API keys set in your environment.\n")

    # router creation
    model_router = create_model_router(agents)

    table = PokerTable(
        player_names=player_names,
        starting_stacks=starting_stacks,
        small_blind=cfg.game.small_blind,
        big_blind=cfg.game.big_blind,
        seed=cfg.game.seed,
    )

    # output file setup
    log_filename = cfg.logging.output_file
    log_dir = os.path.dirname(log_filename)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Log file: {log_filename}")

    initial_stacks = list(starting_stacks)

    with tqdm(total=cfg.game.num_hands, desc="Playing hands", unit="hand") as pbar:
        for hand_num in range(1, cfg.game.num_hands + 1):
            logger.info(f"\n{'-' * 20}")
            logger.info(f"Hand {hand_num}/{cfg.game.num_hands}")
            logger.info(f"{'-' * 20}")

            # check tournament status
            active_players = [
                name
                for name, stack in zip(table.player_names, table.stacks)
                if stack > 0
            ]
            if len(active_players) <= 1:
                logger.info(
                    f"\nTOURNAMENT OVER - {len(active_players)} player(s) remaining"
                )
                break

            table.play_hand(model_router)
            
            # log after each hand
            table.save_game_log(log_filename)
            logger.debug(f"Log updated: {hand_num} hands")

            pbar.update(1)
            pbar.set_postfix({"Hand": hand_num, "Log": os.path.basename(log_filename)})

    logger.info(f"\n{'=' * 70}")
    logger.info("TOURNAMENT RESULTS")
    logger.info("=" * 70)
    summary = table.get_session_summary(initial_stacks)
    logger.info(f"Hands played: {summary['hands_played']}")
    logger.info("")

    results = sorted(
        summary["total_profit_loss"].items(), key=lambda x: x[1], reverse=True
    )

    logger.info("Final Standings:")
    for rank, (name, profit) in enumerate(results, 1):
        final = summary["final_stacks"][name]
        initial = initial_stacks[rank - 1]
        percentage = (profit / initial) * 100 if initial > 0 else 0
        logger.info(f"{rank}. {name}: {final} chips ({profit:+d}, {percentage:+.1f}%)")

    # find winner
    winner = results[0]
    logger.info(f"\nWINNER: {winner[0]} with {winner[1]:+d} chip profit!")

    table.save_game_log(log_filename)
    logger.info(f"\nFinal game log saved to {log_filename}")

    if cfg.logging.verbose:
        logger.info("\nThe log contains:")
        logger.info("  - Starting state with hole cards and positions for each hand")
        logger.info("  - All player actions with chip changes and stack sizes")
        logger.info("  - Model responses with reasoning for each model's decisions")
        logger.info("  - Final board state and ending stacks for each hand")
        logger.info("  - Complete session metadata")


if __name__ == "__main__":
    main()
