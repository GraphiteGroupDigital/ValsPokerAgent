#!/usr/bin/env python3
"""Generate poker tournament configs with different seating arrangements and seeds."""

import yaml
import random
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any


def load_base_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def shuffle_players(players: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    random.seed(seed)
    shuffled_players = players.copy()
    random.shuffle(shuffled_players)

    for i, player in enumerate(shuffled_players):
        player["seat"] = i

    return shuffled_players


def generate_variation(
    base_config: Dict[str, Any], seating_num: int, seed: int
) -> Dict[str, Any]:
    config = base_config.copy()
    config["game"]["seed"] = seed
    config["players"] = shuffle_players(config["players"], seating_num)
    config["logging"]["output_file"] = f"logs/seed_{seed}/seating_{seating_num}.json"
    return config


def save_config(config: Dict[str, Any], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def print_player_order(config: Dict[str, Any], seating_num: int):
    print(f"Seating {seating_num}:")
    for i, player in enumerate(config["players"]):
        print(f"  Seat {i}: {player['name']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate poker tournament config variations"
    )
    parser.add_argument(
        "--num-seating",
        type=int,
        default=9,
        help="Number of seating arrangements to generate (default: 9)",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of different seeds to use (default: 5)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed for variations (default: 42)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/config.yaml",
        help="Path to base config file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="configs",
        help="Output directory for generated configs (default: configs)",
    )

    args = parser.parse_args()

    total_configs = args.num_seating * args.num_seeds
    print(f"Generating {total_configs} poker tournament config variations...")
    print(f"Seating arrangements: {args.num_seating}")
    print(f"Different seeds: {args.num_seeds}")
    print(f"Base seed: {args.base_seed}")
    print(f"Base config: {args.config_path}")
    print(f"Output directory: {args.output_dir}")
    print()

    try:
        base_config = load_base_config(args.config_path)
    except FileNotFoundError:
        print(f"Error: Base config file '{args.config_path}' not found!")
        return
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in '{args.config_path}': {e}")
        return

    variations = []
    seeds = [args.base_seed + i * 1000 for i in range(args.num_seeds)]

    for seed in seeds:
        print(f"=== SEED {seed} ===")
        seed_variations = []

        for seating_num in range(args.num_seating):
            config = generate_variation(base_config, seating_num, seed)

            output_path = os.path.join(
                args.output_dir, f"seed_{seed}", f"seating_{seating_num}.yaml"
            )
            save_config(config, output_path)

            variation_info = {
                "seating": seating_num,
                "seed": seed,
                "output_file": config["logging"]["output_file"],
                "config_path": output_path,
            }
            variations.append(variation_info)
            seed_variations.append(variation_info)

            print_player_order(config, seating_num)

        print(f"Generated {args.num_seating} seating arrangements for seed {seed}")
        print()

    summary_path = os.path.join(args.output_dir, "variations_summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(
            {
                "total_configs": total_configs,
                "num_seating": args.num_seating,
                "num_seeds": args.num_seeds,
                "seeds": seeds,
                "variations": variations,
            },
            f,
            default_flow_style=False,
        )

    print(f"Generated {total_configs} config variations!")
    print(f"Summary saved to: {summary_path}")
    print()
    print("Directory structure:")
    for seed in seeds:
        print(f"  configs/seed_{seed}/")
        for seating in range(args.num_seating):
            print(f"    seating_{seating}.yaml")
    print()
    print("To run a specific variation:")
    print(f"  python llm_agent_play.py --config configs/seed_{seeds[0]}/seating_0.yaml")
    print()
    print("To run all variations for a specific seed:")
    print(
        f"  for i in {{0..{args.num_seating - 1}}}; do python llm_agent_play.py --config configs/seed_{seeds[0]}/seating_$i.yaml; done"
    )
    print()
    print("To run all variations:")
    for seed in seeds:
        print(f"  # Seed {seed}")
        print(
            f"  for i in {{0..{args.num_seating - 1}}}; do python llm_agent_play.py --config configs/seed_{seed}/seating_$i.yaml; done"
        )


if __name__ == "__main__":
    main()
