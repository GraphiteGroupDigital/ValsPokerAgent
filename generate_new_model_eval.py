#!/usr/bin/env python3
"""Generate configurations for new model evaluation using info-score anchor selection."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List
import yaml


class TrueSkillRating:
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma


def load_trueskill_ratings(json_path: Path) -> Dict[str, TrueSkillRating]:
    with open(json_path, "r") as f:
        data = json.load(f)

    ratings = {}

    if "raw_trueskill_ratings" in data:
        for model_name, rating_data in data["raw_trueskill_ratings"].items():
            ratings[model_name] = TrueSkillRating(
                rating_data["mu"], rating_data["sigma"]
            )
    else:
        for entry in data["rankings"]:
            model_name = entry["model"]
            scaled_mu = entry["rating"]
            scaled_sigma = entry["rating_sigma"]

            raw_mu = (scaled_mu - 1000.0) / 40.0 + 25.0
            raw_sigma = scaled_sigma / 40.0

            ratings[model_name] = TrueSkillRating(raw_mu, raw_sigma)

    return ratings


def load_model_mapping(config_path: Path) -> Dict[str, str]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    mapping = {}
    for player in config["players"]:
        mapping[player["name"]] = player["model"]

    return mapping


def info_score(
    mu_a: float, sigma_a: float, mu_n: float, sigma_n: float, s: float = 30.0
) -> float:
    return (sigma_a + sigma_n) * math.exp(-abs(mu_a - mu_n) / s)


def select_best_anchors(
    new_models: List[str],
    anchor_pool: List[str],
    ratings: Dict[str, TrueSkillRating],
    num_anchors: int,
    s: float = 30.0,
    top_k: int = 20,
) -> List[str]:
    anchor_scores = []

    for anchor in anchor_pool:
        mu_a = ratings[anchor].mu
        sigma_a = ratings[anchor].sigma

        total_score = sum(
            info_score(mu_a, sigma_a, ratings[nm].mu, ratings[nm].sigma, s)
            for nm in new_models
        )
        anchor_scores.append((total_score, anchor))

    anchor_scores.sort(reverse=True)
    top_k_actual = min(top_k, len(anchor_scores))
    top_candidates = anchor_scores[:top_k_actual]

    scores = [score for score, _ in top_candidates]
    candidates = [anchor for _, anchor in top_candidates]

    selected = []
    remaining_candidates = list(zip(candidates, scores))

    for _ in range(num_anchors):
        if not remaining_candidates:
            break

        cands, weights = zip(*remaining_candidates)
        chosen = random.choices(cands, weights=weights, k=1)[0]
        selected.append(chosen)

        remaining_candidates = [(c, w) for c, w in remaining_candidates if c != chosen]

    random.shuffle(selected)
    return selected


def extract_seeds_from_configs(config_dir: Path) -> Dict[int, Path]:
    seed_to_file = {}
    for yaml_file in config_dir.rglob("*.yaml"):
        try:
            with open(yaml_file, "r") as f:
                config = yaml.safe_load(f)
                seed = config.get("game", {}).get("seed")
                if seed is not None:
                    seed_to_file[int(seed)] = yaml_file
        except Exception:
            continue
    return seed_to_file


def generate_config(
    players: List[str],
    model_mapping: Dict[str, str],
    seed: int,
    num_hands: int,
    starting_stack: int,
    small_blind: int,
    big_blind: int,
    output_file: str,
) -> Dict[str, Any]:
    return {
        "game": {
            "num_hands": num_hands,
            "small_blind": small_blind,
            "big_blind": big_blind,
            "seed": seed,
        },
        "players": [
            {
                "name": model_name,
                "model": model_mapping[model_name],
                "seat": i,
                "starting_stack": starting_stack,
            }
            for i, model_name in enumerate(players)
        ],
        "logging": {"output_file": output_file, "verbose": True, "level": "INFO"},
    }


def save_config(config: Dict[str, Any], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_new_models_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Generate configs for new model evaluation with adaptive anchor selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  %(prog)s --new-models-config configs/new_models_config.yaml
  
This script:
  1. Always includes ALL new models in every game
  2. Selects best anchor models using info-score
  3. Uses current TrueSkill ratings (run in batches for adaptation)
        """,
    )

    parser.add_argument(
        "--new-models-config",
        type=Path,
        default=Path("configs/new_models_config.yaml"),
        help="Config file with new models and settings",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for generated configs",
    )
    parser.add_argument(
        "--num-games", type=int, required=True, help="Number of games to generate"
    )
    parser.add_argument(
        "--reused-seeds",
        type=int,
        required=True,
        help="Number of reused seeds from existing configs",
    )
    parser.add_argument(
        "--new-seeds", type=int, required=True, help="Number of new seeds to generate"
    )
    parser.add_argument(
        "--seed-start", type=int, required=True, help="Starting seed for new seeds"
    )

    args = parser.parse_args()

    if not args.new_models_config.exists():
        print(f"Error: Config file not found: {args.new_models_config}")
        return 1

    print("=" * 80)
    print("NEW MODEL EVALUATION CONFIG GENERATOR")
    print("=" * 80)
    print()

    print(f"Loading configuration from {args.new_models_config}...")
    config = load_new_models_config(args.new_models_config)

    new_models_data = config.get("new_models", [])
    new_model_names = [m["name"] for m in new_models_data]

    eval_config = config.get("evaluation", {})
    num_hands = eval_config.get("num_hands", 25)
    players_per_game = eval_config.get("players_per_game", 10)
    starting_stack = eval_config.get("starting_stack", 200)
    small_blind = eval_config.get("small_blind", 1)
    big_blind = eval_config.get("big_blind", 2)

    reused_seeds = args.reused_seeds
    new_seeds = args.new_seeds
    total_games = args.num_games

    if reused_seeds + new_seeds != total_games:
        print(
            f"Error: reused_seeds ({reused_seeds}) + new_seeds ({new_seeds}) != num_games ({total_games})"
        )
        return 1

    info_config = config.get("info_score", {})
    decay_rate = info_config.get("decay_rate", 30.0)
    output_dir = args.output_dir

    ratings_file = Path("analysis/results/trueskill_rankings.json")
    template_config = Path("configs/config.yaml")
    existing_configs_dir = Path("configs/final")

    if len(new_model_names) >= players_per_game:
        print(
            f"Error: Too many new models ({len(new_model_names)}) for {players_per_game} players"
        )
        return 1

    print(f"New models: {', '.join(new_model_names)}")
    print(f"Generating {total_games} games: {reused_seeds} reused + {new_seeds} new")
    print(f"Output: {output_dir}")
    print()

    print(f"Loading TrueSkill ratings from {ratings_file}...")
    ratings = load_trueskill_ratings(ratings_file)
    print(f"Loaded ratings for {len(ratings)} models")

    print(f"Loading model mappings from {template_config}...")
    model_mapping = load_model_mapping(template_config)
    print(f"Loaded {len(model_mapping)} existing model mappings")

    for model_data in new_models_data:
        name = model_data["name"]
        api_string = model_data.get("model")
        if api_string and name not in model_mapping:
            model_mapping[name] = api_string
            print(f"  Added new model mapping: {name} -> {api_string}")

    missing = [m for m in new_model_names if m not in model_mapping]
    if missing:
        print(f"\nError: New models missing API strings: {', '.join(missing)}")
        print(
            "Add them to either configs/config.yaml or configs/new_models_config.yaml"
        )
        return 1

    for new_model in new_model_names:
        if new_model not in ratings:
            print(f"Initializing rating for new model: {new_model}")
            ratings[new_model] = TrueSkillRating(mu=25.0, sigma=8.333)

    print(f"\nNew Models ({len(new_model_names)}):")
    for m in new_model_names:
        print(f"  • {m}: μ={ratings[m].mu:.2f}, σ={ratings[m].sigma:.2f}")

    anchor_pool = [m for m in ratings.keys() if m not in new_model_names]
    num_anchors = players_per_game - len(new_model_names)
    print(f"\nAnchor Pool: {len(anchor_pool)} models")
    print(f"Will select {num_anchors} best anchors per game")

    reused_seed_list = []
    seed_to_config_file = {}
    if existing_configs_dir.exists():
        print(f"\nExtracting seeds from {existing_configs_dir}...")
        seed_to_file = extract_seeds_from_configs(existing_configs_dir)
        print(f"Found {len(seed_to_file)} unique seeds")

        if seed_to_file:
            num_reused = min(reused_seeds, len(seed_to_file))
            reused_seed_list = random.sample(list(seed_to_file.keys()), num_reused)
            reused_seed_list.sort()
            seed_to_config_file = {seed: seed_to_file[seed] for seed in reused_seed_list}
            print(f"Selected {len(reused_seed_list)} random seeds to reuse")

    new_seed_list = list(range(args.seed_start, args.seed_start + new_seeds))
    all_seeds = reused_seed_list + new_seed_list

    print(f"\nTotal Games: {len(all_seeds)}")
    print(f"  - {len(reused_seed_list)} reused seeds: {reused_seed_list}")
    print(
        f"  - {len(new_seed_list)} new seeds: {args.seed_start} to {args.seed_start + new_seeds - 1}"
    )
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    for game_num, seed in enumerate(all_seeds):
        is_reused = seed in reused_seed_list
        seed_type = "REUSED" if is_reused else "NEW"

        print(f"Game {game_num + 1}/{len(all_seeds)} ({seed_type} seed={seed}):")

        if is_reused:
            original_config_path = seed_to_config_file[seed]
            original_filename = original_config_path.stem
            seating_num = original_filename.split("_")[-1] if "_" in original_filename else "0"

            print(f"  Loading original config: {original_config_path}")

            with open(original_config_path, "r") as f:
                original_config = yaml.safe_load(f)

            original_players = sorted(original_config["players"], key=lambda p: p["seat"])

            num_new = len(new_model_names)
            seats_to_replace = random.sample(range(players_per_game), num_new)
            seats_to_replace.sort()

            batch_num = output_dir.name.split("batch_")[1]
            config = {
                "game": {
                    "num_hands": num_hands,
                    "small_blind": small_blind,
                    "big_blind": big_blind,
                    "seed": seed,
                },
                "players": [],
                "logging": {
                    "output_file": f"logs/new_model_eval/batch_{batch_num}_game_{game_num}_seed_{seed}.json",
                    "verbose": True,
                    "level": "INFO",
                },
            }

            new_model_idx = 0
            for seat in range(players_per_game):
                if seat in seats_to_replace:
                    config["players"].append(
                        {
                            "name": new_model_names[new_model_idx],
                            "model": model_mapping[new_model_names[new_model_idx]],
                            "seat": seat,
                            "starting_stack": starting_stack,
                        }
                    )
                    new_model_idx += 1
                else:
                    original_player = original_players[seat]
                    config["players"].append(
                        {
                            "name": original_player["name"],
                            "model": original_player["model"],
                            "seat": seat,
                            "starting_stack": starting_stack,
                        }
                    )

            kept_models = [
                original_players[i]["name"]
                for i in range(players_per_game)
                if i not in seats_to_replace
            ]
            print(
                f"  New models (seats {seats_to_replace}): {', '.join(new_model_names)}"
            )
            print(f"  Kept original: {', '.join(kept_models)}")

        else:
            print(f"  New models: {', '.join(new_model_names)}")

            selected_anchors = select_best_anchors(
                new_model_names, anchor_pool, ratings, num_anchors, s=decay_rate, top_k=20
            )
            print(f"  Selected anchors: {', '.join(selected_anchors)}")

            all_players = new_model_names + selected_anchors
            random.shuffle(all_players)
            print(f"  Seating order: {', '.join(all_players)}")

            log_file = f"logs/new_model_eval/game_{game_num}_seed_{seed}.json"
            config = generate_config(
                all_players,
                model_mapping,
                seed,
                num_hands,
                starting_stack,
                small_blind,
                big_blind,
                log_file,
            )

        if is_reused:
            suffix = f"_seed_{seed}_seating_{seating_num}"
            output_path = output_dir / f"game_{game_num}{suffix}.yaml"
        else:
            output_path = output_dir / f"game_{game_num}_seed_{seed}.yaml"
        save_config(config, output_path)

        if not is_reused:
            anchor_info_scores = []
            for anchor in selected_anchors:
                mu_a = ratings[anchor].mu
                sigma_a = ratings[anchor].sigma
                total_score = sum(
                    info_score(
                        mu_a, sigma_a, ratings[nm].mu, ratings[nm].sigma, decay_rate
                    )
                    for nm in new_model_names
                )
                anchor_info_scores.append((anchor, total_score))

            anchor_info_scores.sort(key=lambda x: x[1], reverse=True)

            print("  Info scores (top 3): ", end="")
            for anchor, score in anchor_info_scores[:3]:
                print(f"{anchor}={score:.2f} ", end="")
            print()

            player_mus = [ratings[p].mu for p in all_players]
            print(
                f"  Rating spread: μ∈[{min(player_mus):.2f}, {max(player_mus):.2f}], Δ={max(player_mus) - min(player_mus):.2f}"
            )

        print(f"  Saved to {output_path}")
        print()

    print(f"Generated {len(all_seeds)} configs in {output_dir}/")
    print(f"  New models: {', '.join(new_model_names)}")
    print(f"  {num_hands} hands per game")

    return 0


if __name__ == "__main__":
    exit(main())
