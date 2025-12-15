#!/usr/bin/env python
"""Validate poker game logs for consistency and correctness."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict


def load_log(filepath: str) -> dict:
    with open(filepath, "r") as f:
        return json.load(f)


def check_seat_invariance(logs: Dict[str, dict], seed: str) -> bool:
    print("=" * 70)
    print(f"CHECK 1: Seat Invariance for {seed}")
    print("=" * 70)

    seating_names = sorted(logs.keys())

    if len(seating_names) < 2:
        print("Need at least 2 seatings to compare")
        return False

    player_names = logs[seating_names[0]]["session_info"]["player_names"]
    num_players = len(player_names)
    all_passed = True
    max_hands = min(log["session_info"]["total_hands"] for log in logs.values())

    for hand_num in range(1, max_hands + 1):
        hand_cards_by_seating = {}

        for seating_name, log in logs.items():
            if hand_num <= len(log["hands"]):
                hand_data = log["hands"][hand_num - 1]
                starting_state = hand_data["starting_state"]

                cards_by_seat = {
                    data["seat"]: data["hole_cards"]
                    for player_name, data in starting_state.items()
                }
                hand_cards_by_seating[seating_name] = cards_by_seat

        reference_seating = seating_names[0]
        reference_cards = hand_cards_by_seating[reference_seating]

        for seating_name in seating_names[1:]:
            compare_cards = hand_cards_by_seating[seating_name]

            for seat in range(num_players):
                ref = reference_cards.get(seat, [])
                cmp = compare_cards.get(seat, [])

                if ref and cmp and ref != cmp:
                    print(f"Hand {hand_num}, Seat {seat}: Cards mismatch!")
                    print(f"   {reference_seating}: {ref}")
                    print(f"   {seating_name}: {cmp}")
                    all_passed = False

    if all_passed:
        print(
            f"✓ All {max_hands} hands: Seat invariance verified across {len(seating_names)} seatings"
        )

    return all_passed


def check_blind_rotation(log: dict, log_name: str) -> bool:
    print("\n" + "=" * 70)
    print(f"CHECK 2: Blind Rotation for {log_name}")
    print("=" * 70)

    player_names = log["session_info"]["player_names"]
    num_players = len(player_names)
    all_passed = True

    for hand_idx, hand_data in enumerate(log["hands"]):
        hand_num = hand_data["hand_number"]
        button_seat = hand_data["button_seat"]
        starting_state = hand_data["starting_state"]

        active_seats = sorted(
            [data["seat"] for player_name, data in starting_state.items() if data["starting_stack"] > 0]
        )

        if len(active_seats) < 2:
            continue

        button_idx = active_seats.index(button_seat)

        if len(active_seats) == 2:
            expected_bb_seat = button_seat
            expected_sb_seat = active_seats[(button_idx + 1) % 2]
            expected_positions = {button_seat: 0, expected_sb_seat: 1}
        else:
            expected_sb_seat = active_seats[(button_idx + 1) % len(active_seats)]
            expected_bb_seat = active_seats[(button_idx + 2) % len(active_seats)]
            expected_positions = {expected_sb_seat: 0, expected_bb_seat: 1}
        for player_name, data in starting_state.items():
            seat = data["seat"]
            position = data["position"]

            if seat in expected_positions:
                expected_pos = expected_positions[seat]
                if position != expected_pos:
                    print(
                        f" Hand {hand_num}: {player_name} (seat {seat}) has position {position}, expected {expected_pos}"
                    )
                    print(f"   Button={button_seat}, Active={active_seats}")
                    all_passed = False

        if button_seat not in active_seats:
            print(
                f" Hand {hand_num}: Button {button_seat} not on active player! Active={active_seats}"
            )
            all_passed = False

    if all_passed:
        print(f"All {len(log['hands'])} hands: Blind rotation correct")

    return all_passed


def check_no_resurrection(log: dict, log_name: str) -> bool:
    print("\n" + "=" * 70)
    print(f"CHECK 3: No Resurrection for {log_name}")
    print("=" * 70)

    player_names = log["session_info"]["player_names"]
    player_stacks = {
        name: log["session_info"]["starting_stacks"][i]
        for i, name in enumerate(player_names)
    }
    eliminated_at_hand = {}
    all_passed = True

    for hand_data in log["hands"]:
        hand_num = hand_data["hand_number"]
        ending_stacks = hand_data["ending_stacks"]

        for player_name in player_names:
            old_stack = player_stacks[player_name]
            new_stack = ending_stacks[player_name]

            if old_stack == 0 and new_stack > 0:
                print(
                    f" RESURRECTION BUG: {player_name} went from 0 to {new_stack} chips at hand {hand_num}!"
                )
                print(
                    f" Eliminated at hand: {eliminated_at_hand.get(player_name, 'unknown')}"
                )
                all_passed = False

            if old_stack > 0 and new_stack == 0:
                eliminated_at_hand[player_name] = hand_num

            player_stacks[player_name] = new_stack

    if all_passed:
        eliminated_count = len(eliminated_at_hand)
        print("No resurrection bugs found")
        if eliminated_count > 0:
            print(f"  {eliminated_count} players eliminated during game:")
            for player, hand in eliminated_at_hand.items():
                print(f"    - {player} at hand {hand}")

    return all_passed


def check_zero_sum(log: dict, log_name: str) -> bool:
    print("\n" + "=" * 70)
    print(f"CHECK 4: Zero-Sum Property for {log_name}")
    print("=" * 70)

    player_names = log["session_info"]["player_names"]
    sb = log["session_info"]["small_blind"]
    bb = log["session_info"]["big_blind"]
    all_passed = True
    failed_hands = []

    for hand_data in log["hands"]:
        hand_num = hand_data["hand_number"]
        starting = hand_data["starting_state"]
        ending = hand_data["ending_stacks"]

        profits = [
            ending[player_name] - starting[player_name]["starting_stack"]
            for player_name in player_names
        ]
        profit_sum = sum(profits)
        expected_sum = sb + bb

        if profit_sum != expected_sum:
            failed_hands.append((hand_num, profit_sum, expected_sum))
            all_passed = False

    if all_passed:
        print(f" All {len(log['hands'])} hands: Zero-sum property holds")
        print(f" (Each hand: ending - starting = {sb + bb} from blinds)")
    else:
        print(f" Found {len(failed_hands)} hands with incorrect sum:")
        for hand_num, actual, expected in failed_hands[:5]:
            print(f"   Hand {hand_num}: sum = {actual}, expected {expected}")

    return all_passed


def check_chip_conservation(log: dict, log_name: str) -> bool:
    print("\n" + "=" * 70)
    print(f"CHECK 5: Total Chip Conservation for {log_name}")
    print("=" * 70)

    session_start = sum(log["session_info"]["starting_stacks"])
    all_passed = True

    for hand_data in log["hands"]:
        hand_num = hand_data["hand_number"]
        total_chips = sum(hand_data["ending_stacks"].values())

        if total_chips != session_start:
            print(
                f" Hand {hand_num}: Total chips = {total_chips}, expected {session_start}"
            )
            all_passed = False
            break

    if all_passed:
        print(
            f"Total chips conserved: {session_start} chips across all {len(log['hands'])} hands"
        )

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Validate poker game logs for consistency and correctness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s seed_42                      # Check all seatings in seed_42
  %(prog)s seed_42 seating_0 seating_1  # Check specific seatings only
  %(prog)s seed_142                     # Check different seed
        """,
    )

    parser.add_argument("seed", help="Seed name (e.g., seed_42, seed_142)")
    parser.add_argument(
        "seatings",
        nargs="*",
        help="Specific seating names to check (e.g., seating_0 seating_1). If omitted, checks all.",
    )

    args = parser.parse_args()

    seed = args.seed
    log_dir = Path(f"logs/{seed}")

    if not log_dir.exists():
        print(f"Error: {log_dir} not found")
        sys.exit(1)

    if args.seatings:
        seating_files = [f"{s}.json" for s in args.seatings]
    else:
        seating_files = sorted([f.name for f in log_dir.glob("seating_*.json")])

    if not seating_files:
        print(f"No seating files found in {log_dir}")
        sys.exit(1)

    logs = {}
    for filename in seating_files:
        filepath = log_dir / filename
        if filepath.exists():
            seating_name = filename.replace(".json", "")
            logs[seating_name] = load_log(str(filepath))
            hands = logs[seating_name]["session_info"]["total_hands"]
            print(f"Loaded {seating_name}: {hands} hands")
        else:
            print(f"Warning: {filepath} not found, skipping")

    if not logs:
        print("No valid logs loaded")
        sys.exit(1)

    print()

    results = []

    if len(logs) >= 2:
        results.append(("Seat Invariance", check_seat_invariance(logs, seed)))
    else:
        print("Skipping seat invariance check (requires 2+ seatings)")

    for seating_name, log in logs.items():
        results.append(
            (
                f"Blind Rotation ({seating_name})",
                check_blind_rotation(log, seating_name),
            )
        )
        results.append(
            (
                f"No Resurrection ({seating_name})",
                check_no_resurrection(log, seating_name),
            )
        )
        results.append(
            (f"Zero-Sum ({seating_name})", check_zero_sum(log, seating_name))
        )
        results.append(
            (
                f"Chip Conservation ({seating_name})",
                check_chip_conservation(log, seating_name),
            )
        )

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for check_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {check_name}")

    print(f"\n{passed}/{total} checks passed")

    if passed == total:
        print("\nAll validation checks passed!")
        return 0
    else:
        print(f"\n{total - passed} check(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
