"""Poker table manager with button rotation and stack tracking across multiple hands."""

from pokerkit import NoLimitTexasHoldem, Automation, Rank, Suit, Card
import random
import json
import logging
from typing import List, Tuple, Optional, Callable, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class PokerTable:
    def __init__(
        self,
        player_names: List[str],
        starting_stacks: List[int],
        small_blind: int = 1,
        big_blind: int = 2,
        seed: Optional[int] = None,
    ):

        assert len(player_names) == len(starting_stacks), "Names and stacks must match"
        assert len(player_names) >= 2, "Need at least 2 players"

        self.num_players = len(player_names)
        self.player_names = player_names
        self.stacks = list(starting_stacks)
        self.small_blind = small_blind
        self.big_blind = big_blind

        self.button_seat = self.num_players - 1
        self.hand_count = 0

        if seed is not None:
            random.seed(seed)
            self.seed = seed
        else:
            self.seed = None

        self.game_log = {
            "hands": [],
            "session_info": {
                "player_names": player_names,
                "starting_stacks": list(starting_stacks),
                "small_blind": small_blind,
                "big_blind": big_blind,
                "seed": seed,
                "start_time": datetime.now().isoformat(),
            },
        }
        self.current_hand_log = None

        self.game_context: List[dict] = []
        self.current_action_num = 0

    def position_to_seat(self, position: int) -> int:
        if self.num_players == 2:
            return (self.button_seat + position) % 2
        return (self.button_seat + 1 + position) % self.num_players

    def seat_to_position(self, seat: int) -> int:
        if self.num_players == 2:
            return (seat - self.button_seat) % 2
        return (seat - self.button_seat - 1) % self.num_players

    def _get_position_stacks(self) -> Tuple[int, ...]:
        return tuple(
            self.stacks[self.position_to_seat(pos)] for pos in range(self.num_players)
        )

    def _get_active_players_info(self) -> Tuple[List[int], List[str], List[int]]:
        active_seats = []
        active_names = []
        active_stacks = []

        for seat in range(self.num_players):
            if self.stacks[seat] > 0:
                active_seats.append(seat)
                active_names.append(self.player_names[seat])
                active_stacks.append(self.stacks[seat])

        return active_seats, active_names, active_stacks

    def play_hand(self, agent_action_fn: Callable) -> dict:
        self.hand_count += 1

        if self.seed is not None:
            hand_seed = self.seed + self.hand_count
            random.seed(hand_seed)
            logger.debug(f"Hand {self.hand_count}: Seed: {hand_seed}")

        active_seats, active_names, active_stacks = self._get_active_players_info()
        num_active_players = len(active_seats)

        original_starting_stacks = list(self.stacks)

        if num_active_players < 2:
            logger.warning("Not enough active players to play a hand")
            return {
                "hand_number": self.hand_count,
                "stacks": list(self.stacks),
                "profits": [0] * self.num_players,
                "board": [],
            }

        if self.button_seat not in active_seats:
            self.button_seat = active_seats[0]


        if num_active_players == 2:
            active_position_to_seat = [self.button_seat]
            sb_seat = active_seats[(active_seats.index(self.button_seat) + 1) % 2]
            active_position_to_seat.append(sb_seat)
        else:
            button_idx = active_seats.index(self.button_seat)
            active_position_to_seat = [
                active_seats[(button_idx + 1 + i) % num_active_players]
                for i in range(num_active_players)
            ]

        all_stacks = [self.stacks[seat] for seat in active_position_to_seat]

        state = NoLimitTexasHoldem.create_state(
            (
                Automation.ANTE_POSTING,
                Automation.BLIND_OR_STRADDLE_POSTING,
                Automation.BET_COLLECTION,
                Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                Automation.HAND_KILLING,
                Automation.CHIPS_PUSHING,
                Automation.CHIPS_PULLING,
            ),
            False,
            {-1: 0},
            (self.small_blind, self.big_blind),
            self.big_blind,
            all_stacks,
            num_active_players,
        )

        pre_shuffled_deck = None
        deck_index = self.num_players * 2

        if self.seed is not None:
            hand_seed = self.seed + self.hand_count
            rng = random.Random(hand_seed)
            pre_shuffled_deck = [
                Card(r, s)
                for r in Rank
                if r != Rank.UNKNOWN
                for s in Suit
                if s != Suit.UNKNOWN
            ]
            rng.shuffle(pre_shuffled_deck)

            for position in range(num_active_players):
                seat = active_position_to_seat[position]
                cards = [pre_shuffled_deck[seat * 2], pre_shuffled_deck[seat * 2 + 1]]
                state.deal_hole(cards, player_index=position)
        else:
            for position in range(num_active_players):
                state.deal_hole(player_index=position)

        starting_stacks_by_seat = [0] * self.num_players
        for position in range(num_active_players):
            seat = active_position_to_seat[position]
            starting_stacks_by_seat[seat] = state.stacks[position]

        self._init_hand_log(state, starting_stacks_by_seat, active_position_to_seat)
        self._print_hand_header()
        self.current_action_num = 0
        timestep = 0

        while state.status:
            if state.can_burn_card():
                if pre_shuffled_deck:
                    state.burn_card(pre_shuffled_deck[deck_index])
                    deck_index += 1
                else:
                    state.burn_card()

            if state.can_deal_hole():
                if pre_shuffled_deck:
                    state.deal_hole(pre_shuffled_deck[deck_index])
                    deck_index += 1
                else:
                    state.deal_hole()

            if state.can_deal_board():
                board_count = state.board_dealing_count
                if pre_shuffled_deck and board_count:
                    cards = pre_shuffled_deck[deck_index : deck_index + board_count]
                    state.deal_board(cards)
                    deck_index += board_count
                else:
                    state.deal_board()

            if state.actor_index is not None:
                position = state.actor_index
                seat = active_position_to_seat[position]

                if (
                    state.can_check_or_call()
                    or state.can_fold()
                    or state.can_complete_bet_or_raise_to()
                ):
                    observation = self._build_observation(
                        state, position, seat, active_position_to_seat
                    )
                    stack_before = state.stacks[position]
                    action_result = agent_action_fn(
                        seat, observation, self.game_context
                    )

                    if isinstance(action_result, tuple):
                        action, model_response = action_result
                    else:
                        action = action_result
                        model_response = None

                    self._execute_action(state, action)

                    stack_after = state.stacks[position]
                    chip_change = stack_after - stack_before
                    self._log_action(
                        timestep,
                        seat,
                        action,
                        chip_change,
                        stack_after,
                        list(state.board_cards),
                        model_response,
                    )

                    context_entry = {
                        "hand_num": self.hand_count,
                        "action_num": self.current_action_num,
                        "player": self.player_names[seat],
                        "action": action,
                        "board": list(state.board_cards),
                    }
                    self.game_context.append(context_entry)

                    self.current_action_num += 1
                    timestep += 1
                else:
                    break
            elif not (
                state.can_burn_card() or state.can_deal_hole() or state.can_deal_board()
            ):
                break

        for position in range(num_active_players):
            seat = active_position_to_seat[position]
            self.stacks[seat] = state.stacks[position]

        profits_by_seat = []
        for seat in range(self.num_players):
            profit = self.stacks[seat] - original_starting_stacks[seat]
            profits_by_seat.append(profit)

        self._print_hand_results(
            state, original_starting_stacks, profits_by_seat, active_position_to_seat
        )

        self._finalize_hand_log(state, active_position_to_seat)

        active_seats, _, _ = self._get_active_players_info()
        if len(active_seats) > 1:
            if self.button_seat not in active_seats:
                self.button_seat = active_seats[0]
            else:
                current_button_idx = active_seats.index(self.button_seat)
                next_button_idx = (current_button_idx + 1) % len(active_seats)
                self.button_seat = active_seats[next_button_idx]

        return {
            "hand_number": self.hand_count,
            "stacks": list(self.stacks),
            "profits": [
                self.stacks[seat] - original_starting_stacks[seat]
                for seat in range(self.num_players)
            ],
            "board": state.board_cards,
        }

    def _build_observation(
        self, state, position: int, seat: int, active_position_to_seat: list
    ) -> dict:
        max_bet = max(state.bets)
        current_bet = state.bets[position]
        is_check = current_bet == max_bet

        active_seats, _, _ = self._get_active_players_info()
        stacks_dict = {
            self.player_names[s]: self.stacks[s] for s in range(self.num_players)
        }

        return {
            "seat": seat,
            "position": position,
            "name": self.player_names[seat],
            "hole_cards": state.hole_cards[position],
            "board": state.board_cards,
            "pot": state.total_pot_amount,
            "stacks": stacks_dict,
            "current_bet": current_bet,
            "to_call": max_bet - current_bet,
            "min_bet_amount": state.min_completion_betting_or_raising_to_amount,
            "max_bet_amount": state.max_completion_betting_or_raising_to_amount,
            "legal_actions": self._get_legal_actions(state, is_check),
            "is_check": is_check,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "num_players": len(active_seats),
            "hand_num": self.hand_count,
            "action_num": self.current_action_num,
        }

    def _get_legal_actions(self, state, is_check: bool) -> List[str]:
        actions = []
        if state.can_fold():
            actions.append("fold")
        if state.can_check_or_call():
            actions.append("check" if is_check else "call")
        if state.can_complete_bet_or_raise_to():
            actions.append("bet" if is_check else "raise")
        return actions

    def _execute_action(self, state, action: dict):
        action_type = action["type"]

        if action_type == "fold":
            state.fold()
        elif action_type in ("check", "call"):
            state.check_or_call()
        elif action_type in ("bet", "raise"):
            amount = action["amount"]
            min_allowed = state.min_completion_betting_or_raising_to_amount
            max_allowed = state.max_completion_betting_or_raising_to_amount

            if min_allowed is not None and max_allowed is not None:
                if amount < min_allowed:
                    amount = min_allowed
                elif amount > max_allowed:
                    amount = max_allowed

            state.complete_bet_or_raise_to(amount)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    def _print_hand_header(self):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"HAND {self.hand_count}")
        logger.info(f"{'=' * 60}")

        active_seats, _, _ = self._get_active_players_info()
        num_active = len(active_seats)

        if num_active == 2:
            button_index = active_seats.index(self.button_seat)
            sb_seat = self.button_seat
            bb_seat = active_seats[(button_index + 1) % 2]
            logger.info(
                f"Button/SB: {self.player_names[sb_seat]}, BB: {self.player_names[bb_seat]}"
            )
        else:
            button_index = active_seats.index(self.button_seat)
            sb_seat = active_seats[(button_index + 1) % num_active]
            bb_seat = active_seats[(button_index + 2) % num_active]
            logger.info(
                f"Button: {self.player_names[self.button_seat]}, "
                f"SB: {self.player_names[sb_seat]}, "
                f"BB: {self.player_names[bb_seat]}"
            )

        stacks_str = ", ".join(
            f"{self.player_names[seat]}: {self.stacks[seat]}" for seat in active_seats
        )
        logger.info(f"Stacks: {stacks_str}")
        logger.info("")

    def _print_hand_results(
        self, state, starting_stacks, profits, active_position_to_seat
    ):
        logger.info("\n=== Hand finished ===")
        logger.info(f"Board: {state.board_cards}")

        for position, seat in enumerate(active_position_to_seat):
            logger.info(f"{self.player_names[seat]}: {state.hole_cards[position]}")

        logger.info("")
        for seat in range(self.num_players):
            profit = profits[seat]
            if profit > 0:
                logger.info(f"  {self.player_names[seat]} +{profit}")
            elif profit < 0:
                logger.info(f"  {self.player_names[seat]} {profit}")

    def can_continue(self) -> bool:
        min_chips_needed = self.small_blind + self.big_blind
        return sum(1 for stack in self.stacks if stack >= min_chips_needed) >= 2

    def get_session_summary(self, initial_stacks: List[int]) -> dict:
        return {
            "hands_played": self.hand_count,
            "final_stacks": dict(zip(self.player_names, self.stacks)),
            "total_profit_loss": {
                name: self.stacks[i] - initial_stacks[i]
                for i, name in enumerate(self.player_names)
            },
        }

    def _init_hand_log(
        self, state, starting_stacks: List[int], active_position_to_seat: list
    ):
        seat_to_position_map = {
            seat: pos for pos, seat in enumerate(active_position_to_seat)
        }

        starting_state = {}
        for seat, stack in enumerate(starting_stacks):
            player_name = self.player_names[seat]
            position = seat_to_position_map.get(seat, -1)

            starting_state[player_name] = {
                "hole_cards": (
                    [str(card) for card in state.hole_cards[position]]
                    if position >= 0
                    else []
                ),
                "position": position,
                "seat": seat,
                "starting_stack": stack,
            }

        self.current_hand_log = {
            "hand_number": self.hand_count,
            "button_seat": self.button_seat,
            "starting_state": starting_state,
            "actions": [],
            "ending_stacks": {},
            "final_board": [],
        }

    def _log_action(
        self,
        timestep: int,
        seat: int,
        action: dict,
        chip_change: int,
        stack_after: int,
        board: List[str],
        model_response: Optional[Any] = None,
    ):
        player_name = self.player_names[seat]

        action_log = {
            "timestep": timestep,
            "player": player_name,
            "seat": seat,
            "action": action.copy(),
            "chip_change": chip_change,
            "stack_after": stack_after,
            "board": [str(card) for card in board],
        }

        if model_response is not None:
            action_log["model_response"] = model_response

        self.current_hand_log["actions"].append(action_log)

    def _finalize_hand_log(self, state, active_position_to_seat):
        for seat in range(self.num_players):
            player_name = self.player_names[seat]
            self.current_hand_log["ending_stacks"][player_name] = self.stacks[seat]

        self.current_hand_log["final_board"] = [str(card) for card in state.board_cards]
        self.game_log["hands"].append(self.current_hand_log)
        self.current_hand_log = None

    def save_game_log(self, filename: str = "game_log.json"):
        self.game_log["session_info"]["end_time"] = datetime.now().isoformat()
        self.game_log["session_info"]["total_hands"] = self.hand_count

        with open(filename, "w") as f:
            json.dump(self.game_log, f, indent=2)

    def get_game_log(self) -> dict:
        return self.game_log
