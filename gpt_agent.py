"""Poker agent framework using vals model proxy."""

import json
import logging
import re
from typing import List, Optional, Tuple, Dict

from dotenv import load_dotenv
from model_library.base import LLM, TextInput
from model_library.exceptions import retry_llm_call
from model_library.registry_utils import get_registry_model

load_dotenv()
logger = logging.getLogger(__name__)

MAX_API_RETRY_TIME = 5 * 60


class PokerAgent:
    """Poker agent that maintains reasoning history and game context."""

    def __init__(
        self,
        seat: int,
        player_name: str,
        model_str: str = "openai/gpt-4o",
        model_instance: Optional[LLM] = None,
    ):
        self.seat = seat
        self.player_name = player_name
        self.model_str = model_str

        if model_instance is None:
            self.model = get_registry_model(model_str)
        else:
            self.model = model_instance

        def create_custom_retrier(logger):
            return retry_llm_call(
                logger,
                max_time=MAX_API_RETRY_TIME,
                max_tries=10,
            )

        self.model.custom_retrier = create_custom_retrier

        self.reasoning_history: List[dict] = []
        self.conversation_history: Dict[int, List] = {}
        self.last_seen_action: Dict[int, int] = {}
        self.has_acted_in_hand: Dict[int, bool] = {}

    async def get_action(
        self,
        observation: dict,
        game_context: List[dict],
        return_response: bool = True,
    ) -> Tuple[dict, dict]:
        current_hand = observation.get("hand_num", 0)
        last_seen_action_num = self.last_seen_action.get(current_hand, -1)
        is_first_action_in_hand = not self.has_acted_in_hand.get(current_hand, False)

        self.has_acted_in_hand[current_hand] = True

        # Providers without conversation history support
        providers_without_history_support = [
            "together",
            "google",
            "mistral",
            "ai21labs",
            "grok",
        ]
        # using conversation histroy as context
        use_conversation_history = (
            self.model.provider not in providers_without_history_support
        )

        # build prompt
        current_prompt = _build_poker_prompt(
            observation,
            game_context,
            last_seen_action_num,
            is_first_action_in_hand,
            use_conversation_history=use_conversation_history,
        )
        logger.info(f"Model prompt for {observation['name']}:\n {current_prompt}\n")

        if use_conversation_history:
            hand_history = self.conversation_history.get(current_hand, [])
        else:
            hand_history = []

        try:
            response = await self.model.query(
                [TextInput(text=current_prompt)],
                system_prompt=_get_system_prompt(observation),
                history=hand_history,
            )

            response_text = _extract_text_from_query_result(response)
            logger.debug(f"Model response for {observation['name']}: {response_text}\n")

            model_response = _extract_model_response(
                response_text, response, self.model
            )
            action = _parse_model_response(response_text, observation)
            logger.info(f"Model action for {observation['name']}: {action}")

            if _is_legal_action(action, observation):
                reasoning_entry = {
                    "hand_num": observation.get("hand_num", 0),
                    "action_num": observation.get("action_num", 0),
                    "reasoning": model_response.get("reasoning", ""),
                    "action": action,
                }
                self.reasoning_history.append(reasoning_entry)

                if use_conversation_history:
                    self._update_conversation_history(
                        current_hand, current_prompt, response_text
                    )
                self.last_seen_action[current_hand] = observation.get("action_num", 0)

                if return_response:
                    return (action, model_response)
                else:
                    return action
            else:
                logger.warning(
                    f"Model returned illegal action {action}. Using fallback for {observation['name']}"
                )
                fallback = _get_fallback_action(observation)
                if return_response:
                    model_response["error"] = "Illegal action returned, using fallback"
                    return (fallback, model_response)
                else:
                    return fallback

        except Exception as e:
            logger.error(f"Error calling model API for {observation['name']}: {e}")
            fallback = _get_fallback_action(observation)
            if return_response:
                return (fallback, {"error": str(e), "fallback_used": True})
            else:
                return fallback

    def _update_conversation_history(
        self, hand_num: int, user_message: str, assistant_response: str
    ):
        if hand_num not in self.conversation_history:
            self.conversation_history[hand_num] = []

        self.conversation_history[hand_num].append(
            self._create_message("user", user_message)
        )

        self.conversation_history[hand_num].append(
            self._create_message("assistant", assistant_response)
        )

    def _create_message(self, role: str, content: str):
        return {"role": role, "content": content}

# helpers to parse responses
def _format_new_actions(
    game_context: List[dict],
    current_hand: int,
    current_action_num: int,
    last_seen_action_num: int,
) -> str:
    if not game_context:
        return ""

    new_actions = [
        entry
        for entry in game_context
        if (
            entry.get("hand_num", 0) == current_hand
            and last_seen_action_num < entry.get("action_num", 0) < current_action_num
        )
    ]

    if not new_actions:
        return ""

    action_lines = []
    for entry in new_actions:
        player = entry.get("player", "Unknown")
        action = entry.get("action", {})
        board = entry.get("board", [])
        action_str = _format_action(action)

        if board:
            action_lines.append(f"{player}: {action_str}, Board: {board}")
        else:
            action_lines.append(f"{player}: {action_str}")

    return "\n".join(action_lines)


def _format_past_hands(game_context: List[dict], current_hand: int) -> str:
    if not game_context:
        return ""

    hands = {}
    for entry in game_context:
        hand_num = entry.get("hand_num", 0)
        if hand_num < current_hand:
            if hand_num not in hands:
                hands[hand_num] = []
            hands[hand_num].append(entry)

    if not hands:
        return ""

    formatted_hands = []
    n_most_recent_hands = 25
    for hand_num in sorted(hands.keys())[-n_most_recent_hands:]:

        hand_actions = hands[hand_num]
        hand_summary = [f"Hand {hand_num}:"]

        current_board = []
        for entry in hand_actions:
            board = entry.get("board", [])
            if board != current_board:
                current_board = board
                if board:
                    hand_summary.append(f"  Board: {board}")

            player = entry.get("player", "Unknown")
            action = entry.get("action", {})
            action_str = _format_action(action)
            hand_summary.append(f"  {player}: {action_str}")

        formatted_hands.append("\n".join(hand_summary))

    return "\n\n".join(formatted_hands)


def _format_action(action: dict) -> str:
    action_type = action.get("type", "unknown")
    if action_type in ("bet", "raise"):
        amount = action.get("amount", 0)
        return f"{action_type.upper()} {amount}"
    return action_type.upper()


def _extract_text_from_query_result(response) -> str:
    try:
        if hasattr(response, "history") and response.history:
            last_message = response.history[-1]
            if hasattr(last_message, "content"):
                for content_block in last_message.content:
                    if hasattr(content_block, "text"):
                        return content_block.text

        return str(response)
    except Exception as e:
        raise ValueError(f"Failed to extract text from QueryResult: {e}")


def _strip_markdown_fences(content: str) -> str:
    cleaned = re.sub(r"^```json\s*", "", content)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _get_system_prompt(observation: dict) -> str:
    small_blind = observation.get("small_blind", 1)
    big_blind = observation.get("big_blind", 2)
    num_players = observation.get("num_players", 3)

    example1_bet = big_blind * 5
    example1_min_raise = example1_bet + (example1_bet - small_blind)

    example2_bet = big_blind * 10
    example2_min_raise = example2_bet + (example2_bet - small_blind)

    return f"""You are an expert poker player playing No Limit Texas Hold'em. 

Your goal is to make poker decisions that maximize your winnings. You should consider:
- Your hole cards and their strength
- The community board cards
- Your position at the table
- Pot odds and implied odds
- Stack sizes and betting patterns
- The number of players in the hand ({num_players} players total)
- The past hands and your actions in those hands

You must respond with a valid JSON object containing your action. The format must be exactly one of:
- {{"action": "fold", "reasoning": "..."}}
- {{"action": "check", "reasoning": "..."}} (when no bet to call)
- {{"action": "call", "reasoning": "..."}} (when there's a bet to call)
- {{"action": "bet", "amount": X, "reasoning": "..."}} (when you can bet)
- {{"action": "raise", "amount": X, "reasoning": "..."}} (when you can raise)

IMPORTANT BETTING RULES:
- Small blind: {small_blind} chips, Big blind: {big_blind} chips
- Minimum bet: {big_blind} chips (big blind amount)
- Minimum raise: Must raise by at least the size of the previous raise
- Formula: min_raise = max_bet + (max_bet - min_bet)
- Example: If bets are [{small_blind}, {big_blind}, {example1_bet}], min raise is {example1_bet} + ({example1_bet}-{small_blind}) = {example1_min_raise}
- Example: If bets are [{small_blind}, {big_blind}, {example2_bet}], min raise is {example2_bet} + ({example2_bet}-{small_blind}) = {example2_min_raise}
- Your raise amount is the TOTAL amount you want to bet, not the additional amount

The amount must be a positive integer. Make sure your action is legal given the available actions."""


def _build_poker_prompt(
    observation: dict,
    game_context: List[dict] = None,
    last_seen_action_num: int = -1,
    is_first_action_in_hand: bool = False,
    use_conversation_history: bool = True,
) -> str:
    """builds prompt for poker situation.
    
    if first action: add past hands + game state
    
    follow-up actions: new actions since last message + game state
    """
    current_hand = observation.get("hand_num", 0)
    action_num = observation.get("action_num", 0)

    prompt_parts = []

    if is_first_action_in_hand:
        prompt_parts.append("=== PREVIOUS HANDS ===")
        if game_context:
            past_hands = _format_past_hands(game_context, current_hand)
            if past_hands:
                prompt_parts.append(past_hands)
            else:
                prompt_parts.append("This is your first hand.")
        else:
            prompt_parts.append("This is your first hand.")
        prompt_parts.append("")

        if game_context:
            current_hand_actions = _format_new_actions(
                game_context, current_hand, action_num, -1
            )
            if current_hand_actions:
                prompt_parts.append("=== CURRENT HAND ===")
                prompt_parts.append(
                    "It is now your turn. The current game state is as follows. The following actions have been played:"
                )
                prompt_parts.append(current_hand_actions)
                prompt_parts.append("")

    elif not is_first_action_in_hand and game_context:
        if use_conversation_history:
            prompt_parts.append("The following actions have occurred:")
            new_actions = _format_new_actions(
                game_context, current_hand, action_num, last_seen_action_num
            )
            if new_actions:
                prompt_parts.append(new_actions)
            prompt_parts.append("")
        else:
            prompt_parts.append("=== CURRENT HAND ===")
            prompt_parts.append(
                "It is now your turn. The current game state is as follows. The following actions have been played:"
            )
            all_hand_actions = _format_new_actions(
                game_context,
                current_hand,
                action_num,
                -1,
            )
            if all_hand_actions:
                prompt_parts.append(all_hand_actions)
            prompt_parts.append("")

    if "=== CURRENT HAND ===" not in prompt_parts:
        prompt_parts.append("=== CURRENT HAND ===")

    prompt_parts.extend(
        [
            f"Player: {observation['name']} (Seat {observation['seat']}, Position {observation['position']})",
            f"Number of players: {observation.get('num_players', 'Unknown')}",
            f"Blinds: {observation.get('small_blind', 'Unknown')}/{observation.get('big_blind', 'Unknown')}",
            f"Hole cards: {observation['hole_cards']}",
            f"Board: {observation['board']}",
            f"Pot size: {observation['pot']}",
            f"Amount to call: {observation['to_call']}",
            f"Minimum raise amount: {observation.get('min_raise', 'N/A')}",
            f"Your stack: {observation['stacks'][observation['name']]}",
            f"All stacks: {observation['stacks']}",
            f"Available actions: {observation['legal_actions']}",
            f"Can check: {observation['is_check']}",
            "",
            "What is your action? Respond with a JSON object containing your decision. Seperately, have a `reasoning` key in your json object,",
            " which contains your thought process behind your decision.",
        ]
    )

    return "\n".join(prompt_parts)


def _extract_model_response(response_text: str, query_result, model: LLM) -> dict:
    """Parse model response and extract metadata for logging."""
    try:
        content = response_text.strip()
        cleaned_content = _strip_markdown_fences(content)
        metadata = query_result.metadata

        response_dict = {
            "model_name": model.model_name,
            "provider": model.provider,
            "latency_seconds": metadata.duration_seconds,
            "in_tokens": metadata.in_tokens,
            "out_tokens": metadata.out_tokens,
        }

        if metadata.reasoning_tokens is not None:
            response_dict["reasoning_tokens"] = metadata.reasoning_tokens
        # parse reasononing
        try:
            parsed = json.loads(cleaned_content)
            response_dict["reasoning"] = parsed.get("reasoning", "")
        except json.JSONDecodeError:
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', cleaned_content)
            if reasoning_match:
                response_dict["reasoning"] = reasoning_match.group(1)
            else:
                response_dict["reasoning"] = ""

        return response_dict
    except Exception as e:
        try:
            metadata = query_result.metadata
            return {
                "error": f"Failed to extract model response: {e}",
                "model_name": getattr(model, "model_name", "unknown"),
                "provider": getattr(model, "provider", "unknown"),
                "latency_seconds": getattr(metadata, "duration_seconds", 0),
                "in_tokens": getattr(metadata, "in_tokens", 0),
                "out_tokens": getattr(metadata, "out_tokens", 0),
                "reasoning_tokens": getattr(metadata, "reasoning_tokens", 0),
            }
        except:
            return {"error": f"Failed to extract model response: {e}"}


def _parse_model_response(response: str, observation: dict) -> dict:
    """Parse model response and extract poker action."""
    try:
        content = response.strip()
        content = _strip_markdown_fences(content)

        parsed = _robust_json_parse(content)

        action_type = parsed.get("action", "").lower()

        if action_type == "fold":
            return {"type": "fold"}
        elif action_type == "check":
            return {"type": "check"}
        elif action_type == "call":
            return {"type": "call"}
        elif action_type in ("bet", "raise"):
            amount = parsed.get("amount")
            if amount is None or not isinstance(amount, (int, float)) or amount <= 0:
                raise ValueError(f"Invalid amount for {action_type}: {amount}")
            return {"type": action_type, "amount": int(amount)}
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    except Exception as e:
        logger.error(f"Failed to parse model response: {e}")
        logger.error(f"Raw response that failed: {response[:500]}")
        raise ValueError(f"Failed to parse model response: {e}")


def _robust_json_parse(content: str) -> dict:
    """Robustly parse JSON with multiple fallback strategies."""

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    try:
        fixed_content = _fix_json_escapes(content)
        return json.loads(fixed_content)
    except json.JSONDecodeError:
        pass

    try:
        start_idx = content.find("{")
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            for i in range(start_idx, len(content)):
                if content[i] == "{":
                    brace_count += 1
                elif content[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break

            if brace_count == 0:
                json_text = content[start_idx : end_idx + 1]
                fixed_json = _fix_json_escapes(json_text)
                return json.loads(fixed_json)
    except json.JSONDecodeError:
        pass

    try:
        return _extract_action_manually(content)
    except ValueError:
        pass

    raise ValueError("No valid JSON found in response after all parsing strategies")


def _fix_json_escapes(content: str) -> str:
    fixes = [
        (r'\\(?!["\\/bfnrt])', r"\\\\"),
        (r"\\n", r"\\n"),
        (r"\\t", r"\\t"),
    ]

    fixed = content
    for pattern, replacement in fixes:
        fixed = re.sub(pattern, replacement, fixed)

    return fixed


def _extract_action_manually(content: str) -> dict:
    content_lower = content.lower()

    if '"action"' in content_lower or "'action'" in content_lower:
        action_patterns = [
            r'"action"\s*:\s*"(fold|check|call|bet|raise)"',
            r"'action'\s*:\s*'(fold|check|call|bet|raise)'",
        ]

        action_type = None
        for pattern in action_patterns:
            match = re.search(pattern, content_lower)
            if match:
                action_type = match.group(1)
                break

        if action_type:
            result = {"action": action_type}

            if action_type in ("bet", "raise"):
                amount_patterns = [
                    r'"amount"\s*:\s*(\d+)',
                    r"'amount'\s*:\s*(\d+)",
                ]

                for pattern in amount_patterns:
                    match = re.search(pattern, content)
                    if match:
                        result["amount"] = int(match.group(1))
                        break

                if "amount" not in result:
                    raise ValueError(f"Could not extract amount for {action_type}")

            return result

    raise ValueError("Could not extract action information")


def _is_legal_action(action: dict, observation: dict) -> bool:
    action_type = action["type"]
    legal_actions = observation["legal_actions"]

    if action_type not in legal_actions:
        logger.warning(f"Action '{action_type}' not in legal actions: {legal_actions}")
        return False

    if action_type in ("bet", "raise") and "amount" in action:
        amount = action["amount"]
        player_stack = observation["stacks"][observation["name"]]

        if amount <= 0:
            logger.warning(f"Amount {amount} must be positive")
            return False
        if amount > player_stack:
            logger.warning(f"Amount {amount} exceeds stack {player_stack}")
            return False

        if action_type == "raise":
            min_raise = observation.get("min_raise")
            if min_raise and amount < min_raise:
                logger.warning(f"Raise amount {amount} below minimum {min_raise}")
                return False

    return True


def _get_fallback_action(observation: dict) -> dict:
    legal_actions = observation["legal_actions"]
    logger.warning(f"Using fallback action for {observation['name']}")

    if "check" in legal_actions:
        return {"type": "check"}
    elif "call" in legal_actions:
        return {"type": "call"}
    elif "fold" in legal_actions:
        return {"type": "fold"}
    else:
        return {"type": "fold"}
