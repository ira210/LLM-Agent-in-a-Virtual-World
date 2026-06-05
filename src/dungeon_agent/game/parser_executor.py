"""Command parsing and deterministic execution against world state."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Callable

from dungeon_agent.game.domain import Direction, ItemLocationKind, ItemRelationType, WorldState
from dungeon_agent.game.interfaces import GameEngine, GameState, TurnOutcome
from dungeon_agent.game.observation import render_inventory_observation, render_room_observation

_ARTICLES = {"A", "AN", "THE"}
_DIRECTION_ALIASES = {
    "N": "NORTH",
    "NORTH": "NORTH",
    "S": "SOUTH",
    "SOUTH": "SOUTH",
    "E": "EAST",
    "EAST": "EAST",
    "W": "WEST",
    "WEST": "WEST",
}
_VERB_ALIASES = {
    "COMMANDS": "HELP",
    "TURNS": "MOVES",
    "MOVES": "MOVES",
    "EXIT": "QUIT",
    "Q": "QUIT",
    "QUIT": "QUIT",
    "HELP": "HELP",
    "LOOK": "LOOK",
    "L": "LOOK",
    "INVENTORY": "INVENTORY",
    "I": "INVENTORY",
    "INV": "INVENTORY",
    "EXAMINE": "EXAMINE",
    "X": "EXAMINE",
    "INSPECT": "EXAMINE",
    "SEARCH": "SEARCH",
    "OPEN": "OPEN",
    "MOVE": "MOVE",
    "PUSH": "MOVE",
    "TAKE": "TAKE",
    "GET": "TAKE",
    "WEAR": "WEAR",
    "DON": "WEAR",
    "USE": "USE",
}
_MULTI_COMMAND_SEPARATORS = (";", "&&", "||", "\n")
_DIRECTION_TO_ENUM = {
    "NORTH": Direction.NORTH,
    "SOUTH": Direction.SOUTH,
    "EAST": Direction.EAST,
    "WEST": Direction.WEST,
}


def _tokens_without_articles(text: str) -> set[str]:
    normalized = text.upper().replace("’", "'").replace("`", "'")
    tokens = {token for token in re.findall(r"[A-Z0-9']+", normalized) if token and token not in _ARTICLES}
    return tokens


def _normalize_tokens(command: str) -> list[str]:
    normalized = " ".join(command.upper().replace("’", "'").replace("`", "'").strip().split())
    return [token for token in re.findall(r"[A-Z0-9']+", normalized) if token]


def _clean_phrase(phrase: str) -> str:
    tokens = [token for token in _normalize_tokens(phrase) if token not in _ARTICLES]
    return " ".join(tokens)


def _resolve_unambiguous(token: str, candidates: set[str]) -> str | None:
    if token in candidates:
        return token
    prefix_matches = sorted(candidate for candidate in candidates if candidate.startswith(token))
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    return None


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    verb: str
    args: tuple[str, ...] = ()
    normalized_command: str = ""


@dataclass(frozen=True, slots=True)
class ParseResult:
    parsed: ParsedCommand | None = None
    error: str | None = None


class CommandParser:
    def parse(self, raw_command: str) -> ParseResult:
        if any(separator in raw_command for separator in _MULTI_COMMAND_SEPARATORS):
            return ParseResult(error="Parser error: only one command is allowed per turn.")

        tokens = _normalize_tokens(raw_command)
        if not tokens:
            return ParseResult(error="Parser error: command cannot be empty.")

        direction = _resolve_unambiguous(tokens[0], set(_DIRECTION_ALIASES))
        if direction:
            if len(tokens) != 1:
                return ParseResult(error="Parser error: movement commands only accept one direction token.")
            normalized_direction = _DIRECTION_ALIASES[direction]
            return ParseResult(
                parsed=ParsedCommand(
                    verb="GO",
                    args=(normalized_direction,),
                    normalized_command=f"GO {normalized_direction}",
                )
            )

        if tokens[0] == "GO":
            if len(tokens) != 2:
                return ParseResult(error="Parser error: GO requires exactly one direction.")
            direction = _resolve_unambiguous(tokens[1], set(_DIRECTION_ALIASES))
            if direction is None:
                return ParseResult(error=f"Parser error: unknown direction '{tokens[1]}'.")
            normalized_direction = _DIRECTION_ALIASES[direction]
            return ParseResult(
                parsed=ParsedCommand(
                    verb="GO",
                    args=(normalized_direction,),
                    normalized_command=f"GO {normalized_direction}",
                )
            )

        verb = _resolve_unambiguous(tokens[0], set(_VERB_ALIASES))
        if verb is None:
            return ParseResult(error=f"Parser error: unknown command '{tokens[0]}'.")

        canonical_verb = _VERB_ALIASES[verb]
        if canonical_verb in {"LOOK", "INVENTORY", "HELP", "QUIT", "MOVES"}:
            if len(tokens) != 1:
                return ParseResult(error=f"Parser error: {canonical_verb} does not take arguments.")
            return ParseResult(parsed=ParsedCommand(verb=canonical_verb, normalized_command=canonical_verb))

        if canonical_verb in {"EXAMINE", "SEARCH", "OPEN", "MOVE", "TAKE", "WEAR"}:
            if len(tokens) < 2:
                return ParseResult(error=f"Parser error: {canonical_verb} requires a target.")
            argument = " ".join(tokens[1:])
            return ParseResult(
                parsed=ParsedCommand(
                    verb=canonical_verb,
                    args=(argument,),
                    normalized_command=f"{canonical_verb} {argument}",
                )
            )

        if canonical_verb == "USE":
            if len(tokens) < 2:
                return ParseResult(error="Parser error: USE requires an item name.")
            if "ON" not in tokens[1:]:
                argument = " ".join(tokens[1:])
                return ParseResult(
                    parsed=ParsedCommand(
                        verb="USE",
                        args=(argument,),
                        normalized_command=f"USE {argument}",
                    )
                )
            on_index = tokens.index("ON")
            if on_index == 1 or on_index == len(tokens) - 1:
                return ParseResult(error="Parser error: USE <item> ON <target> requires both sides.")
            item_name = " ".join(tokens[1:on_index])
            target_name = " ".join(tokens[on_index + 1 :])
            return ParseResult(
                parsed=ParsedCommand(
                    verb="USE",
                    args=(item_name, target_name),
                    normalized_command=f"USE {item_name} ON {target_name}",
                )
            )

        return ParseResult(error=f"Parser error: unsupported command '{tokens[0]}'.")


class ParserExecutorEngine(GameEngine):
    def __init__(self, world_factory: Callable[[int], WorldState], *, parser: CommandParser | None = None) -> None:
        self._world_factory = world_factory
        self._parser = parser or CommandParser()
        self._world: WorldState | None = None
        self._quit_confirmation_armed = False
        self._focused_item_id: str | None = None

    @classmethod
    def from_world_template(
        cls,
        world: WorldState,
        *,
        parser: CommandParser | None = None,
    ) -> "ParserExecutorEngine":
        template = deepcopy(world)
        return cls(lambda _seed: deepcopy(template), parser=parser)

    def reset(self, *, seed: int) -> str:
        self._world = self._world_factory(seed)
        self._quit_confirmation_armed = False
        self._focused_item_id = None
        return render_room_observation(self._world)

    def snapshot(self) -> GameState:
        world = self._require_world()
        return GameState(turn_index=world.turn_index, done=world.done)

    def step(self, command: str) -> TurnOutcome:
        world = self._require_world()
        parse_result = self._parser.parse(command)
        if parse_result.error:
            return self._finalize_turn(parse_result.error)
        parsed = parse_result.parsed
        if parsed is None:
            return self._finalize_turn("Parser error: command could not be interpreted.")

        if self._quit_confirmation_armed and parsed.verb != "QUIT":
            self._quit_confirmation_armed = False

        if parsed.verb == "GO":
            return self._handle_movement(parsed.args[0])
        if parsed.verb == "LOOK":
            return self._finalize_turn(render_room_observation(world))
        if parsed.verb == "HELP":
            return self._finalize_turn(self.command_reference(), consumed_turn=False)
        if parsed.verb == "MOVES":
            return self._finalize_turn("Moves remaining are tracked by the runner.", consumed_turn=False)
        if parsed.verb == "QUIT":
            return self._handle_quit()
        if parsed.verb == "INVENTORY":
            return self._finalize_turn(render_inventory_observation(world))
        if parsed.verb == "EXAMINE":
            return self._finalize_turn(self._handle_examine(parsed.args[0]))
        if parsed.verb == "SEARCH":
            return self._finalize_turn(self._handle_search(parsed.args[0]))
        if parsed.verb == "OPEN":
            return self._finalize_turn(self._handle_open(parsed.args[0]))
        if parsed.verb == "MOVE":
            return self._finalize_turn(self._handle_move(parsed.args[0]))
        if parsed.verb == "TAKE":
            return self._finalize_turn(self._handle_take(parsed.args[0]))
        if parsed.verb == "WEAR":
            return self._finalize_turn(self._handle_wear(parsed.args[0]))
        if parsed.verb == "USE":
            return self._finalize_turn(self._handle_use(*parsed.args))
        return self._finalize_turn(f"Parser error: unsupported normalized command '{parsed.normalized_command}'.")

    def _handle_movement(self, direction_name: str) -> TurnOutcome:
        world = self._require_world()
        origin_room_id = world.player.current_room_id
        direction = _DIRECTION_TO_ENUM[direction_name]
        room = world.rooms[origin_room_id]
        destination = room.exits.get(direction)
        if destination is None:
            return self._finalize_turn(f"You cannot go {direction_name.lower()} from here.")
        if destination == origin_room_id:
            return self._finalize_turn("You cannot go that way; the path circles back to this same room.")
        world.player.current_room_id = destination
        return self._finalize_turn(f"You move {direction_name.lower()} to {world.rooms[destination].name}.")

    def _handle_examine(self, target_phrase: str) -> str:
        world = self._require_world()
        target_id = self._resolve_item_id(
            target_phrase,
            candidates=self._visible_room_items() | set(world.player.inventory),
            allow_context_aliases=False,
        )
        if target_id is None:
            if self._is_current_room_feature(target_phrase):
                return f"The {target_phrase.lower().strip()} is part of the scenery."
            return f"You do not see '{target_phrase.lower()}' here."
        self._focused_item_id = target_id
        return self._describe_item(target_id)

    def _handle_search(self, target_phrase: str) -> str:
        world = self._require_world()
        if world.is_dark:
            return "It is too dark to search effectively."

        target_key = _clean_phrase(target_phrase)
        if target_key in {"", "ROOM", "AREA"}:
            hidden_count = len(self._hidden_room_items())
            if hidden_count:
                return (
                    "You search the area carefully. Something is concealed. "
                    "Check likely anchors with SEARCH <target>, then OPEN containers or MOVE fixtures."
                )
            return "You search the area carefully but find nothing new."

        target_id = self._resolve_item_id(
            target_phrase,
            candidates=self._visible_room_items() | set(world.player.inventory),
            allow_context_aliases=False,
        )
        if target_id is None:
            return f"You cannot identify '{target_phrase.lower()}' to search it."
        self._focused_item_id = target_id

        hidden = self._hidden_children_for_anchor(target_id)
        if not hidden:
            return f"You find nothing unusual about the {world.items[target_id].name.lower()}."

        hints: list[str] = []
        if any(relation_type is ItemRelationType.IN for _, relation_type in hidden):
            hints.append(f"Try OPEN {world.items[target_id].name.upper()}.")
        if any(relation_type in {ItemRelationType.UNDER, ItemRelationType.BEHIND} for _, relation_type in hidden):
            hints.append(f"Try MOVE {world.items[target_id].name.upper()}.")
        return "You suspect something is concealed. " + " ".join(hints)

    def _handle_open(self, target_phrase: str) -> str:
        world = self._require_world()
        target_id = self._resolve_item_id(
            target_phrase,
            candidates=self._visible_room_items() | set(world.player.inventory),
            allow_context_aliases=False,
        )
        if target_id is None:
            if self._is_current_room_feature(target_phrase):
                if self._phrase_mentions_any(target_phrase, {"DOOR", "GATE", "LOCK", "LOCKPLATE"}):
                    return f"The {target_phrase.lower().strip()} is fixed in place and will not open this way."
                return f"The {target_phrase.lower().strip()} is part of the room, not something you can open."
            return f"You do not see '{target_phrase.lower()}' to open."
        self._focused_item_id = target_id
        item = world.items[target_id]
        if not item.is_container:
            return f"The {item.name.lower()} cannot be opened."
        if item.is_locked:
            return f"The {item.name.lower()} is locked."
        if item.is_open:
            return f"The {item.name.lower()} is already open."

        world.set_item_open(target_id, is_open=True)
        revealed = self._revealed_children_for_anchor(target_id, relation_filter={ItemRelationType.IN})
        if revealed:
            return f"You open the {item.name.lower()}. You reveal {', '.join(revealed)}."
        return f"You open the {item.name.lower()}."

    def _handle_move(self, target_phrase: str) -> str:
        world = self._require_world()
        target_id = self._resolve_item_id(
            target_phrase,
            candidates=self._visible_room_items(),
            allow_context_aliases=False,
        )
        if target_id is None:
            return f"You do not see '{target_phrase.lower()}' to move."
        self._focused_item_id = target_id
        item = world.items[target_id]
        if item.is_moved:
            return f"You have already moved the {item.name.lower()}."

        world.set_item_moved(target_id, is_moved=True)
        revealed = self._revealed_children_for_anchor(
            target_id, relation_filter={ItemRelationType.UNDER, ItemRelationType.BEHIND}
        )
        if revealed:
            return f"You move the {item.name.lower()}. You reveal {', '.join(revealed)}."
        return f"You move the {item.name.lower()}."

    def _handle_take(self, target_phrase: str) -> str:
        world = self._require_world()
        target_id = self._resolve_item_id(
            target_phrase,
            candidates=self._visible_room_items(),
            allow_context_aliases=True,
        )
        if target_id is None:
            return f"You do not see '{target_phrase.lower()}' to take."
        self._focused_item_id = target_id
        try:
            world.add_item_to_inventory(target_id)
        except ValueError as exc:
            return str(exc)
        self._remove_taken_item_reference_from_room(target_id)
        return f"You take the {world.items[target_id].name.lower()}."

    def _handle_wear(self, target_phrase: str) -> str:
        world = self._require_world()
        target_id = self._resolve_item_id(
            target_phrase,
            candidates=set(world.player.inventory),
            allow_context_aliases=False,
        )
        if target_id is None:
            return f"You need to be carrying '{target_phrase.lower()}' to wear it."
        self._focused_item_id = target_id
        try:
            world.wear_item(target_id)
        except ValueError as exc:
            return str(exc)
        return f"You wear the {world.items[target_id].name.lower()}."

    def _handle_use(self, item_phrase: str, target_phrase: str | None = None) -> str:
        world = self._require_world()
        inventory_item_id = self._resolve_item_id(
            item_phrase,
            candidates=set(world.player.inventory),
            allow_context_aliases=False,
        )
        if inventory_item_id is None:
            return f"You need to hold '{item_phrase.lower()}' before using it."
        self._focused_item_id = inventory_item_id
        held_item = world.items[inventory_item_id]

        if target_phrase is None:
            if held_item.is_light_source:
                return f"You raise the {held_item.name.lower()}, casting useful light."
            if self._phrase_mentions_any(held_item.name, {"KEY"}) and self._is_current_room_feature("DOOR"):
                return f"You try the {held_item.name.lower()} on the nearby door, but it does not fit."
            return f"You use the {held_item.name.lower()}. {self._render_use_flavor(held_item.detail)}"

        target_id = self._resolve_item_id(
            target_phrase,
            candidates=self._visible_room_items() | set(world.player.inventory),
            allow_context_aliases=False,
        )
        if target_id is None:
            if self._is_current_room_feature(target_phrase):
                if self._phrase_mentions_any(held_item.name, {"KEY"}) and self._phrase_mentions_any(
                    target_phrase, {"DOOR", "GATE", "LOCK", "LOCKPLATE"}
                ):
                    return (
                        f"You try the {held_item.name.lower()} on the {target_phrase.lower().strip()}, "
                        "but it does not fit."
                    )
                return f"You cannot use the {held_item.name.lower()} on the {target_phrase.lower().strip()}."
            return f"You cannot find '{target_phrase.lower()}' to use that on."
        self._focused_item_id = target_id
        target_item = world.items[target_id]
        if self._phrase_mentions_any(held_item.name, {"KEY"}) and target_item.is_locked:
            if target_item.key_item_id == inventory_item_id:
                world.unlock_item(target_id, key_item_id=inventory_item_id)
                return f"You unlock the {target_item.name.lower()} with the {held_item.name.lower()}."
            return f"The {held_item.name.lower()} does not fit the {target_item.name.lower()}."
        return (
            f"You use the {held_item.name.lower()} on the {target_item.name.lower()}. "
            f"{self._render_use_flavor(held_item.detail)}"
        )

    def _handle_quit(self) -> TurnOutcome:
        world = self._require_world()
        if not self._quit_confirmation_armed:
            self._quit_confirmation_armed = True
            return self._finalize_turn("Confirm quit by entering QUIT again.")
        world.done = True
        self._quit_confirmation_armed = False
        return self._finalize_turn("You end your run and leave the dungeon behind.")

    def _resolve_item_id(
        self,
        phrase: str,
        *,
        candidates: set[str],
        allow_context_aliases: bool,
    ) -> str | None:
        world = self._require_world()
        key = _clean_phrase(phrase)
        if not key:
            return None

        exact_matches: list[str] = []
        fuzzy_matches: list[tuple[str, int]] = []
        name_token_matches: list[tuple[str, int]] = []
        typo_matches: list[tuple[str, float]] = []
        key_tokens = set(key.split())
        for item_id in sorted(candidates):
            item = world.items[item_id]
            primary_aliases = {
                _clean_phrase(item_id.replace("_", " ")),
                _clean_phrase(item.name),
            }
            aliases = set(primary_aliases)
            if allow_context_aliases:
                aliases |= {_clean_phrase(item.short_description), _clean_phrase(item.detail)}
            if key in aliases:
                exact_matches.append(item_id)
                continue

            name_tokens = set(_clean_phrase(item.name).split())
            if name_tokens and name_tokens.issubset(key_tokens):
                name_token_matches.append((item_id, len(key_tokens) - len(name_tokens)))

            best_delta: int | None = None
            for alias in aliases:
                alias_tokens = set(alias.split())
                if not key_tokens.issubset(alias_tokens):
                    continue
                delta = len(alias_tokens) - len(key_tokens)
                if best_delta is None or delta < best_delta:
                    best_delta = delta
            if best_delta is not None:
                fuzzy_matches.append((item_id, best_delta))
            typo_score = _single_token_typo_score(key_tokens=key_tokens, aliases=aliases)
            if typo_score is not None:
                typo_matches.append((item_id, typo_score))

        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            if self._focused_item_id in exact_matches:
                return self._focused_item_id
            return None
        if fuzzy_matches:
            min_delta = min(delta for _, delta in fuzzy_matches)
            narrowed = sorted(item_id for item_id, delta in fuzzy_matches if delta == min_delta)
            if len(narrowed) == 1:
                return narrowed[0]
            if self._focused_item_id in narrowed:
                return self._focused_item_id
            return None
        if name_token_matches:
            min_delta = min(delta for _, delta in name_token_matches)
            narrowed = sorted(item_id for item_id, delta in name_token_matches if delta == min_delta)
            if len(narrowed) == 1:
                return narrowed[0]
            if self._focused_item_id in narrowed:
                return self._focused_item_id
            return None
        if typo_matches:
            best_score = max(score for _, score in typo_matches)
            narrowed = sorted(item_id for item_id, score in typo_matches if score == best_score)
            if len(narrowed) == 1:
                return narrowed[0]
            if self._focused_item_id in narrowed:
                return self._focused_item_id
            return None
        if (
            self._focused_item_id in candidates
            and self._focused_item_id is not None
            and self._matches_focus_part_reference(key=key, item_id=self._focused_item_id)
        ):
            return self._focused_item_id
        return None

    def _matches_focus_part_reference(self, *, key: str, item_id: str) -> bool:
        world = self._require_world()
        item = world.items[item_id]
        key_tokens = set(key.split())
        if not key_tokens:
            return False
        focus_text_tokens = set(
            _clean_phrase(f"{item.name} {item.short_description} {item.detail}".strip()).split()
        )
        return key_tokens.issubset(focus_text_tokens)

    @staticmethod
    def command_reference() -> str:
        return (
            "Available commands:\n"
            "- N|S|E|W (or GO <direction>)\n"
            "- LOOK (L)\n"
            "- INVENTORY (I)\n"
            "- EXAMINE <target> (X)\n"
            "- SEARCH [ROOM|<target>]\n"
            "- TAKE <target> (GET)\n"
            "- WEAR <target> (DON)\n"
            "- OPEN <target>\n"
            "- MOVE <target>\n"
            "- USE <item> [ON <target>]\n"
            "- MOVES (or TURNS)\n"
            "- HELP (or COMMANDS)\n"
            "- QUIT (confirmation required: enter QUIT twice)\n"
            "Notes: one command per turn; room SEARCH requires light for useful results."
        )

    def _visible_room_items(self) -> set[str]:
        world = self._require_world()
        return set(world.visible_items_in_room(world.player.current_room_id))

    def _hidden_room_items(self) -> set[str]:
        world = self._require_world()
        room_id = world.player.current_room_id
        hidden = set()
        for item_id in world.items:
            if self._root_room_for_item(item_id) == room_id and not world.is_item_revealed(item_id):
                hidden.add(item_id)
        return hidden

    def _hidden_children_for_anchor(self, anchor_item_id: str) -> list[tuple[str, ItemRelationType]]:
        world = self._require_world()
        hidden: list[tuple[str, ItemRelationType]] = []
        for item_id, location in world.item_locations.items():
            if location.kind is not ItemLocationKind.RELATION or location.relation is None:
                continue
            relation = location.relation
            if relation.anchor_item_id == anchor_item_id and not world.is_item_revealed(item_id):
                hidden.append((item_id, relation.relation_type))
        return hidden

    def _revealed_children_for_anchor(
        self,
        anchor_item_id: str,
        *,
        relation_filter: set[ItemRelationType],
    ) -> list[str]:
        world = self._require_world()
        revealed = []
        for item_id, location in world.item_locations.items():
            if location.kind is not ItemLocationKind.RELATION or location.relation is None:
                continue
            relation = location.relation
            if relation.anchor_item_id != anchor_item_id or relation.relation_type not in relation_filter:
                continue
            if world.is_item_revealed(item_id):
                revealed.append(world.items[item_id].name.lower())
        return sorted(revealed)

    def _root_room_for_item(self, item_id: str) -> str | None:
        world = self._require_world()
        location = world.item_locations[item_id]
        if location.kind is ItemLocationKind.ROOM:
            return location.room_id
        if location.kind is ItemLocationKind.INVENTORY:
            return None
        if location.relation is None:
            return None
        return self._root_room_for_item(location.relation.anchor_item_id)

    def _finalize_turn(self, result_text: str, *, consumed_turn: bool = True) -> TurnOutcome:
        world = self._require_world()
        if consumed_turn:
            world.turn_index += 1
        return TurnOutcome(
            observation_text=render_room_observation(world),
            result_text=result_text,
            done=world.done,
            consumed_turn=consumed_turn,
        )

    def _remove_taken_item_reference_from_room(self, item_id: str) -> None:
        world = self._require_world()
        item = world.items[item_id]
        snippet = item.short_description.strip()
        if not snippet:
            return
        room = world.rooms[world.player.current_room_id]
        updated = re.sub(re.escape(snippet), "", room.description, flags=re.IGNORECASE)
        updated = re.sub(r"\s+", " ", updated)
        updated = re.sub(r"\s+([,.;:])", r"\1", updated)
        room.description = updated.strip(" ,.;:")

    def _describe_item(self, item_id: str) -> str:
        world = self._require_world()
        item = world.items[item_id]
        details = [item.detail]
        if item.is_container:
            if item.is_locked:
                details.append("It is locked.")
            elif item.is_open:
                details.append("It is open.")
            else:
                details.append("It is closed.")
        if item.is_moved:
            details.append("It has been shifted from its original position.")
        if item_id in world.player.worn:
            details.append("You are wearing it.")
        return " ".join(details)

    def _render_use_flavor(self, detail: str) -> str:
        first_sentence = detail.strip().split(".")[0].strip()
        if first_sentence:
            return f"{first_sentence}. It might be valuable, even if not immediately useful."
        return "It might be valuable, even if not immediately useful."

    def _is_current_room_feature(self, target_phrase: str) -> bool:
        world = self._require_world()
        room = world.rooms[world.player.current_room_id]
        target_tokens = _tokens_without_articles(target_phrase)
        if not target_tokens:
            return False
        room_tokens = _tokens_without_articles(
            f"{room.name} {room.description} {getattr(room, 'search_description', '')}"
        )
        return target_tokens.issubset(room_tokens)

    def _phrase_mentions_any(self, phrase: str, tokens: set[str]) -> bool:
        phrase_tokens = _tokens_without_articles(phrase)
        return bool(phrase_tokens & tokens)

    def _require_world(self) -> WorldState:
        if self._world is None:
            raise RuntimeError("engine must be reset before stepping")
        return self._world


def _single_token_typo_score(*, key_tokens: set[str], aliases: set[str]) -> float | None:
    if len(key_tokens) != 1:
        return None
    query = next(iter(key_tokens))
    best = 0.0
    for alias in aliases:
        for alias_token in alias.split():
            score = SequenceMatcher(None, query, alias_token).ratio()
            if score > best:
                best = score
    if best >= 0.74:
        return best
    return None
