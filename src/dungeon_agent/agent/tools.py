"""Deterministic tool-layer components for command robustness and planning state."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from dungeon_agent.schemas import ActiveSubgoal, StateFlags, ValidatorAction

if TYPE_CHECKING:
    from dungeon_agent.agent.policy import PolicyInput, PolicyOutput

ALLOWED_DIRECTIONS = ("N", "S", "E", "W", "NORTH", "SOUTH", "EAST", "WEST")
BASE_COMMANDS = (
    "N",
    "S",
    "E",
    "W",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "GO",
    "L",
    "LOOK",
    "MOVES",
    "TURNS",
    "HELP",
    "COMMANDS",
    "QUIT",
    "EXIT",
    "Q",
    "I",
    "INVENTORY",
    "SEARCH",
    "X",
    "EXAMINE",
    "GET",
    "TAKE",
    "WEAR",
    "DON",
    "USE",
    "OPEN",
    "MOVE",
)
SHORT_DIRECTIONS = {"NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W"}
REVERSE_DIRECTIONS = {"N": "S", "S": "N", "E": "W", "W": "E"}
MULTI_COMMAND_SEPARATORS = (";", "&&", "||", "\n")


@dataclass(frozen=True, slots=True)
class ToolContext:
    policy_input: "PolicyInput"
    policy_output: "PolicyOutput"
    state_flags: StateFlags
    room_id: str | None = None


class MapMemoryStore(Protocol):
    def observe(self, event: "MapObservation") -> None:
        """Persist deterministic map facts from an observation event."""


class GoalTracker(Protocol):
    def update(self, state_flags: StateFlags) -> ActiveSubgoal:
        """Advance active subgoal deterministically from state flags."""


class CommandValidationTool(Protocol):
    def validate(self, proposed_command: str) -> "ValidationResult":
        """Validate and normalize a single proposed command."""


class LoopRecoveryTool(Protocol):
    def record_turn(self, *, command: str, room_id: str) -> "RecoveryDecision":
        """Track turn history and trigger deterministic recovery decisions."""


@dataclass(frozen=True, slots=True)
class MapObservation:
    room_id: str
    exits: tuple[str, ...] = ()
    items: tuple[str, ...] = ()


def _normalize_token(token: str) -> str:
    return " ".join(token.strip().upper().split())


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def _correct_token_unambiguous(token: str, candidates: tuple[str, ...], *, max_distance: int = 2) -> str | None:
    normalized = _normalize_token(token)
    if normalized in candidates:
        return normalized

    prefix_matches = tuple(candidate for candidate in candidates if candidate.startswith(normalized))
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        return None

    distances = [(candidate, _edit_distance(normalized, candidate)) for candidate in candidates]
    min_distance = min(distance for _, distance in distances)
    if min_distance > max_distance:
        return None
    nearest = tuple(candidate for candidate, distance in distances if distance == min_distance)
    if len(nearest) == 1:
        return nearest[0]
    return None


def _canonicalize_direction(direction: str) -> str:
    normalized = _normalize_token(direction)
    return SHORT_DIRECTIONS.get(normalized, normalized)


@dataclass(slots=True)
class MapMemory:
    discovered_rooms: set[str] = field(default_factory=set)
    discovered_exits: dict[str, set[str]] = field(default_factory=dict)
    discovered_items: dict[str, set[str]] = field(default_factory=dict)
    graph: dict[str, dict[str, str]] = field(default_factory=dict)

    def observe(self, event: MapObservation) -> None:
        room_id = event.room_id.strip()
        if not room_id:
            return

        self.discovered_rooms.add(room_id)
        room_exits = self.discovered_exits.setdefault(room_id, set())
        room_items = self.discovered_items.setdefault(room_id, set())
        self.graph.setdefault(room_id, {})
        room_exits.update(_canonicalize_direction(direction) for direction in event.exits if direction.strip())
        room_items.update(item.strip() for item in event.items if item.strip())

    def add_edge(self, room_id: str, direction: str, destination_room_id: str) -> None:
        start = room_id.strip()
        dest = destination_room_id.strip()
        if not start or not dest:
            return
        canonical_direction = _canonicalize_direction(direction)
        if canonical_direction not in {"N", "S", "E", "W"}:
            return
        self.discovered_rooms.update({start, dest})
        self.discovered_exits.setdefault(start, set()).add(canonical_direction)
        self.discovered_exits.setdefault(dest, set()).add(REVERSE_DIRECTIONS[canonical_direction])
        self.graph.setdefault(start, {})[canonical_direction] = dest
        self.graph.setdefault(dest, {})[REVERSE_DIRECTIONS[canonical_direction]] = start

    def path_to(self, start_room_id: str, target_room_id: str) -> tuple[str, ...]:
        if start_room_id == target_room_id:
            return tuple()
        queue: deque[tuple[str, tuple[str, ...]]] = deque([(start_room_id, tuple())])
        visited = {start_room_id}
        while queue:
            room_id, path = queue.popleft()
            for direction, next_room in self.graph.get(room_id, {}).items():
                if next_room in visited:
                    continue
                next_path = (*path, direction)
                if next_room == target_room_id:
                    return next_path
                visited.add(next_room)
                queue.append((next_room, next_path))
        return tuple()

    def unexplored_exits(self, room_id: str) -> tuple[str, ...]:
        observed = self.discovered_exits.get(room_id, set())
        traversed = set(self.graph.get(room_id, {}).keys())
        remaining = tuple(sorted(direction for direction in observed if direction not in traversed))
        return remaining

    def path_to_nearest_frontier(self, start_room_id: str) -> tuple[str, ...]:
        if self.unexplored_exits(start_room_id):
            return tuple()
        queue: deque[tuple[str, tuple[str, ...]]] = deque([(start_room_id, tuple())])
        visited = {start_room_id}
        while queue:
            room_id, path = queue.popleft()
            for direction in sorted(self.graph.get(room_id, {})):
                next_room = self.graph[room_id][direction]
                if next_room in visited:
                    continue
                next_path = (*path, direction)
                if self.unexplored_exits(next_room):
                    return next_path
                visited.add(next_room)
                queue.append((next_room, next_path))
        return tuple()

    def get_exits(self, room_id: str) -> frozenset[str]:
        return frozenset(self.discovered_exits.get(room_id, set()))

    def get_items(self, room_id: str) -> frozenset[str]:
        return frozenset(self.discovered_items.get(room_id, set()))

    def has_room(self, room_id: str) -> bool:
        return room_id in self.discovered_rooms


@dataclass(slots=True)
class GoalManager:
    active_subgoal: ActiveSubgoal = ActiveSubgoal.FIND_LIGHT

    def update(self, state_flags: StateFlags, *, needs_key: bool = False) -> ActiveSubgoal:
        if needs_key and not state_flags.has_treasure and self.active_subgoal is not ActiveSubgoal.RETURN_TO_EXIT:
            self.active_subgoal = ActiveSubgoal.FIND_KEY
        elif self.active_subgoal is ActiveSubgoal.FIND_KEY:
            self.active_subgoal = ActiveSubgoal.FIND_TREASURE
        if self.active_subgoal is ActiveSubgoal.FIND_LIGHT and state_flags.has_light:
            self.active_subgoal = ActiveSubgoal.FIND_TREASURE
        if self.active_subgoal in {ActiveSubgoal.FIND_TREASURE, ActiveSubgoal.FIND_KEY} and state_flags.has_treasure:
            self.active_subgoal = ActiveSubgoal.RETURN_TO_EXIT
        return self.active_subgoal


@dataclass(slots=True)
class CommandHistory:
    events: deque[tuple[str, str]] = field(default_factory=lambda: deque(maxlen=200))

    def record(self, *, command: str, room_id: str) -> None:
        self.events.append((_normalize_token(command), room_id.strip()))

    def last(self, n: int) -> tuple[tuple[str, str], ...]:
        if n <= 0:
            return tuple()
        return tuple(list(self.events)[-n:])


@dataclass(slots=True)
class WorldMemory:
    seen_inventory: set[str] = field(default_factory=set)
    seen_items_by_room: dict[str, set[str]] = field(default_factory=dict)
    locked_targets: set[str] = field(default_factory=set)
    clue_notes: list[str] = field(default_factory=list)
    searched_targets_by_room: dict[str, set[str]] = field(default_factory=dict)

    def observe_room_items(self, room_id: str, item_descriptors: tuple[str, ...]) -> None:
        room_key = room_id.strip()
        if not room_key:
            return
        bucket = self.seen_items_by_room.setdefault(room_key, set())
        bucket.update(descriptor.strip() for descriptor in item_descriptors if descriptor.strip())

    def observe_inventory(self, item_names: tuple[str, ...]) -> None:
        self.seen_inventory.update(item.strip() for item in item_names if item.strip())

    def note_locked_target(self, target: str) -> None:
        cleaned = target.strip()
        if cleaned:
            self.locked_targets.add(cleaned)

    def add_clue(self, clue_text: str) -> None:
        clue = clue_text.strip()
        if clue and clue not in self.clue_notes:
            self.clue_notes.append(clue)

    def note_search(self, *, room_id: str, target: str) -> None:
        room_key = room_id.strip()
        target_key = " ".join(target.strip().upper().split())
        if not room_key or not target_key:
            return
        bucket = self.searched_targets_by_room.setdefault(room_key, set())
        bucket.add(target_key)

    def searched_targets_for_room(self, room_id: str | None) -> tuple[str, ...]:
        if room_id is None:
            return tuple()
        room_key = room_id.strip()
        if not room_key:
            return tuple()
        return tuple(sorted(self.searched_targets_by_room.get(room_key, set())))


@dataclass(frozen=True, slots=True)
class ValidationResult:
    validated_command: str
    action: ValidatorAction


class CommandValidator:
    def _parse_and_normalize(self, proposed_command: str) -> ValidationResult:
        candidate = _normalize_token(proposed_command)
        if not candidate or any(separator in proposed_command for separator in MULTI_COMMAND_SEPARATORS):
            return ValidationResult(validated_command="", action=ValidatorAction.REPLAN)

        tokens = candidate.split()
        corrected_head = _correct_token_unambiguous(tokens[0], BASE_COMMANDS)
        if corrected_head is None:
            return ValidationResult(validated_command="", action=ValidatorAction.REPLAN)
        rewritten = corrected_head != tokens[0]
        tokens[0] = corrected_head

        if tokens[0] in {"N", "S", "E", "W", "NORTH", "SOUTH", "EAST", "WEST"}:
            if len(tokens) != 1:
                return ValidationResult(validated_command="", action=ValidatorAction.REPLAN)
            command = _canonicalize_direction(tokens[0])
            return ValidationResult(
                validated_command=command, action=ValidatorAction.REWRITTEN if rewritten else ValidatorAction.ACCEPTED
            )

        if tokens[0] == "GO":
            if len(tokens) != 2:
                return ValidationResult(validated_command="", action=ValidatorAction.REPLAN)
            corrected_direction = _correct_token_unambiguous(tokens[1], ALLOWED_DIRECTIONS)
            if corrected_direction is None:
                return ValidationResult(validated_command="", action=ValidatorAction.REPLAN)
            command = _canonicalize_direction(corrected_direction)
            rewritten = rewritten or corrected_direction != tokens[1]
            return ValidationResult(
                validated_command=command, action=ValidatorAction.REWRITTEN if rewritten else ValidatorAction.ACCEPTED
            )

        if tokens[0] in {"L", "LOOK", "I", "INVENTORY", "MOVES", "TURNS", "HELP", "COMMANDS", "QUIT", "EXIT", "Q"}:
            if len(tokens) != 1:
                return ValidationResult(validated_command="", action=ValidatorAction.REPLAN)
            if tokens[0] in {"L", "LOOK"}:
                canonical = "LOOK"
            elif tokens[0] in {"I", "INVENTORY"}:
                canonical = "INVENTORY"
            elif tokens[0] in {"MOVES", "TURNS"}:
                canonical = "MOVES"
            elif tokens[0] in {"QUIT", "EXIT", "Q"}:
                canonical = "QUIT"
            else:
                canonical = "HELP"
            return ValidationResult(
                validated_command=canonical, action=ValidatorAction.REWRITTEN if rewritten else ValidatorAction.ACCEPTED
            )

        if tokens[0] == "SEARCH":
            if len(tokens) == 1:
                return ValidationResult(validated_command="SEARCH", action=ValidatorAction.REWRITTEN if rewritten else ValidatorAction.ACCEPTED)
            return ValidationResult(
                validated_command=f"SEARCH {' '.join(tokens[1:])}",
                action=ValidatorAction.REWRITTEN if rewritten else ValidatorAction.ACCEPTED,
            )

        if tokens[0] in {"X", "EXAMINE", "GET", "TAKE", "OPEN", "MOVE", "WEAR", "DON"}:
            if len(tokens) < 2:
                return ValidationResult(validated_command="", action=ValidatorAction.REPLAN)
            canonical_verb = {
                "X": "EXAMINE",
                "EXAMINE": "EXAMINE",
                "GET": "TAKE",
                "TAKE": "TAKE",
                "OPEN": "OPEN",
                "MOVE": "MOVE",
                "WEAR": "WEAR",
                "DON": "WEAR",
            }[tokens[0]]
            rewritten = rewritten or canonical_verb != tokens[0]
            return ValidationResult(
                validated_command=f"{canonical_verb} {' '.join(tokens[1:])}",
                action=ValidatorAction.REWRITTEN if rewritten else ValidatorAction.ACCEPTED,
            )

        if tokens[0] == "USE":
            if len(tokens) < 2:
                return ValidationResult(validated_command="", action=ValidatorAction.REPLAN)
            if "ON" in tokens[1:]:
                on_index = tokens.index("ON")
                if on_index == 1 or on_index == len(tokens) - 1:
                    return ValidationResult(validated_command="", action=ValidatorAction.REPLAN)
                return ValidationResult(
                    validated_command=f"USE {' '.join(tokens[1:])}",
                    action=ValidatorAction.REWRITTEN if rewritten else ValidatorAction.ACCEPTED,
                )
            return ValidationResult(
                validated_command=f"USE {' '.join(tokens[1:])}",
                action=ValidatorAction.REWRITTEN if rewritten else ValidatorAction.ACCEPTED,
            )

        return ValidationResult(validated_command="", action=ValidatorAction.REPLAN)

    def validate(self, proposed_command: str) -> ValidationResult:
        return self._parse_and_normalize(proposed_command)


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    triggered: bool
    reason: Literal["none", "repeated_command", "room_oscillation"] = "none"


class LoopRecovery:
    def __init__(self) -> None:
        self._commands: deque[str] = deque(maxlen=3)
        self._rooms: deque[str] = deque(maxlen=3)
        self._consecutive_loop_detections = 0
        self._warning_pending = False

    def record_turn(self, *, command: str, room_id: str) -> RecoveryDecision:
        self._commands.append(_normalize_token(command))
        self._rooms.append(room_id)

        if len(self._commands) == 3 and len(set(self._commands)) == 1:
            self._consecutive_loop_detections += 1
            if self._consecutive_loop_detections == 1:
                self._warning_pending = True
            return RecoveryDecision(triggered=True, reason="repeated_command")

        if len(self._rooms) == 3 and self._rooms[0] == self._rooms[2] and self._rooms[0] != self._rooms[1]:
            self._consecutive_loop_detections += 1
            if self._consecutive_loop_detections == 1:
                self._warning_pending = True
            return RecoveryDecision(triggered=True, reason="room_oscillation")

        self._consecutive_loop_detections = 0
        self._warning_pending = False
        return RecoveryDecision(triggered=False, reason="none")

    def should_recover(self) -> bool:
        return self._consecutive_loop_detections >= 2

    def should_warn_about_loop(self) -> bool:
        return self._warning_pending

    def consume_loop_warning(self) -> None:
        self._warning_pending = False

    def loop_warning_message(self) -> str:
        return (
            "Loop warning: recent moves appear to be repeating without progress. "
            "Choose a different action now; if repetition continues, a forced recovery move will be used."
        )

    def choose_recovery_direction(self, exits: tuple[str, ...]) -> str | None:
        candidates = tuple(
            direction
            for direction in (_canonicalize_direction(exit_name) for exit_name in exits)
            if direction in REVERSE_DIRECTIONS
        )
        if not candidates:
            return None

        disallowed: set[str] = set()
        if self._commands:
            last_command = self._commands[-1]
            if last_command in REVERSE_DIRECTIONS:
                disallowed.add(REVERSE_DIRECTIONS[last_command])
        if len(self._commands) >= 2:
            previous_command = self._commands[-2]
            last_command = self._commands[-1]
            if (
                previous_command in REVERSE_DIRECTIONS
                and last_command in REVERSE_DIRECTIONS
                and REVERSE_DIRECTIONS[previous_command] == last_command
            ):
                disallowed.update({previous_command, last_command})

        for direction in candidates:
            if direction not in disallowed:
                return direction
        return candidates[0]
COMMAND_REFERENCE_TEXT = (
    "Commands: N|S|E|W (or GO <direction>), LOOK (L), INVENTORY (I), "
    "EXAMINE <target> (X), SEARCH [ROOM|<target>], TAKE <target> (GET), WEAR <target> (DON), "
    "OPEN <target>, MOVE <target>, USE <item> [ON <target>], MOVES (TURNS), HELP (COMMANDS), "
    "QUIT (confirmation required)."
)


class CommandReference:
    @staticmethod
    def text() -> str:
        return COMMAND_REFERENCE_TEXT
