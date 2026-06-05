"""Seeded procedural dungeon layout generation with solvability checks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from random import Random

from dungeon_agent.content.templates import ROOM_TEMPLATES, RoomTemplate
from dungeon_agent.game.domain import Direction, Room

DEFAULT_ROOM_COUNT = 15
LOCK_PLACEMENT_PROBABILITY = 0.35
_MAX_ATTEMPTS = 128
_ENTRY_TEMPLATE_ID = "entry_vestibule"
_LIGHT_TEMPLATE_ID = "lantern_workshop"
_TREASURE_TEMPLATE_ID = "sunken_treasure_vault"
_KEY_TEMPLATE_ID = "ossuary_walk"
_KEY_ITEM_ID = "vault_key"

_DIRECTION_STEPS: dict[Direction, tuple[int, int]] = {
    Direction.NORTH: (0, 1),
    Direction.SOUTH: (0, -1),
    Direction.EAST: (1, 0),
    Direction.WEST: (-1, 0),
}
_OPPOSITE: dict[Direction, Direction] = {
    Direction.NORTH: Direction.SOUTH,
    Direction.SOUTH: Direction.NORTH,
    Direction.EAST: Direction.WEST,
    Direction.WEST: Direction.EAST,
}


@dataclass(frozen=True, slots=True)
class LockedConnection:
    room_a_id: str
    room_b_id: str
    key_room_id: str
    key_item_id: str = _KEY_ITEM_ID

    @property
    def edge_key(self) -> frozenset[str]:
        return frozenset((self.room_a_id, self.room_b_id))


@dataclass(frozen=True, slots=True)
class DungeonLayout:
    seed: int
    start_room_id: str
    exit_room_id: str
    rooms: dict[str, Room]
    light_room_id: str
    treasure_room_id: str
    locked_connection: LockedConnection | None = None

    @property
    def room_count(self) -> int:
        return len(self.rooms)

    @property
    def key_room_id(self) -> str | None:
        return self.locked_connection.key_room_id if self.locked_connection else None

    def neighbors(self, room_id: str) -> frozenset[str]:
        return frozenset(self.rooms[room_id].exits.values())

    def edge_count(self) -> int:
        return sum(len(room.exits) for room in self.rooms.values()) // 2

    def degree_map(self) -> dict[str, int]:
        return {room_id: len(room.exits) for room_id, room in self.rooms.items()}


def generate_dungeon_layout(*, seed: int, room_count: int = DEFAULT_ROOM_COUNT) -> DungeonLayout:
    """Generate a deterministic, validated dungeon layout for the provided seed."""
    if room_count != DEFAULT_ROOM_COUNT:
        raise ValueError(f"room_count must be {DEFAULT_ROOM_COUNT} for the v1 authored template set")
    if room_count > len(ROOM_TEMPLATES):
        raise ValueError("room_count exceeds authored room template count")

    for attempt in range(_MAX_ATTEMPTS):
        candidate_seed = (seed * 31_337) + attempt
        rng = Random(candidate_seed)
        candidate = _build_candidate_layout(seed=seed, room_count=room_count, rng=rng)
        if validate_dungeon_layout(candidate):
            return candidate
    raise RuntimeError("failed to generate a solvable dungeon layout within attempt limit")


def validate_dungeon_layout(layout: DungeonLayout) -> bool:
    """Validate graph constraints and required progression path solvability."""
    if layout.room_count != DEFAULT_ROOM_COUNT:
        return False
    if layout.start_room_id not in layout.rooms or layout.exit_room_id not in layout.rooms:
        return False
    if layout.light_room_id not in layout.rooms or layout.treasure_room_id not in layout.rooms:
        return False
    if layout.light_room_id == layout.treasure_room_id:
        return False

    if not _is_bidirectional_and_connected(layout):
        return False

    degree_map = layout.degree_map()
    if not any(degree >= 3 for degree in degree_map.values()):
        return False
    if sum(1 for degree in degree_map.values() if degree == 1) < 2:
        return False

    if layout.locked_connection and not _is_valid_locked_connection(layout):
        return False

    return _has_progression_path(layout)


def _build_candidate_layout(*, seed: int, room_count: int, rng: Random) -> DungeonLayout:
    indexed_exits = _generate_indexed_tree(room_count=room_count, rng=rng)
    path_from_start, light_index, treasure_index = _select_progression_indexes(indexed_exits=indexed_exits, rng=rng)

    locked_edge: tuple[int, int] | None = None
    key_index: int | None = None
    if rng.random() < LOCK_PLACEMENT_PROBABILITY:
        locked_edge, key_index = _choose_optional_lock_and_key(
            indexed_exits=indexed_exits,
            path_from_start=path_from_start,
            light_index=light_index,
            treasure_index=treasure_index,
            rng=rng,
        )

    index_to_template = _assign_templates(
        room_count=room_count,
        light_index=light_index,
        treasure_index=treasure_index,
        key_index=key_index,
        rng=rng,
    )
    index_to_room_id = {index: template.room_id for index, template in index_to_template.items()}

    rooms: dict[str, Room] = {}
    for index, template in index_to_template.items():
        room = template.to_room()
        room.exits = {
            direction: index_to_room_id[destination] for direction, destination in indexed_exits[index].items()
        }
        rooms[room.room_id] = room

    locked_connection = None
    if locked_edge is not None and key_index is not None:
        room_a = index_to_room_id[locked_edge[0]]
        room_b = index_to_room_id[locked_edge[1]]
        locked_connection = LockedConnection(
            room_a_id=room_a,
            room_b_id=room_b,
            key_room_id=index_to_room_id[key_index],
        )

    return DungeonLayout(
        seed=seed,
        start_room_id=index_to_room_id[0],
        exit_room_id=index_to_room_id[0],
        rooms=rooms,
        light_room_id=index_to_room_id[light_index],
        treasure_room_id=index_to_room_id[treasure_index],
        locked_connection=locked_connection,
    )


def _generate_indexed_tree(*, room_count: int, rng: Random) -> dict[int, dict[Direction, int]]:
    indexed_exits: dict[int, dict[Direction, int]] = {0: {}}
    coordinates: dict[int, tuple[int, int]] = {0: (0, 0)}
    occupied: dict[tuple[int, int], int] = {(0, 0): 0}
    next_index = 1

    opening_directions = list(Direction)
    rng.shuffle(opening_directions)
    for direction in opening_directions[: min(3, room_count - 1)]:
        _attach_room(
            parent_index=0,
            child_index=next_index,
            direction=direction,
            indexed_exits=indexed_exits,
            coordinates=coordinates,
            occupied=occupied,
        )
        next_index += 1

    while next_index < room_count:
        placements = _candidate_placements(indexed_exits=indexed_exits, coordinates=coordinates, occupied=occupied)
        if not placements:
            raise ValueError("no placements available")
        weights = [max(1, 4 - len(indexed_exits[parent_index])) for parent_index, _ in placements]
        selected_parent, selected_direction = rng.choices(placements, weights=weights, k=1)[0]
        _attach_room(
            parent_index=selected_parent,
            child_index=next_index,
            direction=selected_direction,
            indexed_exits=indexed_exits,
            coordinates=coordinates,
            occupied=occupied,
        )
        next_index += 1

    return indexed_exits


def _candidate_placements(
    *,
    indexed_exits: dict[int, dict[Direction, int]],
    coordinates: dict[int, tuple[int, int]],
    occupied: dict[tuple[int, int], int],
) -> list[tuple[int, Direction]]:
    placements: list[tuple[int, Direction]] = []
    for parent_index, exits in indexed_exits.items():
        parent_x, parent_y = coordinates[parent_index]
        for direction, (dx, dy) in _DIRECTION_STEPS.items():
            if direction in exits:
                continue
            candidate_coord = (parent_x + dx, parent_y + dy)
            if candidate_coord in occupied:
                continue
            placements.append((parent_index, direction))
    return placements


def _attach_room(
    *,
    parent_index: int,
    child_index: int,
    direction: Direction,
    indexed_exits: dict[int, dict[Direction, int]],
    coordinates: dict[int, tuple[int, int]],
    occupied: dict[tuple[int, int], int],
) -> None:
    parent_x, parent_y = coordinates[parent_index]
    dx, dy = _DIRECTION_STEPS[direction]
    child_coord = (parent_x + dx, parent_y + dy)
    if child_coord in occupied:
        raise ValueError("coordinate collision")

    indexed_exits.setdefault(parent_index, {})
    indexed_exits.setdefault(child_index, {})
    indexed_exits[parent_index][direction] = child_index
    indexed_exits[child_index][_OPPOSITE[direction]] = parent_index
    coordinates[child_index] = child_coord
    occupied[child_coord] = child_index


def _select_progression_indexes(
    *,
    indexed_exits: dict[int, dict[Direction, int]],
    rng: Random,
) -> tuple[list[int], int, int]:
    distances = _distances_from_root(indexed_exits=indexed_exits)
    farthest_distance = max(distances.values())
    treasure_candidates = [
        index for index, distance in distances.items() if index != 0 and distance >= max(4, farthest_distance - 1)
    ]
    if not treasure_candidates:
        treasure_candidates = [index for index, distance in distances.items() if distance == farthest_distance]
    treasure_index = rng.choice(sorted(treasure_candidates))

    path = _shortest_path(indexed_exits=indexed_exits, start=0, goal=treasure_index)
    if len(path) < 3:
        raise ValueError("treasure path is too short to place progression checkpoints")

    light_candidates = path[1:-1]
    cutoff = max(1, len(light_candidates) // 2)
    light_index = rng.choice(light_candidates[:cutoff])
    return path, light_index, treasure_index


def _choose_optional_lock_and_key(
    *,
    indexed_exits: dict[int, dict[Direction, int]],
    path_from_start: list[int],
    light_index: int,
    treasure_index: int,
    rng: Random,
) -> tuple[tuple[int, int] | None, int | None]:
    if len(path_from_start) < 4:
        return None, None
    light_position = path_from_start.index(light_index)
    treasure_position = path_from_start.index(treasure_index)
    lock_positions = list(range(light_position + 1, treasure_position))
    if not lock_positions:
        return None, None

    lock_edge_position = rng.choice(lock_positions)
    locked_edge = (
        path_from_start[lock_edge_position],
        path_from_start[lock_edge_position + 1],
    )
    locked_nodes = set(locked_edge)
    start_component = _reachable_without_locked_edge(indexed_exits=indexed_exits, locked_edge=locked_edge)

    excluded_indexes = {0, light_index, treasure_index, *locked_nodes}
    key_candidates = [index for index in start_component if index not in excluded_indexes]
    if not key_candidates:
        key_candidates = [index for index in start_component if index not in {0, treasure_index, *locked_nodes}]
    if not key_candidates:
        return None, None

    distances = _distances_from_root(indexed_exits=indexed_exits)
    furthest = max(distances[index] for index in key_candidates)
    weighted_candidates = [index for index in key_candidates if distances[index] >= furthest - 1]
    key_index = rng.choice(sorted(weighted_candidates))
    return locked_edge, key_index


def _assign_templates(
    *,
    room_count: int,
    light_index: int,
    treasure_index: int,
    key_index: int | None,
    rng: Random,
) -> dict[int, RoomTemplate]:
    templates_by_id = {template.room_id: template for template in ROOM_TEMPLATES}
    required_template_ids = {_ENTRY_TEMPLATE_ID, _LIGHT_TEMPLATE_ID, _TREASURE_TEMPLATE_ID}
    if key_index is not None:
        required_template_ids.add(_KEY_TEMPLATE_ID)

    if any(template_id not in templates_by_id for template_id in required_template_ids):
        raise ValueError("missing required room templates")

    index_to_template: dict[int, RoomTemplate] = {0: templates_by_id[_ENTRY_TEMPLATE_ID]}
    if light_index == 0:
        raise ValueError("light index cannot be start index")
    index_to_template[light_index] = templates_by_id[_LIGHT_TEMPLATE_ID]
    index_to_template[treasure_index] = templates_by_id[_TREASURE_TEMPLATE_ID]
    if key_index is not None:
        if key_index in {0, light_index, treasure_index}:
            raise ValueError("key index must be distinct from special progression rooms")
        index_to_template[key_index] = templates_by_id[_KEY_TEMPLATE_ID]

    reserved_ids = {template.room_id for template in index_to_template.values()}
    remaining_templates = [template for template in ROOM_TEMPLATES if template.room_id not in reserved_ids]
    rng.shuffle(remaining_templates)

    remaining_indexes = [index for index in range(room_count) if index not in index_to_template]
    if len(remaining_indexes) > len(remaining_templates):
        raise ValueError("insufficient templates to populate room indexes")

    for index, template in zip(remaining_indexes, remaining_templates, strict=True):
        index_to_template[index] = template
    return index_to_template


def _distances_from_root(*, indexed_exits: dict[int, dict[Direction, int]]) -> dict[int, int]:
    distances = {0: 0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for neighbor in indexed_exits[node].values():
            if neighbor in distances:
                continue
            distances[neighbor] = distances[node] + 1
            queue.append(neighbor)
    return distances


def _shortest_path(*, indexed_exits: dict[int, dict[Direction, int]], start: int, goal: int) -> list[int]:
    queue = deque([start])
    previous: dict[int, int | None] = {start: None}
    while queue:
        node = queue.popleft()
        if node == goal:
            break
        for neighbor in indexed_exits[node].values():
            if neighbor in previous:
                continue
            previous[neighbor] = node
            queue.append(neighbor)

    if goal not in previous:
        raise ValueError("goal was unreachable from start")

    path = [goal]
    while previous[path[-1]] is not None:
        path.append(previous[path[-1]])  # type: ignore[arg-type]
    path.reverse()
    return path


def _reachable_without_locked_edge(
    *,
    indexed_exits: dict[int, dict[Direction, int]],
    locked_edge: tuple[int, int],
) -> set[int]:
    blocked = frozenset(locked_edge)
    seen = {0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for neighbor in indexed_exits[node].values():
            if frozenset((node, neighbor)) == blocked or neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(neighbor)
    return seen


def _is_bidirectional_and_connected(layout: DungeonLayout) -> bool:
    for room_id, room in layout.rooms.items():
        for direction, destination in room.exits.items():
            if destination not in layout.rooms:
                return False
            if destination == room_id:
                return False
            opposite_room = layout.rooms[destination]
            opposite_direction = _OPPOSITE[direction]
            if opposite_room.exits.get(opposite_direction) != room_id:
                return False

    seen: set[str] = set()
    queue = deque([layout.start_room_id])
    while queue:
        room_id = queue.popleft()
        if room_id in seen:
            continue
        seen.add(room_id)
        queue.extend(layout.neighbors(room_id))
    return len(seen) == layout.room_count


def _is_valid_locked_connection(layout: DungeonLayout) -> bool:
    locked = layout.locked_connection
    if locked is None:
        return True
    if locked.room_a_id not in layout.rooms or locked.room_b_id not in layout.rooms:
        return False
    if locked.key_room_id not in layout.rooms:
        return False
    if locked.key_room_id in {locked.room_a_id, locked.room_b_id}:
        return False
    return locked.room_b_id in layout.neighbors(locked.room_a_id)


def _has_progression_path(layout: DungeonLayout) -> bool:
    start_has_light = layout.start_room_id == layout.light_room_id
    start_has_treasure = layout.start_room_id == layout.treasure_room_id
    visited: set[tuple[str, bool, bool, bool]] = set()
    queue = deque(
        [
            (
                layout.start_room_id,
                False,
                start_has_light,
                start_has_treasure,
            )
        ]
    )
    while queue:
        room_id, has_key, has_light, has_treasure = queue.popleft()
        updated_has_key = has_key
        updated_has_light = has_light or room_id == layout.light_room_id
        updated_has_treasure = has_treasure or room_id == layout.treasure_room_id
        if layout.locked_connection and room_id == layout.locked_connection.key_room_id:
            updated_has_key = True

        state = (room_id, updated_has_key, updated_has_light, updated_has_treasure)
        if state in visited:
            continue
        visited.add(state)

        if room_id == layout.exit_room_id and updated_has_light and updated_has_treasure:
            return True

        for neighbor in layout.neighbors(room_id):
            if _is_locked_edge(layout, room_a=room_id, room_b=neighbor) and not updated_has_key:
                continue
            queue.append((neighbor, updated_has_key, updated_has_light, updated_has_treasure))
    return False


def _is_locked_edge(layout: DungeonLayout, *, room_a: str, room_b: str) -> bool:
    if layout.locked_connection is None:
        return False
    return frozenset((room_a, room_b)) == layout.locked_connection.edge_key
