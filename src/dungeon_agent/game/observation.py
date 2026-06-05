"""Deterministic prose observation rendering for world state."""

from __future__ import annotations

from dungeon_agent.game.domain import Direction, WorldState

_DIRECTION_ORDER: tuple[Direction, ...] = (
    Direction.NORTH,
    Direction.SOUTH,
    Direction.EAST,
    Direction.WEST,
)


def render_room_observation(world: WorldState) -> str:
    room = world.rooms[world.player.current_room_id]
    if world.is_dark:
        return room.dark_description

    parts = [f"{room.name}. {room.description}".strip()]
    if room.exits:
        parts.append(f"Exits lead {', '.join(_ordered_exit_names(room.exits)).lower()}.")
    if world.exit_room_id == room.room_id:
        parts.append("This room is the dungeon exit.")

    visible_items = world.visible_items_in_room(room.room_id)
    if visible_items:
        item_text = "; ".join(world.items[item_id].short_description for item_id in visible_items)
        parts.append(f"You notice {item_text}.")
    return " ".join(parts)


def render_inventory_observation(world: WorldState) -> str:
    if not world.player.inventory:
        return "You are carrying nothing."
    ordered_names = sorted((world.items[item_id].name for item_id in world.player.inventory), key=str.lower)
    carrying = f"You are carrying: {', '.join(ordered_names)}."
    loot_total = sum(world.items[item_id].loot_value for item_id in world.player.inventory)
    loot_text = f" Total loot value: {loot_total}."
    if not world.player.worn:
        return f"{carrying}{loot_text}"
    worn_names = sorted((world.items[item_id].name for item_id in world.player.worn), key=str.lower)
    return f"{carrying} You are wearing: {', '.join(worn_names)}.{loot_text}"


def _ordered_exit_names(exits: dict[Direction, str]) -> list[str]:
    prioritized = [direction.value for direction in _DIRECTION_ORDER if direction in exits]
    remaining = sorted(
        direction.value for direction in exits if direction not in _DIRECTION_ORDER  # pragma: no cover
    )
    return prioritized + remaining
