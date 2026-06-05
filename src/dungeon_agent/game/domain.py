"""Core game-domain state models and relation-aware item placement."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Direction(str, Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


@dataclass(frozen=True, slots=True)
class Exit:
    direction: Direction
    destination_room_id: str


class ItemRelationType(str, Enum):
    IN = "in"
    UNDER = "under"
    BEHIND = "behind"


class ItemLocationKind(str, Enum):
    ROOM = "room"
    INVENTORY = "inventory"
    RELATION = "relation"


@dataclass(frozen=True, slots=True)
class ItemRelation:
    anchor_item_id: str
    relation_type: ItemRelationType


@dataclass(frozen=True, slots=True)
class ItemLocation:
    kind: ItemLocationKind
    room_id: str | None = None
    relation: ItemRelation | None = None


@dataclass(slots=True)
class Room:
    room_id: str
    name: str
    description: str
    dark_description: str = (
        "You are in darkness. Stone walls press close. "
        "You can feel open space nearby, but details are impossible to make out."
    )
    exits: dict[Direction, str] = field(default_factory=dict)
    has_ambient_light: bool = True


@dataclass(slots=True)
class Item:
    item_id: str
    name: str
    short_description: str
    detail: str
    portable: bool = True
    is_container: bool = False
    is_open: bool = False
    is_light_source: bool = False
    is_moved: bool = False
    is_wearable: bool = False
    is_locked: bool = False
    key_item_id: str | None = None
    loot_value: int = 0


@dataclass(slots=True)
class PlayerState:
    current_room_id: str
    inventory: set[str] = field(default_factory=set)
    worn: set[str] = field(default_factory=set)


@dataclass(slots=True)
class WorldState:
    rooms: dict[str, Room]
    items: dict[str, Item]
    player: PlayerState
    item_locations: dict[str, ItemLocation] = field(default_factory=dict)
    exit_room_id: str | None = None
    objective_treasure_item_id: str | None = None
    turn_index: int = 0
    done: bool = False

    def __post_init__(self) -> None:
        if self.player.current_room_id not in self.rooms:
            raise ValueError("player current room must exist")
        if self.exit_room_id is not None and self.exit_room_id not in self.rooms:
            raise ValueError("exit room must exist")

    @property
    def player_has_light(self) -> bool:
        return any(self.items[item_id].is_light_source for item_id in self.player.inventory)

    @property
    def is_dark(self) -> bool:
        current_room = self.rooms[self.player.current_room_id]
        return not current_room.has_ambient_light and not self.player_has_light

    def set_item_open(self, item_id: str, *, is_open: bool = True) -> None:
        item = self.items[item_id]
        if not item.is_container:
            raise ValueError(f"item '{item_id}' is not a container")
        if is_open and item.is_locked:
            raise ValueError(f"item '{item_id}' is locked")
        item.is_open = is_open

    def unlock_item(self, item_id: str, *, key_item_id: str) -> None:
        item = self._require_item(item_id)
        if not item.is_locked:
            raise ValueError(f"item '{item_id}' is not locked")
        if not item.key_item_id:
            raise ValueError(f"item '{item_id}' has no lock key configured")
        if item.key_item_id != key_item_id:
            raise ValueError(f"key '{key_item_id}' does not unlock '{item_id}'")
        item.is_locked = False

    def set_item_moved(self, item_id: str, *, is_moved: bool = True) -> None:
        self.items[item_id].is_moved = is_moved

    def place_item_in_room(self, item_id: str, room_id: str) -> None:
        self._require_item(item_id)
        self._require_room(room_id)
        self._detach_from_inventory(item_id)
        self.item_locations[item_id] = ItemLocation(kind=ItemLocationKind.ROOM, room_id=room_id)

    def place_item_in_inventory(self, item_id: str) -> None:
        self._require_item(item_id)
        self._detach_from_inventory(item_id)
        self.item_locations[item_id] = ItemLocation(kind=ItemLocationKind.INVENTORY)
        self.player.inventory.add(item_id)

    def wear_item(self, item_id: str) -> None:
        item = self._require_item(item_id)
        if item_id not in self.player.inventory:
            raise ValueError(f"item '{item_id}' is not in inventory")
        if not item.is_wearable:
            raise ValueError(f"item '{item_id}' cannot be worn")
        self.player.worn.add(item_id)

    def place_item_with_relation(
        self,
        item_id: str,
        *,
        anchor_item_id: str,
        relation_type: ItemRelationType,
    ) -> None:
        self._require_item(item_id)
        self._require_item(anchor_item_id)
        if item_id == anchor_item_id:
            raise ValueError("item cannot be related to itself")
        self._assert_no_relation_cycle(item_id=item_id, anchor_item_id=anchor_item_id)
        self._detach_from_inventory(item_id)
        self.item_locations[item_id] = ItemLocation(
            kind=ItemLocationKind.RELATION,
            relation=ItemRelation(anchor_item_id=anchor_item_id, relation_type=relation_type),
        )

    def add_item_to_inventory(self, item_id: str) -> None:
        item = self._require_item(item_id)
        if not item.portable:
            raise ValueError(f"item '{item_id}' is not portable")
        if item_id in self.player.inventory:
            return
        if self._root_room_for_item(item_id) != self.player.current_room_id:
            raise ValueError(f"item '{item_id}' is not in the current room")
        if not self.is_item_revealed(item_id):
            raise ValueError(f"item '{item_id}' is not revealed yet")
        self.place_item_in_inventory(item_id)

    def remove_item_from_inventory(self, item_id: str, *, room_id: str | None = None) -> None:
        if item_id not in self.player.inventory:
            raise ValueError(f"item '{item_id}' is not in inventory")
        target_room_id = room_id or self.player.current_room_id
        self.place_item_in_room(item_id, target_room_id)

    def is_item_revealed(self, item_id: str) -> bool:
        self._require_item(item_id)
        location = self._require_location(item_id)
        if location.kind != ItemLocationKind.RELATION:
            return True
        relation = location.relation
        if relation is None:
            return False
        if not self.is_item_revealed(relation.anchor_item_id):
            return False
        anchor_item = self.items[relation.anchor_item_id]
        if relation.relation_type == ItemRelationType.IN:
            return anchor_item.is_open
        return anchor_item.is_moved

    def visible_items_in_room(self, room_id: str) -> list[str]:
        self._require_room(room_id)
        visible = [
            item_id
            for item_id in self.item_locations
            if self._root_room_for_item(item_id) == room_id and self.is_item_revealed(item_id)
        ]
        return sorted(
            visible,
            key=lambda item_id: (self._relation_depth(item_id), self.items[item_id].name.lower()),
        )

    def _root_room_for_item(self, item_id: str) -> str | None:
        location = self._require_location(item_id)
        if location.kind == ItemLocationKind.ROOM:
            return location.room_id
        if location.kind == ItemLocationKind.INVENTORY:
            return None
        relation = location.relation
        if relation is None:
            return None
        return self._root_room_for_item(relation.anchor_item_id)

    def _relation_depth(self, item_id: str) -> int:
        depth = 0
        location = self._require_location(item_id)
        while location.kind == ItemLocationKind.RELATION and location.relation is not None:
            depth += 1
            location = self._require_location(location.relation.anchor_item_id)
        return depth

    def _assert_no_relation_cycle(self, *, item_id: str, anchor_item_id: str) -> None:
        current = anchor_item_id
        while True:
            if current == item_id:
                raise ValueError("relation placement would create a cycle")
            location = self.item_locations.get(current)
            if location is None or location.kind != ItemLocationKind.RELATION or location.relation is None:
                return
            current = location.relation.anchor_item_id

    def _detach_from_inventory(self, item_id: str) -> None:
        self.player.inventory.discard(item_id)
        self.player.worn.discard(item_id)

    def _require_item(self, item_id: str) -> Item:
        if item_id not in self.items:
            raise KeyError(f"unknown item '{item_id}'")
        return self.items[item_id]

    def _require_location(self, item_id: str) -> ItemLocation:
        if item_id not in self.item_locations:
            raise KeyError(f"missing location for item '{item_id}'")
        return self.item_locations[item_id]

    def _require_room(self, room_id: str) -> None:
        if room_id not in self.rooms:
            raise KeyError(f"unknown room '{room_id}'")
