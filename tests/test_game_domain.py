from __future__ import annotations

import pytest

from dungeon_agent.game.domain import (
    Direction,
    Item,
    ItemRelationType,
    PlayerState,
    Room,
    WorldState,
)
from dungeon_agent.game.observation import render_inventory_observation, render_room_observation


def _build_world(*, ambient_light: bool) -> WorldState:
    room = Room(
        room_id="cellar",
        name="Cellar",
        description="A damp cellar with a low ceiling.",
        exits={Direction.NORTH: "hall", Direction.EAST: "store"},
        has_ambient_light=ambient_light,
    )
    return WorldState(
        rooms={"cellar": room},
        items={
            "crate": Item(
                item_id="crate",
                name="Crate",
                short_description="a heavy crate rests by the wall",
                detail="A reinforced storage crate.",
                portable=False,
                is_container=True,
            ),
            "coin": Item(
                item_id="coin",
                name="Coin",
                short_description="a silver coin glints on the floor",
                detail="A tarnished silver coin.",
            ),
            "key": Item(
                item_id="key",
                name="Key",
                short_description="a small key is tucked out of sight",
                detail="A small iron key.",
            ),
            "lamp": Item(
                item_id="lamp",
                name="Lamp",
                short_description="an old oil lamp hangs from a hook",
                detail="A brass lamp with fresh oil.",
                is_light_source=True,
            ),
        },
        player=PlayerState(current_room_id="cellar"),
    )


def test_relation_visibility_requires_open_or_move_state() -> None:
    world = _build_world(ambient_light=True)
    world.place_item_in_room("crate", "cellar")
    world.place_item_with_relation("coin", anchor_item_id="crate", relation_type=ItemRelationType.IN)
    world.place_item_with_relation("key", anchor_item_id="crate", relation_type=ItemRelationType.BEHIND)

    assert world.visible_items_in_room("cellar") == ["crate"]
    with pytest.raises(ValueError):
        world.add_item_to_inventory("coin")
    with pytest.raises(ValueError):
        world.add_item_to_inventory("key")

    world.set_item_open("crate", is_open=True)
    assert world.is_item_revealed("coin") is True
    assert world.visible_items_in_room("cellar") == ["crate", "coin"]
    world.add_item_to_inventory("coin")
    assert "coin" in world.player.inventory

    world.set_item_moved("crate", is_moved=True)
    assert world.is_item_revealed("key") is True
    world.add_item_to_inventory("key")
    assert world.player.inventory == {"coin", "key"}


def test_relation_cycle_is_rejected() -> None:
    world = _build_world(ambient_light=True)
    world.place_item_in_room("crate", "cellar")
    world.place_item_with_relation("coin", anchor_item_id="crate", relation_type=ItemRelationType.UNDER)
    with pytest.raises(ValueError):
        world.place_item_with_relation("crate", anchor_item_id="coin", relation_type=ItemRelationType.BEHIND)


def test_inventory_placement_keeps_single_location_invariant() -> None:
    world = _build_world(ambient_light=True)
    world.place_item_in_room("lamp", "cellar")
    world.add_item_to_inventory("lamp")

    assert world.item_locations["lamp"].kind.value == "inventory"
    assert world.player.inventory == {"lamp"}

    world.remove_item_from_inventory("lamp")
    assert world.item_locations["lamp"].kind.value == "room"
    assert world.item_locations["lamp"].room_id == "cellar"
    assert world.player.inventory == set()


def test_observation_lit_and_dark_behavior_is_deterministic() -> None:
    dark_world = _build_world(ambient_light=False)
    dark_world.place_item_in_room("crate", "cellar")
    dark_world.place_item_in_room("lamp", "cellar")
    dark_text = render_room_observation(dark_world)

    assert "darkness" in dark_text.lower()
    assert "Exits lead" not in dark_text
    assert "lamp" not in dark_text.lower()

    dark_world.add_item_to_inventory("lamp")
    lit_text = render_room_observation(dark_world)
    lit_text_again = render_room_observation(dark_world)

    assert "Cellar. A damp cellar with a low ceiling." in lit_text
    assert "Exits lead north, east." in lit_text
    assert "heavy crate" in lit_text
    assert lit_text == lit_text_again
    assert render_inventory_observation(dark_world) == "You are carrying: Lamp. Total loot value: 0."


def test_observation_marks_exit_room() -> None:
    world = _build_world(ambient_light=True)
    world.exit_room_id = "cellar"
    text = render_room_observation(world)
    assert "This room is the dungeon exit." in text
