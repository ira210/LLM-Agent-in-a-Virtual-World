"""Build deterministic runnable worlds for CLI/replay flows."""

from __future__ import annotations

from random import Random

from dungeon_agent.content.templates import ITEM_TEMPLATES, ROOM_TEMPLATES
from dungeon_agent.game import Item, ItemRelationType, PlayerState, Room, WorldState
from dungeon_agent.game.layout import DEFAULT_ROOM_COUNT, generate_dungeon_layout

_ROOM_TEMPLATE_BY_ID = {template.room_id: template for template in ROOM_TEMPLATES}


def build_seeded_world(*, seed: int, room_count: int = DEFAULT_ROOM_COUNT) -> WorldState:
    """Create a deterministic world from authored content and generated layout."""
    layout = generate_dungeon_layout(seed=seed, room_count=room_count)
    rng = Random(seed ^ 0xBAD5EED)

    rooms: dict[str, Room] = {}
    for room_id, room in layout.rooms.items():
        # Keep the start room lit and treat the rest as dark-by-default until the player has light.
        has_ambient_light = room_id == layout.start_room_id
        rooms[room_id] = Room(
            room_id=room.room_id,
            name=room.name,
            description=room.description,
            dark_description=room.dark_description,
            exits=dict(room.exits),
            has_ambient_light=has_ambient_light,
        )

    world = WorldState(
        rooms=rooms,
        items={template.item_id: template.to_item() for template in ITEM_TEMPLATES},
        player=PlayerState(current_room_id=layout.start_room_id),
        exit_room_id=layout.exit_room_id,
        objective_treasure_item_id="sunshard_treasure",
    )

    if "sword" not in world.items:
        world.items["sword"] = Item(
            item_id="sword",
            name="Sword",
            short_description="a steel sword hangs from your belt",
            detail="A dependable steel sword with a plain crossguard.",
            loot_value=40,
        )
    if "leather_armour" not in world.items:
        world.items["leather_armour"] = Item(
            item_id="leather_armour",
            name="Leather Armour",
            short_description="studded leather armour fits over your tunic",
            detail="Well-worn leather armour, flexible and battle-tested.",
            is_wearable=True,
            loot_value=45,
        )
    world.place_item_in_inventory("sword")
    world.place_item_in_inventory("leather_armour")

    room_ids = sorted(rooms)
    room_tags_by_id = {room_id: _ROOM_TEMPLATE_BY_ID[room_id].room_tags for room_id in room_ids}

    def choose_room_id(preferred_tags: tuple[str, ...]) -> str:
        if preferred_tags:
            candidates = [
                room_id
                for room_id in room_ids
                if any(tag in room_tags_by_id.get(room_id, ()) for tag in preferred_tags)
            ]
            if candidates:
                return rng.choice(sorted(candidates))
        return rng.choice(room_ids)

    for template in ITEM_TEMPLATES:
        if template.relation_hooks:
            continue
        world.place_item_in_room(template.item_id, choose_room_id(template.preferred_room_tags))

    for template in ITEM_TEMPLATES:
        if not template.relation_hooks:
            continue
        hook = template.relation_hooks[0]
        if hook.anchor_item_id in world.item_locations:
            world.place_item_with_relation(
                template.item_id,
                anchor_item_id=hook.anchor_item_id,
                relation_type=hook.relation_type,
            )
            continue
        world.place_item_in_room(template.item_id, choose_room_id(template.preferred_room_tags))

    # Guarantee a usable light source at the start so runs never dead-end in darkness.
    if "ember_lantern" in world.items:
        world.place_item_in_room("ember_lantern", layout.start_room_id)

    return world
