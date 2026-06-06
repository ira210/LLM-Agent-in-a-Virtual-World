from __future__ import annotations

from dungeon_agent.game.domain import Direction, Item, ItemRelationType, PlayerState, Room, WorldState
from dungeon_agent.game.parser_executor import CommandParser, ParserExecutorEngine


def _build_world(*, ambient_light: bool) -> WorldState:
    cellar = Room(
        room_id="cellar",
        name="Cellar",
        description="A damp cellar with a low ceiling.",
        exits={Direction.NORTH: "hall"},
        has_ambient_light=ambient_light,
    )
    hall = Room(
        room_id="hall",
        name="Hall",
        description="A narrow hall with cracked plaster.",
        exits={Direction.SOUTH: "cellar"},
        has_ambient_light=True,
    )
    world = WorldState(
        rooms={"cellar": cellar, "hall": hall},
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
                short_description="a silver coin glints from a hidden nook",
                detail="A tarnished silver coin.",
            ),
            "key": Item(
                item_id="key",
                name="Key",
                short_description="a small key lies where the crate stood",
                detail="A small iron key.",
            ),
            "ring": Item(
                item_id="ring",
                name="Ring",
                short_description="a brass ring sits in the dust beneath the crate",
                detail="A plain brass ring.",
            ),
            "lamp": Item(
                item_id="lamp",
                name="Lamp",
                short_description="an oil lamp hangs from a hook",
                detail="A brass lamp with fresh oil.",
                is_light_source=True,
            ),
            "salted_ration": Item(
                item_id="salted_ration",
                name="Salted Ration",
                short_description="a wrapped ration packet sits on a ledge",
                detail="Dried rations wrapped in waxed cloth.",
            ),
            "leather_armour": Item(
                item_id="leather_armour",
                name="Leather Armour",
                short_description="studded leather armour rests on a peg",
                detail="A worn but sturdy leather cuirass.",
                is_wearable=True,
            ),
            "stone_sarcophagus": Item(
                item_id="stone_sarcophagus",
                name="Stone Sarcophagus",
                short_description="a carved sarcophagus with a heavy lid stands against the wall",
                detail="A carved sarcophagus with a heavy lid that can be shifted further.",
                portable=False,
            ),
            "vault_chest": Item(
                item_id="vault_chest",
                name="Vault Chest",
                short_description="a cedar chest with iron bands rests nearby",
                detail="A stout chest with a stubborn lid and iron latch.",
                portable=False,
            ),
        },
        player=PlayerState(current_room_id="cellar"),
    )
    world.place_item_in_room("crate", "cellar")
    world.place_item_in_room("lamp", "cellar")
    world.place_item_with_relation("coin", anchor_item_id="crate", relation_type=ItemRelationType.IN)
    world.place_item_with_relation("key", anchor_item_id="crate", relation_type=ItemRelationType.BEHIND)
    world.place_item_with_relation("ring", anchor_item_id="crate", relation_type=ItemRelationType.UNDER)
    world.place_item_in_room("salted_ration", "cellar")
    world.place_item_in_room("leather_armour", "cellar")
    world.place_item_in_room("stone_sarcophagus", "cellar")
    world.place_item_in_room("vault_chest", "cellar")
    return world


def test_command_parser_aliases_and_normalization() -> None:
    parser = CommandParser()

    assert parser.parse("n").parsed.normalized_command == "GO NORTH"
    assert parser.parse("go w").parsed.normalized_command == "GO WEST"
    assert parser.parse("l").parsed.normalized_command == "LOOK"
    assert parser.parse("inv").parsed.normalized_command == "INVENTORY"
    assert parser.parse("x crate").parsed.normalized_command == "EXAMINE CRATE"
    assert parser.parse("get coin").parsed.normalized_command == "TAKE COIN"
    assert parser.parse("use key on crate").parsed.normalized_command == "USE KEY ON CRATE"
    assert parser.parse("commands").parsed.normalized_command == "HELP"
    assert parser.parse("q").parsed.normalized_command == "QUIT"
    assert parser.parse("don leather armour").parsed.normalized_command == "WEAR LEATHER ARMOUR"
    assert parser.parse("turns").parsed.normalized_command == "MOVES"


def test_command_parser_rejects_invalid_command_shapes() -> None:
    parser = CommandParser()

    assert "cannot be empty" in parser.parse("   ").error.lower()
    assert "one command is allowed" in parser.parse("look && north").error.lower()
    assert "unknown direction" in parser.parse("go northeast").error.lower()


def test_executor_handles_core_transitions_and_relation_gates() -> None:
    engine = ParserExecutorEngine.from_world_template(_build_world(ambient_light=True))
    engine.reset(seed=7)

    assert "concealed" in engine.step("search room").result_text.lower()
    open_result = engine.step("open crate").result_text.lower()
    assert "reveal coin" in open_result
    take_coin = engine.step("take coin").result_text.lower()
    assert "take the coin" in take_coin
    move_result = engine.step("move crate").result_text.lower()
    assert "reveal key" in move_result
    assert "ring" in move_result
    take_key = engine.step("get key").result_text.lower()
    assert "take the key" in take_key
    take_ring = engine.step("take ring").result_text.lower()
    assert "take the ring" in take_ring
    inventory_result = engine.step("i").result_text
    assert "Coin" in inventory_result and "Key" in inventory_result and "Ring" in inventory_result
    movement_outcome = engine.step("north")
    assert "move north" in movement_outcome.result_text.lower()
    assert "to hall" in movement_outcome.result_text.lower()
    assert "Hall." in movement_outcome.observation_text


def test_parser_error_is_non_fatal_and_returns_feedback() -> None:
    engine = ParserExecutorEngine.from_world_template(_build_world(ambient_light=True))
    engine.reset(seed=1)

    invalid = engine.step("dance quickly")
    assert invalid.done is False
    assert "parser error" in invalid.result_text.lower()

    follow_up = engine.step("look")
    assert "cellar" in follow_up.result_text.lower()


def test_search_is_effective_only_when_lit() -> None:
    dark_engine = ParserExecutorEngine.from_world_template(_build_world(ambient_light=False))
    dark_engine.reset(seed=2)

    dark_search = dark_engine.step("search room")
    assert "too dark" in dark_search.result_text.lower()
    dark_target_search = dark_engine.step("search crate")
    assert "too dark" in dark_target_search.result_text.lower()
    assert "darkness" in dark_search.observation_text.lower()

    light_use_error = dark_engine.step("use lamp")
    assert "need to hold" in light_use_error.result_text.lower()
    dark_engine.step("take lamp")
    lit_search = dark_engine.step("search room")
    assert "too dark" not in lit_search.result_text.lower()


def test_targeted_search_in_lit_room_provides_relation_hints() -> None:
    engine = ParserExecutorEngine.from_world_template(_build_world(ambient_light=True))
    engine.reset(seed=9)

    hinted = engine.step("search crate").result_text.lower()
    assert "try open crate" in hinted
    assert "try move crate" in hinted

    engine.step("open crate")
    post_open = engine.step("search crate").result_text.lower()
    assert "try move crate" in post_open


def test_search_current_room_name_is_treated_as_search_room() -> None:
    engine = ParserExecutorEngine.from_world_template(_build_world(ambient_light=True))
    engine.reset(seed=9)

    by_name = engine.step("SEARCH CELLAR").result_text.lower()
    by_room = engine.step("SEARCH ROOM").result_text.lower()
    assert by_name == by_room


def test_take_command_uses_permissive_item_matching() -> None:
    engine = ParserExecutorEngine.from_world_template(_build_world(ambient_light=True))
    engine.reset(seed=13)

    assert "take the salted ration" in engine.step("TAKE RATION").result_text.lower()
    engine.step("look")
    engine._world.remove_item_from_inventory("salted_ration")
    assert "take the salted ration" in engine.step("TAKE wrapped ration packet").result_text.lower()
    engine._world.remove_item_from_inventory("salted_ration")
    assert "take the salted ration" in engine.step("TAKE packet").result_text.lower()


def test_take_matching_normalizes_curly_and_straight_apostrophes() -> None:
    room = Room(
        room_id="workshop",
        name="Workshop",
        description="A tool room.",
        exits={},
        has_ambient_light=True,
    )
    world = WorldState(
        rooms={"workshop": room},
        items={
            "mason_hammer": Item(
                item_id="mason_hammer",
                name="Mason's Hammer",
                short_description="a mason’s hammer lies beside loose stone",
                detail="A heavy hammer chipped from years of use.",
            )
        },
        player=PlayerState(current_room_id="workshop"),
    )
    world.place_item_in_room("mason_hammer", "workshop")
    engine = ParserExecutorEngine.from_world_template(world)
    engine.reset(seed=1)

    assert "take the mason's hammer" in engine.step("TAKE MASON'S HAMMER").result_text.lower()


def test_examine_matching_normalizes_modifier_apostrophe_variant() -> None:
    room = Room(
        room_id="workshop",
        name="Workshop",
        description="A tool room.",
        exits={},
        has_ambient_light=True,
    )
    world = WorldState(
        rooms={"workshop": room},
        items={
            "mason_hammer": Item(
                item_id="mason_hammer",
                name="Mason's Hammer",
                short_description="a mason’s hammer lies beside loose stone",
                detail="A heavy hammer chipped from years of use.",
            )
        },
        player=PlayerState(current_room_id="workshop"),
    )
    world.place_item_in_room("mason_hammer", "workshop")
    engine = ParserExecutorEngine.from_world_template(world)
    engine.reset(seed=1)

    assert "heavy hammer" in engine.step("EXAMINE MASONʼS HAMMER").result_text.lower()


def test_help_and_commands_alias_return_syntax_reference() -> None:
    engine = ParserExecutorEngine.from_world_template(_build_world(ambient_light=True))
    engine.reset(seed=21)

    help_result = engine.step("HELP").result_text
    commands_result = engine.step("COMMANDS").result_text
    assert "Available commands" in help_result
    assert "HELP (or COMMANDS)" in help_result
    assert "WEAR <target> (DON)" in help_result
    assert "MOVES (or TURNS)" in help_result
    assert "QUIT" in help_result
    assert commands_result == help_result
    assert engine.snapshot().turn_index == 0


def test_quit_requires_confirmation_and_then_ends_run() -> None:
    engine = ParserExecutorEngine.from_world_template(_build_world(ambient_light=True))
    engine.reset(seed=55)

    first_quit = engine.step("QUIT")
    assert first_quit.done is False
    assert "confirm quit" in first_quit.result_text.lower()

    second_quit = engine.step("QUIT")
    assert second_quit.done is True
    assert "end your run" in second_quit.result_text.lower()


def test_contextual_part_reference_prefers_last_examined_item() -> None:
    engine = ParserExecutorEngine.from_world_template(_build_world(ambient_light=True))
    engine.reset(seed=89)

    examine = engine.step("EXAMINE SARCOPHAGUS").result_text.lower()
    assert "heavy lid" in examine
    move = engine.step("MOVE LID").result_text.lower()
    assert "move the stone sarcophagus" in move


def test_examine_resolves_when_phrase_contains_full_descriptive_clause() -> None:
    engine = ParserExecutorEngine.from_world_template(_build_world(ambient_light=True))
    engine.reset(seed=91)

    result = engine.step("EXAMINE A STONE SARCOPHAGUS RESTS WITH ITS LID AJAR").result_text.lower()
    assert "heavy lid" in result


def test_move_resolves_minor_single_token_typo_for_visible_item() -> None:
    room = Room(
        room_id="gallery",
        name="Gallery",
        description="A long gallery with frayed banners.",
        exits={Direction.SOUTH: "hall"},
        has_ambient_light=True,
    )
    hall = Room(
        room_id="hall",
        name="Hall",
        description="A narrow hall.",
        exits={Direction.NORTH: "gallery"},
        has_ambient_light=True,
    )
    world = WorldState(
        rooms={"gallery": room, "hall": hall},
        items={
            "tarnished_tapestry": Item(
                item_id="tarnished_tapestry",
                name="Tarnished Tapestry",
                short_description="a tarnished tapestry hangs crooked on iron hooks",
                detail="An old tapestry with brittle fabric and rusted rings.",
                portable=False,
            )
        },
        player=PlayerState(current_room_id="gallery"),
    )
    world.place_item_in_room("tarnished_tapestry", "gallery")
    engine = ParserExecutorEngine.from_world_template(world)
    engine.reset(seed=3)

    result = engine.step("MOVE TARNESTRY").result_text.lower()
    assert "move the tarnished tapestry" in result


def test_examine_room_feature_from_description_returns_generic_detail() -> None:
    vault = Room(
        room_id="vault",
        name="Sunken Treasure Vault",
        description="Broken pedestals ring a sunken dais where gold dust glitters in stale air.",
        exits={Direction.NORTH: "hall"},
        has_ambient_light=True,
    )
    hall = Room(
        room_id="hall",
        name="Hall",
        description="A narrow hall.",
        exits={Direction.SOUTH: "vault"},
        has_ambient_light=True,
    )
    world = WorldState(rooms={"vault": vault, "hall": hall}, items={}, player=PlayerState(current_room_id="vault"))
    engine = ParserExecutorEngine.from_world_template(world)
    engine.reset(seed=91)

    result = engine.step("EXAMINE PEDESTALS").result_text.lower()
    assert "part of the scenery" in result


def test_open_room_feature_door_returns_feature_feedback() -> None:
    vault = Room(
        room_id="vault",
        name="Vault Antechamber",
        description="A circular room of polished basalt channels every sound toward a sealed bronze door.",
        exits={Direction.EAST: "hall"},
        has_ambient_light=True,
    )
    hall = Room(
        room_id="hall",
        name="Hall",
        description="A narrow hall.",
        exits={Direction.WEST: "vault"},
        has_ambient_light=True,
    )
    world = WorldState(rooms={"vault": vault, "hall": hall}, items={}, player=PlayerState(current_room_id="vault"))
    engine = ParserExecutorEngine.from_world_template(world)
    engine.reset(seed=92)

    result = engine.step("OPEN DOOR").result_text.lower()
    assert "fixed in place" in result


def test_use_key_without_target_in_door_room_is_not_generic_noop() -> None:
    vault = Room(
        room_id="vault",
        name="Vault Antechamber",
        description="A circular room of polished basalt channels every sound toward a sealed bronze door.",
        exits={Direction.EAST: "hall"},
        has_ambient_light=True,
    )
    hall = Room(
        room_id="hall",
        name="Hall",
        description="A narrow hall.",
        exits={Direction.WEST: "vault"},
        has_ambient_light=True,
    )
    world = WorldState(
        rooms={"vault": vault, "hall": hall},
        items={
            "vault_key": Item(
                item_id="vault_key",
                name="Vault Key",
                short_description="a narrow iron key glints in dust",
                detail="A long-shafted key etched with ward markings.",
            )
        },
        player=PlayerState(current_room_id="vault"),
    )
    world.place_item_in_inventory("vault_key")
    engine = ParserExecutorEngine.from_world_template(world)
    engine.reset(seed=93)

    result = engine.step("USE VAULT KEY").result_text.lower()
    assert "does not fit" in result


def test_examine_does_not_resolve_scenery_word_to_item() -> None:
    room = Room(
        room_id="crypt",
        name="Flooded Crypt",
        description="Shallow black water ripples between stone biers.",
        exits={Direction.NORTH: "hall"},
        has_ambient_light=True,
    )
    hall = Room(
        room_id="hall",
        name="Hall",
        description="A narrow hall.",
        exits={Direction.SOUTH: "crypt"},
        has_ambient_light=True,
    )
    world = WorldState(
        rooms={"crypt": room, "hall": hall},
        items={
            "copper_token": Item(
                item_id="copper_token",
                name="Copper Token",
                short_description="a copper token glints in pooled water",
                detail="A token stamped with a moon crest.",
            )
        },
        player=PlayerState(current_room_id="crypt"),
    )
    world.place_item_in_room("copper_token", "crypt")
    engine = ParserExecutorEngine.from_world_template(world)
    engine.reset(seed=95)

    result = engine.step("INSPECT WATER").result_text.lower()
    assert "part of the scenery" in result


def test_use_useless_item_returns_richer_flavor_feedback() -> None:
    engine = ParserExecutorEngine.from_world_template(_build_world(ambient_light=True))
    engine.reset(seed=96)
    engine.step("TAKE RATION")

    result = engine.step("USE RATION").result_text.lower()
    assert "dried rations wrapped in waxed cloth" in result
    assert "valuable" in result


def test_wear_requires_inventory_and_wearable_item() -> None:
    engine = ParserExecutorEngine.from_world_template(_build_world(ambient_light=True))
    engine.reset(seed=90)

    not_held = engine.step("WEAR LEATHER ARMOUR").result_text.lower()
    assert "need to be carrying" in not_held

    engine.step("TAKE LEATHER ARMOUR")
    worn = engine.step("WEAR LEATHER ARMOUR")
    assert "wear the leather armour" in worn.result_text.lower()
    assert "You are wearing: Leather Armour." in engine.step("INVENTORY").result_text

    not_wearable = engine.step("WEAR LAMP").result_text.lower()
    assert "need to be carrying" in not_wearable
    engine.step("TAKE LAMP")
    not_wearable = engine.step("WEAR LAMP").result_text.lower()
    assert "cannot be worn" in not_wearable


def test_locked_container_requires_correct_key_before_opening() -> None:
    cellar = Room(
        room_id="cellar",
        name="Cellar",
        description="A damp cellar.",
        exits={Direction.NORTH: "hall"},
        has_ambient_light=True,
    )
    hall = Room(
        room_id="hall",
        name="Hall",
        description="A narrow hall.",
        exits={Direction.SOUTH: "cellar"},
        has_ambient_light=True,
    )
    world = WorldState(
        rooms={"cellar": cellar, "hall": hall},
        items={
            "prayer_chest": Item(
                item_id="prayer_chest",
                name="Prayer Chest",
                short_description="a prayer chest rests beneath the winch",
                detail="A cedar chest with iron corners and a narrow lockplate.",
                portable=False,
                is_container=True,
                is_locked=True,
                key_item_id="vault_key",
            ),
            "vault_key": Item(
                item_id="vault_key",
                name="Vault Key",
                short_description="a narrow iron key glints in dust",
                detail="A long-shafted key etched with ward markings.",
            ),
            "coffer_key": Item(
                item_id="coffer_key",
                name="Coffer Key",
                short_description="a broad brass key lies in a felt-lined slot",
                detail="A brass key with an inner ward profile.",
            ),
            "coin": Item(
                item_id="coin",
                name="Coin",
                short_description="a silver coin glints in the chest",
                detail="A tarnished silver coin.",
            ),
        },
        player=PlayerState(current_room_id="cellar"),
    )
    world.place_item_in_room("prayer_chest", "cellar")
    world.place_item_in_inventory("vault_key")
    world.place_item_in_inventory("coffer_key")
    world.place_item_with_relation("coin", anchor_item_id="prayer_chest", relation_type=ItemRelationType.IN)
    engine = ParserExecutorEngine.from_world_template(world)
    engine.reset(seed=94)

    locked = engine.step("OPEN PRAYER CHEST").result_text.lower()
    assert "is locked" in locked

    wrong_key = engine.step("USE COFFER KEY ON PRAYER CHEST").result_text.lower()
    assert "does not fit" in wrong_key

    unlock = engine.step("USE VAULT KEY ON PRAYER CHEST").result_text.lower()
    assert "unlock the prayer chest" in unlock

    post_unlock_examine = engine.step("EXAMINE PRAYER CHEST").result_text.lower()
    assert "it is closed." in post_unlock_examine

    opened = engine.step("OPEN PRAYER CHEST").result_text.lower()
    assert "you reveal coin" in opened

    post_open_examine = engine.step("EXAMINE PRAYER CHEST").result_text.lower()
    assert "it is open." in post_open_examine


def test_room_description_updates_when_described_item_is_taken() -> None:
    cellar = Room(
        room_id="cellar",
        name="Cellar",
        description="A damp cellar where a wrapped ration packet sits on a ledge.",
        exits={Direction.NORTH: "hall"},
        has_ambient_light=True,
    )
    hall = Room(
        room_id="hall",
        name="Hall",
        description="A narrow hall.",
        exits={Direction.SOUTH: "cellar"},
        has_ambient_light=True,
    )
    world = WorldState(
        rooms={"cellar": cellar, "hall": hall},
        items={
            "salted_ration": Item(
                item_id="salted_ration",
                name="Salted Ration",
                short_description="a wrapped ration packet sits on a ledge",
                detail="Dried rations wrapped in waxed cloth.",
            )
        },
        player=PlayerState(current_room_id="cellar"),
    )
    world.place_item_in_room("salted_ration", "cellar")
    engine = ParserExecutorEngine.from_world_template(world)
    engine.reset(seed=12)

    before = engine.step("LOOK").observation_text.lower()
    assert "wrapped ration packet sits on a ledge" in before

    after_take = engine.step("TAKE RATION").observation_text.lower()
    assert "wrapped ration packet sits on a ledge" not in after_take
