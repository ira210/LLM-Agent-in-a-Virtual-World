from dungeon_agent.content import DISCOVERY_CHAINS, ITEM_TEMPLATES, ROOM_TEMPLATES
from dungeon_agent.game import ItemRelationType


def test_authored_template_counts_and_unique_ids() -> None:
    assert len(ROOM_TEMPLATES) == 15
    assert len(ITEM_TEMPLATES) == 30

    room_ids = [room.room_id for room in ROOM_TEMPLATES]
    item_ids = [item.item_id for item in ITEM_TEMPLATES]
    assert len(room_ids) == len(set(room_ids))
    assert len(item_ids) == len(set(item_ids))


def test_quest_critical_entities_exist() -> None:
    items_by_id = {item.item_id: item for item in ITEM_TEMPLATES}

    assert "ember_lantern" in items_by_id
    assert items_by_id["ember_lantern"].is_light_source is True

    assert "sunshard_treasure" in items_by_id
    assert items_by_id["sunshard_treasure"].is_treasure is True

    assert "vault_key" in items_by_id
    assert "coffer_key" in items_by_id
    assert items_by_id["prayer_chest"].is_locked is True
    assert items_by_id["prayer_chest"].key_item_id == "vault_key"
    assert items_by_id["ironbound_coffer"].is_locked is True
    assert items_by_id["ironbound_coffer"].key_item_id == "coffer_key"


def test_relation_hooks_reference_valid_anchors() -> None:
    item_ids = {item.item_id for item in ITEM_TEMPLATES}
    relation_types = {ItemRelationType.IN, ItemRelationType.UNDER, ItemRelationType.BEHIND}

    for item in ITEM_TEMPLATES:
        for hook in item.relation_hooks:
            assert hook.anchor_item_id in item_ids
            assert hook.relation_type in relation_types
            assert hook.reveal_with_action in {"open", "move", "search"}


def test_discovery_chains_cover_light_and_treasure_paths() -> None:
    chain_ids = {chain.chain_id for chain in DISCOVERY_CHAINS}
    assert {"chain-light", "chain-treasure"}.issubset(chain_ids)

    steps = [step for chain in DISCOVERY_CHAINS for step in chain.steps]
    step_ids = [step.step_id for step in steps]
    assert len(step_ids) == len(set(step_ids))

    room_ids = {room.room_id for room in ROOM_TEMPLATES}
    item_ids = {item.item_id for item in ITEM_TEMPLATES}
    for step in steps:
        assert step.room_id in room_ids
        assert step.focus_item_id in item_ids
        if step.yields_item_id is not None:
            assert step.yields_item_id in item_ids
