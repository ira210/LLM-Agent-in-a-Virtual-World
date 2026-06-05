from dungeon_agent.game.layout import (
    DEFAULT_ROOM_COUNT,
    generate_dungeon_layout,
    validate_dungeon_layout,
)


def test_layout_generation_room_count_and_branching_shape() -> None:
    layout = generate_dungeon_layout(seed=7)

    assert layout.room_count == DEFAULT_ROOM_COUNT
    degrees = layout.degree_map()
    assert any(degree >= 3 for degree in degrees.values())
    assert sum(1 for degree in degrees.values() if degree == 1) >= 2
    assert layout.edge_count() == layout.room_count - 1


def test_lock_and_key_placement_is_occasional_within_bounds() -> None:
    sample_size = 200
    locked_layouts = 0

    for seed in range(sample_size):
        layout = generate_dungeon_layout(seed=seed)
        if layout.locked_connection is None:
            continue
        locked_layouts += 1
        assert layout.key_room_id is not None
        assert layout.locked_connection.key_room_id not in {
            layout.locked_connection.room_a_id,
            layout.locked_connection.room_b_id,
        }

    assert 30 <= locked_layouts <= 110


def test_layout_solvability_invariants_hold_across_seed_samples() -> None:
    for seed in range(150):
        layout = generate_dungeon_layout(seed=seed)
        assert validate_dungeon_layout(layout)
        for room_id, room in layout.rooms.items():
            assert all(destination != room_id for destination in room.exits.values())


def test_layout_generation_is_deterministic_for_a_fixed_seed() -> None:
    seed = 123
    first = generate_dungeon_layout(seed=seed)
    second = generate_dungeon_layout(seed=seed)

    first_edges = sorted(
        tuple(sorted((room_id, neighbor)))
        for room_id in first.rooms
        for neighbor in first.neighbors(room_id)
        if room_id < neighbor
    )
    second_edges = sorted(
        tuple(sorted((room_id, neighbor)))
        for room_id in second.rooms
        for neighbor in second.neighbors(room_id)
        if room_id < neighbor
    )

    assert first.start_room_id == second.start_room_id
    assert first.exit_room_id == second.exit_room_id
    assert first.light_room_id == second.light_room_id
    assert first.treasure_room_id == second.treasure_room_id
    assert first_edges == second_edges
