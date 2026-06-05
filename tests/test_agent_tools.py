from dungeon_agent.agent.tools import (
    CommandHistory,
    CommandReference,
    CommandValidator,
    GoalManager,
    LoopRecovery,
    MapMemory,
    MapObservation,
    WorldMemory,
)
from dungeon_agent.schemas import ActiveSubgoal, StateFlags, ValidatorAction


def test_map_memory_persists_rooms_exits_and_items() -> None:
    memory = MapMemory()
    memory.observe(MapObservation(room_id="foyer", exits=("north", "east"), items=("torch",)))
    memory.observe(MapObservation(room_id="foyer", exits=("N",), items=("key", "torch")))

    assert memory.has_room("foyer")
    assert memory.get_exits("foyer") == frozenset({"N", "E"})
    assert memory.get_items("foyer") == frozenset({"torch", "key"})


def test_map_memory_ignores_blank_room_and_tokens() -> None:
    memory = MapMemory()
    memory.observe(MapObservation(room_id="  ", exits=("north",), items=("torch",)))
    memory.observe(MapObservation(room_id="foyer", exits=(" ", "south"), items=(" ", "coin")))

    assert memory.has_room("foyer") is True
    assert memory.has_room("  ") is False
    assert memory.get_exits("foyer") == frozenset({"S"})
    assert memory.get_items("foyer") == frozenset({"coin"})


def test_map_memory_add_edge_and_path_to() -> None:
    memory = MapMemory()
    memory.add_edge("foyer", "N", "hall")
    memory.add_edge("hall", "EAST", "kitchen")

    assert memory.path_to("foyer", "kitchen") == ("N", "E")
    assert memory.path_to("kitchen", "foyer") == ("W", "S")


def test_map_memory_identifies_unexplored_exits() -> None:
    memory = MapMemory()
    memory.observe(MapObservation(room_id="foyer", exits=("north", "east", "south")))
    memory.add_edge("foyer", "north", "hall")

    assert memory.unexplored_exits("foyer") == ("E", "S")


def test_map_memory_path_to_nearest_frontier() -> None:
    memory = MapMemory()
    memory.observe(MapObservation(room_id="foyer", exits=("north",)))
    memory.observe(MapObservation(room_id="hall", exits=("south", "east")))
    memory.add_edge("foyer", "north", "hall")

    assert memory.path_to_nearest_frontier("foyer") == ("N",)


def test_command_history_tracks_last_n() -> None:
    history = CommandHistory()
    history.record(command="LOOK", room_id="foyer")
    history.record(command="N", room_id="hall")
    history.record(command="E", room_id="kitchen")

    assert history.last(2) == (("N", "hall"), ("E", "kitchen"))


def test_world_memory_tracks_clues_and_locks() -> None:
    memory = WorldMemory()
    memory.note_locked_target("Prayer Chest is locked")
    memory.add_clue("Twin wards match two keys.")
    memory.observe_room_items("vault", ("an ironbound coffer sits on the dais",))

    assert "Prayer Chest is locked" in memory.locked_targets
    assert "Twin wards match two keys." in memory.clue_notes
    assert "an ironbound coffer sits on the dais" in memory.seen_items_by_room["vault"]


def test_world_memory_tracks_search_targets_by_room() -> None:
    memory = WorldMemory()
    memory.note_search(room_id="vault", target="room")
    memory.note_search(room_id="vault", target="iron chest")
    memory.note_search(room_id="vault", target=" room ")

    assert memory.searched_targets_for_room("vault") == ("IRON CHEST", "ROOM")


def test_goal_manager_transitions_subgoals_in_order() -> None:
    manager = GoalManager()
    assert manager.active_subgoal is ActiveSubgoal.FIND_LIGHT

    assert manager.update(StateFlags(has_light=True)) is ActiveSubgoal.FIND_TREASURE
    assert manager.update(StateFlags(has_light=True), needs_key=True) is ActiveSubgoal.FIND_KEY
    assert manager.update(StateFlags(has_light=True), needs_key=False) is ActiveSubgoal.FIND_TREASURE
    assert manager.update(StateFlags(has_light=True, has_treasure=True)) is ActiveSubgoal.RETURN_TO_EXIT

    assert manager.update(StateFlags()) is ActiveSubgoal.RETURN_TO_EXIT


def test_command_validator_accepts_valid_command() -> None:
    validator = CommandValidator()

    result = validator.validate("GO NORTH")

    assert result.validated_command == "N"
    assert result.action is ValidatorAction.ACCEPTED


def test_command_validator_rewrites_unambiguous_typo() -> None:
    validator = CommandValidator()

    result = validator.validate("go nort")

    assert result.validated_command == "N"
    assert result.action is ValidatorAction.REWRITTEN


def test_command_validator_rewrites_go_direction_typo() -> None:
    validator = CommandValidator()

    result = validator.validate("go sout")

    assert result.validated_command == "S"
    assert result.action is ValidatorAction.REWRITTEN


def test_command_validator_requests_replan_for_ambiguous_invalid_command() -> None:
    validator = CommandValidator()

    result = validator.validate("z")

    assert result.validated_command == ""
    assert result.action is ValidatorAction.REPLAN


def test_command_validator_rejects_multi_command_input() -> None:
    validator = CommandValidator()

    result = validator.validate("LOOK; TAKE TORCH")
    newline_result = validator.validate("LOOK\nTAKE TORCH")

    assert result.action is ValidatorAction.REPLAN
    assert newline_result.action is ValidatorAction.REPLAN


def test_command_validator_accepts_help_aliases() -> None:
    validator = CommandValidator()

    assert validator.validate("help").validated_command == "HELP"
    assert validator.validate("commands").validated_command == "HELP"
    assert "HELP" in CommandReference.text()
    assert "WEAR <target>" in CommandReference.text()


def test_command_validator_accepts_quit_aliases() -> None:
    validator = CommandValidator()

    assert validator.validate("quit").validated_command == "QUIT"
    assert validator.validate("exit").validated_command == "QUIT"
    assert validator.validate("q").validated_command == "QUIT"


def test_command_validator_accepts_wear_aliases() -> None:
    validator = CommandValidator()

    assert validator.validate("wear leather armour").validated_command == "WEAR LEATHER ARMOUR"
    assert validator.validate("don helm").validated_command == "WEAR HELM"


def test_loop_recovery_triggers_on_three_repeated_commands() -> None:
    recovery = LoopRecovery()
    recovery.record_turn(command="look", room_id="r1")
    recovery.record_turn(command=" LOOK ", room_id="r1")
    decision = recovery.record_turn(command="LOOK", room_id="r1")

    assert decision.triggered is True
    assert decision.reason == "repeated_command"
    assert recovery.should_warn_about_loop() is True
    recovery.consume_loop_warning()
    assert recovery.should_recover() is False
    recovery.record_turn(command="LOOK", room_id="r1")
    assert recovery.should_recover() is True


def test_loop_recovery_triggers_on_room_oscillation() -> None:
    recovery = LoopRecovery()
    recovery.record_turn(command="N", room_id="room-a")
    recovery.record_turn(command="S", room_id="room-b")
    decision = recovery.record_turn(command="N", room_id="room-a")

    assert decision.triggered is True
    assert decision.reason == "room_oscillation"


def test_loop_recovery_prefers_non_oscillating_exit_when_available() -> None:
    recovery = LoopRecovery()
    recovery.record_turn(command="S", room_id="room-b")
    recovery.record_turn(command="N", room_id="room-a")
    recovery.record_turn(command="S", room_id="room-b")

    assert recovery.choose_recovery_direction(("north", "west")) == "W"


def test_loop_recovery_does_not_trigger_below_threshold() -> None:
    recovery = LoopRecovery()
    recovery.record_turn(command="LOOK", room_id="r1")
    decision = recovery.record_turn(command="N", room_id="r2")

    assert decision.triggered is False
    assert decision.reason == "none"
    assert recovery.should_recover() is False


def test_loop_recovery_resets_warning_when_pattern_breaks() -> None:
    recovery = LoopRecovery()
    recovery.record_turn(command="LOOK", room_id="r1")
    recovery.record_turn(command="LOOK", room_id="r1")
    recovery.record_turn(command="LOOK", room_id="r1")
    assert recovery.should_warn_about_loop() is True

    recovery.record_turn(command="N", room_id="r2")
    assert recovery.should_warn_about_loop() is False
    assert recovery.should_recover() is False
