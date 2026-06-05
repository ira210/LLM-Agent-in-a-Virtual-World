from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from dungeon_agent.agent.policy import PolicyInput
from dungeon_agent.game import (
    Direction,
    GameState,
    Item,
    ItemRelationType,
    PlayerState,
    Room,
    TurnOutcome,
    WorldState,
)
from dungeon_agent.game.parser_executor import ParserExecutorEngine
from dungeon_agent.runner.cli import (
    _build_cli_world,
    build_parser,
    main as cli_main,
    resolve_max_turns,
)
from dungeon_agent.runner.core import ConsoleTurnPrinter, RunSummary, Runner, TurnOutput, compute_turn_limit
from dungeon_agent.runner.human_hints import (
    FixedIntervalCheckpointPolicy,
    HumanHintIngestionService,
)


class _RecordingPolicy:
    def __init__(self, command: str = "LOOK") -> None:
        self.command = command
        self.inputs: list[PolicyInput] = []

    def propose_command(self, payload: PolicyInput) -> str:
        self.inputs.append(payload)
        return self.command


class _SequencePolicy:
    def __init__(self, commands: list[str]) -> None:
        self._commands = commands
        self._index = 0
        self.inputs: list[PolicyInput] = []

    def propose_command(self, payload: PolicyInput) -> str:
        self.inputs.append(payload)
        if self._index >= len(self._commands):
            return self._commands[-1]
        command = self._commands[self._index]
        self._index += 1
        return command


class _StubEngine:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self._turn_index = 0

    def reset(self, *, seed: int) -> str:
        _ = seed
        self._turn_index = 0
        return "Stub room."

    def step(self, command: str) -> TurnOutcome:
        self.commands.append(command)
        self._turn_index += 1
        return TurnOutcome(
            observation_text="Stub room.", result_text=f"executed {command}", done=False
        )

    def snapshot(self) -> GameState:
        return GameState(turn_index=self._turn_index, done=False)


class _OscillatingExitEngine:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self._turn_index = 0
        self._room_id = "room-a"
        self._world = SimpleNamespace(
            player=SimpleNamespace(current_room_id="room-a", inventory=[]),
            is_dark=False,
            player_has_light=True,
            exit_room_id=None,
        )

    def reset(self, *, seed: int) -> str:
        _ = seed
        self._turn_index = 0
        self._room_id = "room-a"
        self._world.player.current_room_id = self._room_id
        return self._observation()

    def step(self, command: str) -> TurnOutcome:
        self.commands.append(command)
        self._turn_index += 1
        if self._room_id == "room-a" and command == "S":
            self._room_id = "room-b"
        elif self._room_id == "room-a" and command == "W":
            self._room_id = "room-west"
        elif self._room_id == "room-b" and command == "N":
            self._room_id = "room-a"
        elif self._room_id == "room-b" and command == "W":
            self._room_id = "room-west"
        self._world.player.current_room_id = self._room_id
        return TurnOutcome(
            observation_text=self._observation(),
            result_text=f"executed {command}",
            done=False,
        )

    def snapshot(self) -> GameState:
        return GameState(turn_index=self._turn_index, done=False)

    def _observation(self) -> str:
        if self._room_id == "room-a":
            return "Room A. Exits lead south, west."
        if self._room_id == "room-b":
            return "Room B. Exits lead north, west."
        return "Room West. Exits lead east."


class _LightRoomEngine:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self._turn_index = 0
        self._world = SimpleNamespace(
            player=SimpleNamespace(current_room_id="light-room", inventory=[]),
            is_dark=True,
            player_has_light=False,
            exit_room_id=None,
        )

    def reset(self, *, seed: int) -> str:
        _ = seed
        self._turn_index = 0
        return (
            "Flooded Antechamber. Cold water laps your ankles while stone coffins loom unseen. "
            "Exits lead north. You notice a brass lantern hangs above the bench."
        )

    def step(self, command: str) -> TurnOutcome:
        self.commands.append(command)
        self._turn_index += 1
        return TurnOutcome(
            observation_text=(
                "Flooded Antechamber. Exits lead north. "
                "You notice a brass lantern hangs above the bench."
            ),
            result_text=f"executed {command}",
            done=False,
        )

    def snapshot(self) -> GameState:
        return GameState(turn_index=self._turn_index, done=False)


class _GoalAwareEngine:
    def __init__(self, *, exit_room_id: str, treasure_item_id: str) -> None:
        self._engine = ParserExecutorEngine.from_world_template(_goal_world())
        self._exit_room_id = exit_room_id
        self._treasure_item_id = treasure_item_id

    def reset(self, *, seed: int) -> str:
        return self._engine.reset(seed=seed)

    def step(self, command: str) -> TurnOutcome:
        outcome = self._engine.step(command)
        world = self._engine._world
        if world is None:
            return outcome
        if (
            world.player.current_room_id == self._exit_room_id
            and self._treasure_item_id in world.player.inventory
        ):
            world.done = True
            return TurnOutcome(
                observation_text=outcome.observation_text,
                result_text=outcome.result_text,
                done=True,
            )
        return outcome

    def snapshot(self) -> GameState:
        return self._engine.snapshot()


@dataclass
class _SilentPrinter:
    summaries: list[RunSummary]
    turns: list[TurnOutput]

    def __init__(self) -> None:
        self.summaries = []
        self.turns = []

    def write_turn(self, output: TurnOutput) -> None:
        self.turns.append(output)

    def write_summary(self, summary: RunSummary) -> None:
        self.summaries.append(summary)


class _IntroCapturingPrinter(_SilentPrinter):
    def __init__(self) -> None:
        super().__init__()
        self.intro_goal_text: str | None = None
        self.intro_observation_text: str | None = None
        self.intro_max_turns: int | None = None

    def write_intro(self, *, observation_text: str, goal_text: str, max_turns: int) -> None:
        self.intro_observation_text = observation_text
        self.intro_goal_text = goal_text
        self.intro_max_turns = max_turns


def _world() -> WorldState:
    cellar = Room(
        room_id="cellar",
        name="Cellar",
        description="A damp cellar with a low ceiling.",
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
        },
        player=PlayerState(current_room_id="cellar"),
    )
    world.place_item_in_room("crate", "cellar")
    world.place_item_with_relation(
        "coin", anchor_item_id="crate", relation_type=ItemRelationType.IN
    )
    return world


def _goal_world() -> WorldState:
    foyer = Room(
        room_id="foyer",
        name="Foyer",
        description="A drafty foyer with the dungeon exit marked by chalk.",
        exits={Direction.NORTH: "vault"},
        has_ambient_light=True,
    )
    vault = Room(
        room_id="vault",
        name="Vault",
        description="A cramped vault packed with old lockboxes.",
        exits={Direction.SOUTH: "foyer"},
        has_ambient_light=True,
    )
    world = WorldState(
        rooms={"foyer": foyer, "vault": vault},
        items={
            "chest": Item(
                item_id="chest",
                name="Chest",
                short_description="an iron chest sits against the wall",
                detail="An iron chest with a stiff lid.",
                portable=False,
                is_container=True,
            ),
            "sunshard_treasure": Item(
                item_id="sunshard_treasure",
                name="Sunshard Treasure",
                short_description="a bright treasure shard lies within",
                detail="A warm shard that pulses with amber light.",
                loot_value=250,
            ),
        },
        player=PlayerState(current_room_id="foyer"),
    )
    world.place_item_in_room("chest", "vault")
    world.place_item_with_relation(
        "sunshard_treasure",
        anchor_item_id="chest",
        relation_type=ItemRelationType.IN,
    )
    return world


def test_turn_limit_formula() -> None:
    assert compute_turn_limit(15) == 150
    assert compute_turn_limit(8) == 80
    assert resolve_max_turns(room_count=15, explicit_max_turns=None) == 150
    assert resolve_max_turns(room_count=15, explicit_max_turns=10) == 150


def test_cli_world_uses_full_layout_and_dark_exploration() -> None:
    world = _build_cli_world(seed=7)
    assert len(world.rooms) == 15
    start_room_id = world.player.current_room_id
    assert world.rooms[start_room_id].has_ambient_light is True
    assert world.exit_room_id == start_room_id
    assert world.objective_treasure_item_id == "sunshard_treasure"
    assert world.player.inventory == {"sword", "leather_armour"}
    assert all(
        not room.has_ambient_light
        for room_id, room in world.rooms.items()
        if room_id != start_room_id
    )


def test_runner_intro_goal_names_specific_treasure_and_exit() -> None:
    engine = ParserExecutorEngine(lambda seed: _build_cli_world(seed=seed))
    printer = _IntroCapturingPrinter()
    runner = Runner(
        mode="human-only",
        seed=7,
        max_turns=1,
        engine=engine,
        policy=_RecordingPolicy(),
        human_command_provider=lambda _turn_index, _obs: "look",
        printer=printer,
    )
    runner.run()

    assert printer.intro_goal_text is not None
    assert "Sunshard" in printer.intro_goal_text
    assert "Entry Vestibule" in printer.intro_goal_text
    assert "Secondary goal" in printer.intro_goal_text
    assert "before 1 moves run out" in printer.intro_goal_text
    assert printer.intro_max_turns == 1


def test_moves_command_reports_remaining_without_consuming_turn() -> None:
    engine = _StubEngine()
    printer = _SilentPrinter()
    runner = Runner(
        mode="human-only",
        seed=7,
        max_turns=1,
        engine=engine,
        policy=_RecordingPolicy(),
        human_command_provider=lambda turn_index, _obs: "moves" if turn_index == 0 else "look",
        printer=printer,
    )

    summary = runner.run()

    assert summary.turns_executed == 1
    assert summary.terminal is False
    assert engine.commands == ["LOOK"]
    assert printer.turns[0].emitted_command == "MOVES"
    assert printer.turns[0].result_text == "Moves remaining: 1."


def test_cli_parser_accepts_mode_variants() -> None:
    parser = build_parser()
    assert parser.parse_args(["--mode", "agent-only"]).mode == "agent-only"
    assert parser.parse_args(["--mode", "agent+human"]).mode == "agent+human"
    assert parser.parse_args(["--mode", "human-only"]).mode == "human-only"


def test_cli_agent_mode_fails_fast_without_openai_credentials(
    monkeypatch: object, capsys: object, tmp_path: object
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv", ["dungeon-agent", "--mode", "agent-only", "--seed", "7", "--max-turns", "1"]
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_PROJECT_ID", raising=False)

    exit_code = cli_main()
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Configuration error" in captured.err


def test_cli_human_only_does_not_require_openai_credentials(
    monkeypatch: object, tmp_path: object
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "dungeon-agent",
            "--mode",
            "human-only",
            "--seed",
            "7",
            "--max-turns",
            "1",
            "--human-command",
            "LOOK",
            "--human-command",
            "QUIT",
            "--human-command",
            "QUIT",
        ],
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_PROJECT_ID", raising=False)

    assert cli_main() == 0


def test_runner_selects_command_source_by_mode() -> None:
    printer = _SilentPrinter()
    agent_engine = _StubEngine()
    agent_runner = Runner(
        mode="agent-only",
        seed=1,
        max_turns=1,
        engine=agent_engine,
        policy=_RecordingPolicy(command="LOOK"),
        printer=printer,
    )
    agent_runner.run()
    assert agent_engine.commands == ["LOOK"]

    human_engine = _StubEngine()
    human_runner = Runner(
        mode="human-only",
        seed=1,
        max_turns=1,
        engine=human_engine,
        policy=_RecordingPolicy(command="LOOK"),
        human_command_provider=lambda _turn_index, _obs: "N",
        printer=_SilentPrinter(),
    )
    human_runner.run()
    assert human_engine.commands == ["N"]


def test_runner_loop_recovery_warns_then_overrides_repeated_look() -> None:
    policy = _RecordingPolicy(command="LOOK")
    engine = _StubEngine()
    runner = Runner(
        mode="agent-only",
        seed=1,
        max_turns=5,
        engine=engine,
        policy=policy,
        printer=_SilentPrinter(),
    )
    runner.run()

    assert engine.commands[:4] == ["LOOK", "LOOK", "LOOK", "LOOK"]
    assert engine.commands[4] == "INVENTORY"
    assert policy.inputs[3].loop_warning_text is not None


def test_runner_loop_recovery_breaks_room_oscillation_with_alternate_exit() -> None:
    engine = _OscillatingExitEngine()
    runner = Runner(
        mode="agent-only",
        seed=1,
        max_turns=5,
        engine=engine,
        policy=_SequencePolicy(commands=["S", "N", "S", "N", "N"]),
        printer=_SilentPrinter(),
    )
    runner.run()

    assert engine.commands[:4] == ["S", "N", "S", "N"]
    assert engine.commands[4] == "W"


def test_runner_continues_after_parser_error() -> None:
    commands = ["dance quickly", "look"]
    engine = ParserExecutorEngine.from_world_template(_world())
    printer = _SilentPrinter()
    runner = Runner(
        mode="human-only",
        seed=7,
        max_turns=2,
        engine=engine,
        policy=_RecordingPolicy(),
        human_command_provider=lambda turn_index, _obs: commands[turn_index],
        printer=printer,
    )
    summary = runner.run()

    assert summary.turns_executed == 2
    assert summary.terminal is False
    assert "parser error" in printer.turns[0].result_text.lower()
    assert "cellar" in printer.turns[1].result_text.lower()


def test_runner_prints_post_move_observation() -> None:
    engine = ParserExecutorEngine.from_world_template(_world())
    printer = _SilentPrinter()
    runner = Runner(
        mode="human-only",
        seed=2,
        max_turns=1,
        engine=engine,
        policy=_RecordingPolicy(),
        human_command_provider=lambda _turn_index, _obs: "north",
        printer=printer,
    )
    runner.run()
    assert "hall" in printer.turns[0].observation_text.lower()


def test_runner_prints_agent_planning_status(capsys: object) -> None:
    runner = Runner(
        mode="agent-only",
        seed=2,
        max_turns=1,
        engine=_StubEngine(),
        policy=_RecordingPolicy(command="LOOK"),
        printer=_SilentPrinter(),
    )
    runner.run()
    captured = capsys.readouterr()
    assert "[agent] Planning next move..." in captured.out


def test_help_does_not_consume_turn_budget() -> None:
    commands = ["help", "look"]
    engine = ParserExecutorEngine.from_world_template(_world())
    printer = _SilentPrinter()
    runner = Runner(
        mode="human-only",
        seed=4,
        max_turns=1,
        engine=engine,
        policy=_RecordingPolicy(),
        human_command_provider=lambda turn_index, _obs: (
            commands[turn_index] if turn_index < len(commands) else "look"
        ),
        printer=printer,
    )
    summary = runner.run()
    assert summary.turns_executed == 1
    assert len(printer.turns) == 2
    assert "available commands" in printer.turns[0].result_text.lower()


def test_agent_human_mode_applies_checkpoint_hints() -> None:
    policy = _RecordingPolicy(command="LOOK")
    engine = _StubEngine()
    hint_service = HumanHintIngestionService(
        mode="agent+human",
        checkpoint_policy=FixedIntervalCheckpointPolicy(turn_interval=2, first_checkpoint_turn=0),
    )
    hints = ["find light first", "go north"]
    runner = Runner(
        mode="agent+human",
        seed=3,
        max_turns=2,
        engine=engine,
        policy=policy,
        hint_service=hint_service,
        hint_provider=lambda turn_index, _obs: hints[turn_index],
        printer=_SilentPrinter(),
    )
    runner.run()

    assert policy.inputs[0].human_input_text == "find light first"
    assert policy.inputs[1].human_input_text is None


def test_runner_logs_hint_events_and_turn_records_at_checkpoints() -> None:
    policy = _RecordingPolicy(command="LOOK")
    hint_events = []
    turn_records = []
    runner = Runner(
        mode="agent+human",
        seed=5,
        max_turns=3,
        engine=_StubEngine(),
        policy=policy,
        hint_service=HumanHintIngestionService(
            mode="agent+human",
            checkpoint_policy=FixedIntervalCheckpointPolicy(
                turn_interval=2, first_checkpoint_turn=0
            ),
        ),
        hint_provider=lambda turn_index, _obs: ["north", "skip this", "find treasure"][turn_index],
        turn_record_sink=turn_records.append,
        hint_event_sink=hint_events.append,
        printer=_SilentPrinter(),
    )
    runner.run()

    assert [event.turn_index for event in hint_events] == [0, 2]
    assert [event.hint_text for event in hint_events] == ["north", "find treasure"]
    assert turn_records[0].human_input_text == "north"
    assert turn_records[1].human_input_text is None
    assert turn_records[2].human_input_text == "find treasure"


def test_runner_agent_initiated_hints_request_early_and_with_min_gap() -> None:
    policy = _RecordingPolicy(command="LOOK")
    requested_turns: list[int] = []
    turn_records = []
    runner = Runner(
        mode="agent+human",
        seed=5,
        max_turns=12,
        engine=_StubEngine(),
        policy=policy,
        hint_service=HumanHintIngestionService(
            mode="agent+human",
            checkpoint_policy=FixedIntervalCheckpointPolicy(
                turn_interval=1, first_checkpoint_turn=0
            ),
        ),
        hint_provider=lambda turn_index, _obs: (
            requested_turns.append(turn_index) or f"hint-{turn_index}"
        ),
        agent_initiated_hints=True,
        min_hint_request_gap=10,
        first_hint_request_turn=0,
        turn_record_sink=turn_records.append,
        printer=_SilentPrinter(),
    )
    runner.run()

    assert requested_turns == [0, 10]
    assert turn_records[0].human_input_text == "hint-0"
    assert turn_records[10].human_input_text == "hint-10"


def test_runner_rewrites_take_single_letter_to_visible_light_item() -> None:
    engine = _LightRoomEngine()
    runner = Runner(
        mode="agent-only",
        seed=5,
        max_turns=1,
        engine=engine,
        policy=_RecordingPolicy(command="TAKE L"),
        printer=_SilentPrinter(),
    )
    runner.run()

    assert engine.commands == ["TAKE LANTERN"]


def test_e2e_seeded_human_only_run_retrieves_treasure_and_exits() -> None:
    engine = _GoalAwareEngine(exit_room_id="foyer", treasure_item_id="sunshard_treasure")
    commands = ["north", "open chest", "take sunshard treasure", "south"]
    runner = Runner(
        mode="human-only",
        seed=11,
        max_turns=10,
        engine=engine,
        policy=_RecordingPolicy(command="LOOK"),
        human_command_provider=lambda turn_index, _obs: (
            commands[turn_index] if turn_index < len(commands) else "LOOK"
        ),
        printer=_SilentPrinter(),
    )
    summary = runner.run()

    assert summary.terminal is True
    assert summary.turns_executed == 4
    assert summary.goal_completed is True
    assert summary.total_treasure_value == 250


def test_console_summary_prints_victory_and_total_treasure_value_on_goal_completion(
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = _GoalAwareEngine(exit_room_id="foyer", treasure_item_id="sunshard_treasure")
    commands = ["north", "open chest", "take sunshard treasure", "south"]
    runner = Runner(
        mode="human-only",
        seed=11,
        max_turns=10,
        engine=engine,
        policy=_RecordingPolicy(command="LOOK"),
        human_command_provider=lambda turn_index, _obs: (
            commands[turn_index] if turn_index < len(commands) else "LOOK"
        ),
        printer=ConsoleTurnPrinter(debug_output=False),
    )

    _ = runner.run()
    captured = capsys.readouterr()
    assert "Congratulations, you win! Total treasure value: 250." in captured.out
