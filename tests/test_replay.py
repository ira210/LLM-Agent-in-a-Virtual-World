from __future__ import annotations

import sys
from pathlib import Path

from dungeon_agent.game import ParserExecutorEngine
from dungeon_agent.runner.cli import _build_cli_world
from dungeon_agent.runner.core import Runner
from dungeon_agent.runner.hint_stream import write_hint_event_stream
from dungeon_agent.runner.human_hints import FixedIntervalCheckpointPolicy, HumanHintIngestionService
from dungeon_agent.runner.replay import main as replay_cli_main
from dungeon_agent.runner.replay_runner import replay_and_compare
from dungeon_agent.runner.turn_log import (
    deserialize_turn_records_jsonl,
    serialize_turn_records_jsonl,
    write_turn_log,
)
from dungeon_agent.schemas import HumanHintEvent, HumanHintEventStream, TurnRecord


class _ScriptedPolicy:
    def __init__(self, commands: list[str]) -> None:
        self._commands = commands
        self._index = 0

    def propose_command(self, _payload: object) -> str:
        command = self._commands[self._index]
        self._index += 1
        return command


class _SilentPrinter:
    def write_turn(self, _output: object) -> None:
        return None

    def write_summary(self, _summary: object) -> None:
        return None


def _record_sample_run(seed: int = 17) -> tuple[list[TurnRecord], list[HumanHintEvent]]:
    turns: list[TurnRecord] = []
    hints: list[HumanHintEvent] = []
    scripted_hints = {0: "Find a light source.", 2: "Check the vault threshold."}
    commands = ["LOOK", "LOOK", "LOOK", "LOOK"]
    runner = Runner(
        mode="agent+human",
        seed=seed,
        run_id=f"run-{seed}",
        max_turns=len(commands),
        engine=ParserExecutorEngine(lambda world_seed: _build_cli_world(seed=world_seed)),
        policy=_ScriptedPolicy(commands),
        hint_service=HumanHintIngestionService(
            mode="agent+human",
            checkpoint_policy=FixedIntervalCheckpointPolicy(turn_interval=1, first_checkpoint_turn=0),
        ),
        hint_provider=lambda turn_index, _observation_text: scripted_hints.get(turn_index),
        printer=_SilentPrinter(),
        turn_record_sink=turns.append,
        hint_event_sink=hints.append,
    )
    runner.run()
    return turns, hints


def test_turn_log_jsonl_serialization_round_trip() -> None:
    turns, _hints = _record_sample_run(seed=23)
    payload = serialize_turn_records_jsonl(turns)
    restored = deserialize_turn_records_jsonl(payload)
    assert restored == turns


def test_replay_determinism_matches_fixed_seed_streams() -> None:
    turns, hints = _record_sample_run(seed=29)
    result = replay_and_compare(turns=turns, hints=hints)
    assert result.success is True


def test_replay_cli_reads_turn_and_hint_logs(tmp_path: Path, capsys: object, monkeypatch: object) -> None:
    turns, hints = _record_sample_run(seed=31)
    turn_log = tmp_path / "turns.jsonl"
    hint_log = tmp_path / "hints.jsonl"
    write_turn_log(turn_log, turns)
    write_hint_event_stream(hint_log, HumanHintEventStream(run_id="run-31", events=hints))

    monkeypatch.setattr(sys, "argv", ["dungeon-agent-replay", "--turn-log", str(turn_log), "--hint-log", str(hint_log)])
    exit_code = replay_cli_main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "replay-ok" in captured.out
