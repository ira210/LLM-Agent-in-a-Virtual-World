"""Deterministic replay runner and equivalence checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dungeon_agent.game import ParserExecutorEngine, WorldState
from dungeon_agent.runner.cli import _build_cli_world
from dungeon_agent.runner.core import RunSummary, Runner, RunnerMode, TurnOutput
from dungeon_agent.runner.hint_stream import read_hint_event_stream
from dungeon_agent.runner.human_hints import FixedIntervalCheckpointPolicy, HumanHintIngestionService
from dungeon_agent.runner.turn_log import read_turn_log
from dungeon_agent.schemas import HumanHintEvent, TurnRecord


@dataclass(frozen=True, slots=True)
class ReplayCheckResult:
    success: bool
    run_id: str
    expected_turns: int
    actual_turns: int
    message: str


class _ScriptedPolicy:
    def __init__(self, commands: list[str]) -> None:
        self._commands = commands
        self._index = 0

    def propose_command(self, _payload: object) -> str:
        if self._index >= len(self._commands):
            return "LOOK"
        command = self._commands[self._index]
        self._index += 1
        return command


class _SilentPrinter:
    def write_turn(self, output: TurnOutput) -> None:
        _ = output

    def write_summary(self, summary: RunSummary) -> None:
        _ = summary


def _resolve_mode(turns: list[TurnRecord], hints: list[HumanHintEvent]) -> RunnerMode:
    if hints or any(turn.human_input_text is not None for turn in turns):
        return "agent+human"
    return "agent-only"


def replay_and_compare(
    *,
    turns: list[TurnRecord],
    hints: list[HumanHintEvent],
    world_factory: Callable[[int], WorldState] | None = None,
) -> ReplayCheckResult:
    if not turns:
        return ReplayCheckResult(
            success=False,
            run_id="",
            expected_turns=0,
            actual_turns=0,
            message="turn log is empty",
        )
    run_id = turns[0].run_id
    seed = turns[0].seed
    if any(turn.run_id != run_id for turn in turns):
        return ReplayCheckResult(
            success=False,
            run_id=run_id,
            expected_turns=len(turns),
            actual_turns=0,
            message="turn log contains multiple run_ids",
        )
    if any(turn.seed != seed for turn in turns):
        return ReplayCheckResult(
            success=False,
            run_id=run_id,
            expected_turns=len(turns),
            actual_turns=0,
            message="turn log contains multiple seeds",
        )

    hint_map = {event.turn_index: event.hint_text for event in hints}
    hint_mode_required = any(turn.human_input_text is not None for turn in turns)
    if hint_mode_required and not hints:
        return ReplayCheckResult(
            success=False,
            run_id=run_id,
            expected_turns=len(turns),
            actual_turns=0,
            message="turn log includes human_input_text but no hint stream was provided",
        )

    for turn in turns:
        expected_hint = hint_map.get(turn.turn_index)
        if turn.human_input_text != expected_hint:
            return ReplayCheckResult(
                success=False,
                run_id=run_id,
                expected_turns=len(turns),
                actual_turns=0,
                message=f"hint mismatch at turn {turn.turn_index}: expected={expected_hint!r} actual={turn.human_input_text!r}",
            )

    replay_turns: list[TurnRecord] = []
    replay_hints: list[HumanHintEvent] = []
    mode = _resolve_mode(turns, hints)
    hint_service = None
    if mode == "agent+human":
        hint_service = HumanHintIngestionService(
            mode="agent+human",
            checkpoint_policy=FixedIntervalCheckpointPolicy(turn_interval=1, first_checkpoint_turn=0),
        )

    build_world = world_factory or (lambda replay_seed: _build_cli_world(seed=replay_seed))
    runner = Runner(
        mode=mode,
        seed=seed,
        run_id=run_id,
        max_turns=max(1, sum(1 for turn in turns if turn.consumed_turn)),
        engine=ParserExecutorEngine(build_world),
        policy=_ScriptedPolicy([turn.proposed_command for turn in turns]),
        hint_service=hint_service,
        hint_provider=lambda turn_index, _observation_text: hint_map.get(turn_index),
        printer=_SilentPrinter(),
        turn_record_sink=replay_turns.append,
        hint_event_sink=replay_hints.append,
    )
    runner.run()
    if len(replay_turns) != len(turns):
        return ReplayCheckResult(
            success=False,
            run_id=run_id,
            expected_turns=len(turns),
            actual_turns=len(replay_turns),
            message="turn count mismatch",
        )

    for expected, replayed in zip(turns, replay_turns):
        if replayed != expected:
            return ReplayCheckResult(
                success=False,
                run_id=run_id,
                expected_turns=len(turns),
                actual_turns=len(replay_turns),
                message=f"turn mismatch at index {expected.turn_index}",
            )

    if replay_hints != hints:
        return ReplayCheckResult(
            success=False,
            run_id=run_id,
            expected_turns=len(turns),
            actual_turns=len(replay_turns),
            message="hint event stream mismatch",
        )
    return ReplayCheckResult(
        success=True,
        run_id=run_id,
        expected_turns=len(turns),
        actual_turns=len(replay_turns),
        message="deterministic replay matched",
    )


def replay_logs(turn_log_path: Path, hint_log_path: Path | None = None) -> ReplayCheckResult:
    turns = read_turn_log(turn_log_path)
    run_id = turns[0].run_id if turns else ""
    hints = []
    if hint_log_path is not None:
        hints = read_hint_event_stream(hint_log_path, run_id=run_id).events
    return replay_and_compare(turns=turns, hints=hints)
