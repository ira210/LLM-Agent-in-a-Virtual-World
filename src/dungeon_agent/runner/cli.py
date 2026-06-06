"""Main runner CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from dungeon_agent.agent.policy import NullAgentPolicy, OpenAIAgentPolicy
from dungeon_agent.console import ui_print, ui_prompt
from dungeon_agent.game import ParserExecutorEngine, WorldState
from dungeon_agent.game.layout import DEFAULT_ROOM_COUNT
from dungeon_agent.runner.core import Runner, RunnerMode, compute_turn_limit
from dungeon_agent.runner.hint_stream import write_hint_event_stream
from dungeon_agent.runner.human_hints import FixedIntervalCheckpointPolicy, HumanHintIngestionService
from dungeon_agent.runner.turn_log import write_turn_log
from dungeon_agent.runner.world_factory import build_seeded_world
from dungeon_agent.schemas import HumanHintEvent, HumanHintEventStream, TurnRecord
from dungeon_agent.settings import load_settings_with_overrides


def _sequence_provider(values: list[str] | None) -> Callable[[int, str], str | None]:
    sequence = values or []

    def provider(turn_index: int, _observation_text: str) -> str | None:
        if turn_index < len(sequence):
            return sequence[turn_index]
        return None

    return provider


def _build_cli_world(*, seed: int, room_count: int = DEFAULT_ROOM_COUNT) -> WorldState:
    return build_seeded_world(seed=seed, room_count=room_count)


def resolve_max_turns(*, room_count: int, explicit_max_turns: int | None) -> int:
    baseline = compute_turn_limit(room_count)
    if explicit_max_turns is not None:
        return max(explicit_max_turns, baseline)
    return baseline


def _interactive_human_provider(values: list[str] | None) -> Callable[[int, str], str | None]:
    scripted = values or []

    def provider(turn_index: int, _observation_text: str) -> str | None:
        if turn_index < len(scripted):
            return scripted[turn_index]
        try:
            command = ui_prompt("human>", role="hint_request").strip()
        except EOFError:
            return "LOOK"
        return command or "LOOK"

    return provider


def _interactive_hint_provider() -> Callable[[int, str], str | None]:
    def provider(turn_index: int, _observation_text: str) -> str | None:
        ui_print("[agent] Guidance request: share strategy or press Enter to skip.", role="hint_request")
        try:
            hint = ui_prompt(f"hint[{turn_index}]>", role="hint_request").strip()
        except EOFError:
            return None
        normalized = " ".join(hint.split())
        return normalized or None

    return provider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the dungeon-agent harness.")
    parser.add_argument(
        "--mode",
        choices=("agent-only", "agent+human", "human-only"),
        default="agent-only",
        help="Runner mode.",
    )
    parser.add_argument(
        "--room-count",
        type=int,
        default=DEFAULT_ROOM_COUNT,
        help="Dungeon complexity proxy used for default turn budget.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override seed for deterministic runs.")
    parser.add_argument("--max-turns", type=int, default=None, help="Maximum turns before stopping.")
    parser.add_argument("--model", type=str, default=None, help="Override model name.")
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Override planner goal text for this run.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Checkpoint spacing for hint acceptance in agent+human mode.",
    )
    parser.add_argument(
        "--hint",
        action="append",
        default=None,
        help="Pre-seeded human hint value (repeatable, one per turn).",
    )
    parser.add_argument(
        "--human-command",
        action="append",
        default=None,
        help="Pre-seeded human command (repeatable, used in human-only mode).",
    )
    parser.add_argument("--turn-log-path", type=Path, default=None, help="Path for replay turn-log JSONL.")
    parser.add_argument("--hint-log-path", type=Path, default=None, help="Path for hint event-stream JSONL.")
    parser.add_argument("--dry-run", action="store_true", help="Enable dry-run mode.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug output.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    max_turns = resolve_max_turns(room_count=args.room_count, explicit_max_turns=args.max_turns)
    settings = load_settings_with_overrides(
        seed=args.seed,
        max_turns=max_turns,
        model_name=args.model,
        dry_run=args.dry_run if args.dry_run else None,
        debug_output=True if args.debug else None,
    )
    mode: RunnerMode = args.mode
    seed = settings.runtime.resolved_seed()
    run_id = f"run-{seed}"
    if settings.runtime.debug_output:
        ui_print(
            "Starting dungeon-agent "
            f"mode={mode} model={settings.model.model_name} seed={seed} "
            f"max_turns={settings.runtime.max_turns} room_count={args.room_count} "
            f"dry_run={settings.runtime.dry_run}",
            role="debug",
        )

    hint_service = None
    if mode == "agent+human":
        hint_service = HumanHintIngestionService(
            mode="agent+human",
            checkpoint_policy=FixedIntervalCheckpointPolicy(
                turn_interval=args.checkpoint_interval,
                first_checkpoint_turn=0,
            ),
        )
    should_write_logs = (
        settings.runtime.enable_replay_log or args.turn_log_path is not None or args.hint_log_path is not None
    )
    turn_records: list[TurnRecord] = []
    hint_events: list[HumanHintEvent] = []
    try:
        policy = (
            OpenAIAgentPolicy(
                model_name=settings.model.model_name,
                temperature=settings.model.temperature,
                max_output_tokens=settings.model.max_output_tokens,
                request_timeout_seconds=settings.model.request_timeout_seconds,
                debug_output=settings.runtime.debug_output,
            )
            if mode in {"agent-only", "agent+human"}
            else NullAgentPolicy()
        )
    except RuntimeError as exc:
        ui_print(f"Configuration error: {exc}", role="error", stderr=True)
        return 2

    runner = Runner(
        agent_initiated_hints=mode == "agent+human" and args.hint is None,
        mode=mode,
        seed=seed,
        run_id=run_id,
        max_turns=settings.runtime.max_turns,
        engine=ParserExecutorEngine(
            lambda world_seed: _build_cli_world(seed=world_seed, room_count=args.room_count)
        ),
        policy=policy,
        hint_service=hint_service,
        hint_provider=_sequence_provider(args.hint)
        if args.hint is not None
        else _interactive_hint_provider(),
        human_command_provider=_interactive_human_provider(args.human_command)
        if mode == "human-only"
        else _sequence_provider(args.human_command),
        goal_text=args.goal,
        debug_output=settings.runtime.debug_output,
        turn_record_sink=turn_records.append if should_write_logs else None,
        hint_event_sink=hint_events.append if should_write_logs else None,
    )
    runner.run()
    if should_write_logs:
        turn_log_path = args.turn_log_path or Path(f"{run_id}.turns.jsonl")
        hint_log_path = args.hint_log_path or Path(f"{run_id}.hints.jsonl")
        write_turn_log(turn_log_path, turn_records)
        write_hint_event_stream(hint_log_path, HumanHintEventStream(run_id=run_id, events=hint_events))
        ui_print(f"Replay logs written: turn_log={turn_log_path} hint_log={hint_log_path}", role="debug")
    if settings.runtime.debug_output:
        ui_print(
            "Runner finished. "
            f"mode={mode} seed={seed} max_turns={settings.runtime.max_turns}",
            role="debug",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
