"""Shared turn-loop runner orchestration for all CLI modes."""

from __future__ import annotations

from dataclasses import dataclass
import re
import threading
import time
from typing import Callable, Literal, Protocol

from dungeon_agent.agent.policy import AgentPolicy, PolicyInput
from dungeon_agent.agent.tools import CommandReference, CommandValidator, GoalManager, LoopRecovery
from dungeon_agent.console import ui_print
from dungeon_agent.game.interfaces import GameEngine, TurnOutcome
from dungeon_agent.schemas import ActiveSubgoal, HumanHintEvent, StateFlags, TurnRecord, ValidatorAction

from .human_hints import HumanHintIngestionService, HintIngestionResult

RunnerMode = Literal["agent-only", "agent+human", "human-only"]


def compute_turn_limit(room_count: int) -> int:
    if room_count <= 0:
        raise ValueError("room_count must be > 0")
    return 10 * room_count


@dataclass(frozen=True, slots=True)
class TurnOutput:
    turn_index: int
    max_turns: int
    mode: RunnerMode
    observation_text: str
    proposed_command: str
    validated_command: str
    emitted_command: str
    result_text: str
    validator_action: ValidatorAction
    active_subgoal: ActiveSubgoal
    loop_recovery_triggered: bool
    hint_accepted: bool = False
    hint_text: str | None = None


@dataclass(frozen=True, slots=True)
class RunSummary:
    mode: RunnerMode
    turns_executed: int
    max_turns: int
    terminal: bool
    goal_completed: bool = False
    total_treasure_value: int = 0


class TurnPrinter(Protocol):
    def write_turn(self, output: TurnOutput) -> None:
        """Emit turn text to terminal output."""

    def write_summary(self, summary: RunSummary) -> None:
        """Emit final run summary to terminal output."""


class ConsoleTurnPrinter:
    def __init__(self, *, debug_output: bool = False) -> None:
        self._debug_output = debug_output

    def write_turn(self, output: TurnOutput) -> None:
        if self._debug_output:
            ui_print(f"[turn {output.turn_index + 1}/{output.max_turns}] mode={output.mode}", role="debug")
            ui_print(f"observation: {output.observation_text}", role="observation")
            if output.hint_text is not None or output.hint_accepted:
                state = "accepted" if output.hint_accepted else "ignored"
                ui_print(f"hint ({state}): {output.hint_text or '<none>'}", role="hint_request")
            ui_print(
                "command: "
                f"proposed='{output.proposed_command}' "
                f"validated='{output.validated_command}' "
                f"emitted='{output.emitted_command}' "
                f"action={output.validator_action.value}",
                role="command",
            )
            ui_print(f"result: {output.result_text}", role="result")
            ui_print(
                f"progress: subgoal={output.active_subgoal.value} "
                f"loop_recovery={output.loop_recovery_triggered}",
                role="debug",
            )
            ui_print("-" * 72, role="divider")
            return

        ui_print(f"> {output.emitted_command}", role="command")
        ui_print(output.result_text, role="result")
        ui_print(output.observation_text, role="observation")
        ui_print("-" * 72, role="divider")

    def write_intro(self, *, observation_text: str, goal_text: str, max_turns: int) -> None:
        if self._debug_output:
            ui_print("[start] implicit command='LOOK'", role="debug")
            ui_print(f"observation: {observation_text}", role="observation")
            ui_print(f"goal: {goal_text}", role="goal")
            ui_print(f"move_limit: {max_turns}", role="debug")
            ui_print("-" * 72, role="divider")
            return
        ui_print(observation_text, role="observation")
        ui_print(f"Goal: {goal_text}", role="goal")
        ui_print(f"Move limit: {max_turns}", role="debug")
        ui_print("-" * 72, role="divider")

    def write_summary(self, summary: RunSummary) -> None:
        status = "success" if summary.terminal else "turn-limit-reached"
        if self._debug_output:
            if summary.goal_completed:
                ui_print(
                    f"victory: congratulations, goal complete. total treasure value={summary.total_treasure_value}",
                    role="summary_success",
                )
            ui_print(
                f"run-complete: mode={summary.mode} turns={summary.turns_executed}/"
                f"{summary.max_turns} status={status}",
                role="debug",
            )
            return
        role = "summary_success" if summary.terminal else "summary_warning"
        if summary.goal_completed:
            ui_print(
                f"Congratulations, you win! Total treasure value: {summary.total_treasure_value}.",
                role="summary_success",
            )
        ui_print(
            f"run-complete: turns={summary.turns_executed}/{summary.max_turns} status={status}",
            role=role,
        )


class Runner:
    def __init__(
        self,
        *,
        mode: RunnerMode,
        seed: int,
        max_turns: int,
        engine: GameEngine,
        policy: AgentPolicy,
        command_validator: CommandValidator | None = None,
        goal_manager: GoalManager | None = None,
        loop_recovery: LoopRecovery | None = None,
        hint_service: HumanHintIngestionService | None = None,
        hint_provider: Callable[[int, str], str | None] | None = None,
        human_command_provider: Callable[[int, str], str | None] | None = None,
        printer: TurnPrinter | None = None,
        debug_output: bool = False,
        goal_text: str = "Retrieve the treasure and exit the dungeon.",
        run_id: str | None = None,
        turn_record_sink: Callable[[TurnRecord], None] | None = None,
        hint_event_sink: Callable[[HumanHintEvent], None] | None = None,
        agent_initiated_hints: bool = False,
        min_hint_request_gap: int = 10,
        first_hint_request_turn: int = 0,
    ) -> None:
        self._mode = mode
        self._seed = seed
        self._max_turns = max_turns
        self._engine = engine
        self._policy = policy
        self._command_validator = command_validator or CommandValidator()
        self._goal_manager = goal_manager or GoalManager()
        self._loop_recovery = loop_recovery or LoopRecovery()
        self._hint_service = hint_service
        self._hint_provider = hint_provider
        self._human_command_provider = human_command_provider
        self._printer = printer or ConsoleTurnPrinter(debug_output=debug_output)
        self._goal_text = goal_text
        self._run_id = run_id or f"run-{seed}"
        self._turn_record_sink = turn_record_sink
        self._hint_event_sink = hint_event_sink
        self._agent_initiated_hints = agent_initiated_hints
        self._min_hint_request_gap = min_hint_request_gap
        self._first_hint_request_turn = first_hint_request_turn
        self._last_hint_request_turn: int | None = None
        if self._min_hint_request_gap <= 0:
            raise ValueError("min_hint_request_gap must be > 0")
        if self._first_hint_request_turn < 0:
            raise ValueError("first_hint_request_turn must be >= 0")

    def run(self) -> RunSummary:
        observation = self._engine.reset(seed=self._seed)
        intro_writer = getattr(self._printer, "write_intro", None)
        if callable(intro_writer):
            intro_writer(
                observation_text=observation,
                goal_text=self._resolve_goal_text(),
                max_turns=self._max_turns,
            )
        turns_executed = 0
        command_index = 0
        terminal = False

        while turns_executed < self._max_turns:
            turn_index = turns_executed
            hint_result = self._ingest_hint(
                run_id=self._run_id,
                turn_index=turn_index,
                observation_text=observation,
            )
            if hint_result.event is not None and self._hint_event_sink is not None:
                self._hint_event_sink(hint_result.event)
            proposed_command = self._select_proposed_command(
                command_index=command_index,
                turns_remaining=self._max_turns - turns_executed,
                observation_text=observation,
                hint_result=hint_result,
            )
            validation = self._command_validator.validate(proposed_command)
            emitted_command = validation.validated_command or proposed_command
            emitted_command = self._rewrite_underspecified_command(
                emitted_command=emitted_command,
                observation_text=observation,
            )
            if emitted_command == "MOVES":
                outcome = TurnOutcome(
                    observation_text=observation,
                    result_text=f"Moves remaining: {self._max_turns - turns_executed}.",
                    done=False,
                    consumed_turn=False,
                )
            else:
                outcome = self._engine.step(emitted_command)
            state_flags = self._read_state_flags()
            active_subgoal = self._goal_manager.update(state_flags)
            room_id = self._read_room_id() or "unknown"
            recovery = self._loop_recovery.record_turn(command=emitted_command, room_id=room_id)
            human_input_text = hint_result.hint_text if self._mode == "agent+human" else None

            if self._turn_record_sink is not None:
                self._turn_record_sink(
                    TurnRecord(
                        run_id=self._run_id,
                        seed=self._seed,
                        turn_index=command_index,
                        observation_text=observation,
                        human_input_text=human_input_text,
                        proposed_command=proposed_command,
                        validated_command=validation.validated_command,
                        agent_command=validation.validated_command,
                        result_text=outcome.result_text,
                        active_subgoal=active_subgoal,
                        loop_recovery_triggered=recovery.triggered,
                        validator_action=validation.action,
                        state_flags=state_flags,
                        terminal=outcome.done,
                        consumed_turn=outcome.consumed_turn,
                    )
                )

            self._printer.write_turn(
                TurnOutput(
                    turn_index=turn_index,
                    max_turns=self._max_turns,
                    mode=self._mode,
                    observation_text=outcome.observation_text,
                    proposed_command=proposed_command,
                    validated_command=validation.validated_command,
                    emitted_command=emitted_command,
                    result_text=outcome.result_text,
                    validator_action=validation.action,
                    active_subgoal=active_subgoal,
                    loop_recovery_triggered=recovery.triggered,
                    hint_accepted=hint_result.accepted,
                    hint_text=hint_result.hint_text,
                )
            )
            feedback_hook = getattr(self._policy, "record_turn_feedback", None)
            if callable(feedback_hook):
                feedback_hook(
                    emitted_command=emitted_command,
                    observation_text=outcome.observation_text,
                    result_text=outcome.result_text,
                )

            observation = outcome.observation_text
            command_index += 1
            if outcome.consumed_turn:
                turns_executed += 1
            if outcome.done or self._goal_completed(state_flags):
                terminal = True
                break

        final_state_flags = self._read_state_flags()
        summary = RunSummary(
            mode=self._mode,
            turns_executed=turns_executed,
            max_turns=self._max_turns,
            terminal=terminal,
            goal_completed=self._goal_completed(final_state_flags),
            total_treasure_value=self._total_treasure_value(),
        )
        self._printer.write_summary(summary)
        return summary

    def _ingest_hint(
        self,
        *,
        run_id: str,
        turn_index: int,
        observation_text: str,
    ) -> HintIngestionResult:
        if self._mode != "agent+human" or self._hint_service is None:
            return HintIngestionResult(hint_text=None, event=None, accepted=False)
        if not self._should_request_human_hint(turn_index):
            return HintIngestionResult(hint_text=None, event=None, accepted=False)
        provided_hint = self._hint_provider(turn_index, observation_text) if self._hint_provider else None
        self._last_hint_request_turn = turn_index
        return self._hint_service.ingest_hint(
            run_id=run_id,
            turn_index=turn_index,
            room_id=self._read_room_id(),
            hint_text=provided_hint,
        )

    def _should_request_human_hint(self, turn_index: int) -> bool:
        if not self._agent_initiated_hints:
            return True
        if turn_index < self._first_hint_request_turn:
            return False
        if self._last_hint_request_turn is None:
            return turn_index == self._first_hint_request_turn
        if turn_index - self._last_hint_request_turn < self._min_hint_request_gap:
            return False
        if self._loop_recovery.should_warn_about_loop() or self._loop_recovery.should_recover():
            return True
        return (turn_index - self._first_hint_request_turn) % self._min_hint_request_gap == 0

    def _select_proposed_command(
        self,
        *,
        command_index: int,
        turns_remaining: int,
        observation_text: str,
        hint_result: HintIngestionResult,
    ) -> str:
        if self._mode == "human-only":
            if self._human_command_provider is None:
                return "LOOK"
            command = self._human_command_provider(command_index, observation_text)
            return command or "LOOK"
        if self._loop_recovery.should_recover():
            return self._recovery_command(observation_text)

        human_input_text = None
        loop_warning_text = None
        if self._mode == "agent+human" and self._hint_service is not None:
            context = self._hint_service.build_decision_context(
                observation_text=observation_text,
                hint_result=hint_result,
            )
            human_input_text = context.human_input_text
        if self._loop_recovery.should_warn_about_loop():
            loop_warning_text = self._loop_recovery.loop_warning_message()
            self._loop_recovery.consume_loop_warning()
        state_flags = self._read_state_flags()
        policy_input = PolicyInput(
            observation_text=observation_text,
            human_input_text=human_input_text,
            loop_warning_text=loop_warning_text,
            is_dark=state_flags.is_dark,
            has_light=state_flags.has_light,
            use_exploration_assist=True,
            command_reference_text=CommandReference.text(),
            moves_remaining=turns_remaining,
        )
        return self._propose_with_progress(policy_input=policy_input)

    def _recovery_command(self, observation_text: str) -> str:
        exits_match = re.search(r"Exits lead ([^.]+)\.", observation_text, flags=re.IGNORECASE)
        if exits_match:
            exits = tuple(raw.strip() for raw in exits_match.group(1).split(",") if raw.strip())
            recovery_direction = self._loop_recovery.choose_recovery_direction(exits)
            if recovery_direction is not None:
                return recovery_direction
        if "You notice" in observation_text:
            return "SEARCH ROOM"
        return "INVENTORY"

    def _read_state_flags(self) -> StateFlags:
        world = self._read_world()
        if world is None:
            return StateFlags()
        has_treasure = any("treasure" in item_id for item_id in world.player.inventory)
        exit_room_id = getattr(self._engine, "_exit_room_id", None) or getattr(world, "exit_room_id", None)
        at_exit = bool(exit_room_id) and world.player.current_room_id == exit_room_id
        return StateFlags(
            is_dark=world.is_dark,
            has_light=world.player_has_light,
            has_treasure=has_treasure,
            at_exit=at_exit,
        )

    def _read_room_id(self) -> str | None:
        world = self._read_world()
        if world is None:
            return None
        return world.player.current_room_id

    def _resolve_goal_text(self) -> str:
        world = self._read_world()
        if world is None:
            return self._goal_text
        treasure_item_id = getattr(world, "objective_treasure_item_id", None)
        exit_room_id = getattr(world, "exit_room_id", None)
        treasure_name = "treasure"
        if treasure_item_id and treasure_item_id in world.items:
            treasure_name = world.items[treasure_item_id].name
        exit_name = "the dungeon exit"
        if exit_room_id and exit_room_id in world.rooms:
            exit_name = world.rooms[exit_room_id].name
        return (
            f"Primary goal: explore and map the dungeon from {exit_name}, the dungeon exit. "
            f"Secondary goal: retrieve the {treasure_name}. Complete this before {self._max_turns} moves run out."
        )

    @staticmethod
    def _goal_completed(state_flags: StateFlags) -> bool:
        return state_flags.has_treasure and state_flags.at_exit

    def _total_treasure_value(self) -> int:
        world = self._read_world()
        if world is None:
            return 0
        return sum(world.items[item_id].loot_value for item_id in world.player.inventory if item_id in world.items)

    def _read_world(self) -> object | None:
        world = getattr(self._engine, "_world", None)
        if world is not None:
            return world
        nested_engine = getattr(self._engine, "_engine", None)
        if nested_engine is None:
            return None
        return getattr(nested_engine, "_world", None)

    def _propose_with_progress(self, *, policy_input: PolicyInput) -> str:
        started_at = time.monotonic()
        stop_event = threading.Event()

        def _status_loop() -> None:
            while not stop_event.wait(5.0):
                elapsed = time.monotonic() - started_at
                ui_print(f"[agent] Still planning next move... ({elapsed:.1f}s)", role="agent_status")

        ui_print("[agent] Planning next move...", role="agent_status")
        status_thread = threading.Thread(target=_status_loop, daemon=True)
        status_thread.start()
        try:
            return self._policy.propose_command(policy_input)
        except KeyboardInterrupt:
            elapsed = time.monotonic() - started_at
            ui_print(f"[agent] Planning interrupted after {elapsed:.1f}s.", role="error")
            raise
        finally:
            stop_event.set()
            status_thread.join(timeout=0.1)
            elapsed = time.monotonic() - started_at
            if elapsed >= 1.0:
                ui_print(f"[agent] Plan ready ({elapsed:.1f}s).", role="agent_status")

    @staticmethod
    def _rewrite_underspecified_command(*, emitted_command: str, observation_text: str) -> str:
        normalized = " ".join(emitted_command.strip().upper().split())
        match = re.match(r"^(TAKE|GET)\s+([A-Z])$", normalized)
        if not match:
            return emitted_command
        initial = match.group(2)
        visible_light_items = []
        observation_upper = observation_text.upper()
        if "LANTERN" in observation_upper:
            visible_light_items.append("LANTERN")
        if "LAMP" in observation_upper:
            visible_light_items.append("LAMP")
        candidates = tuple(item for item in visible_light_items if item.startswith(initial))
        if len(candidates) == 1:
            return f"TAKE {candidates[0]}"
        return emitted_command
