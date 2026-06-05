"""Fixed-seed evaluation harness and reporting."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Sequence

from dungeon_agent.agent.policy import AgentPolicy, NullAgentPolicy
from dungeon_agent.game import (
    Direction,
    GameState,
    Item,
    ItemRelationType,
    ParserExecutorEngine,
    PlayerState,
    Room,
    TurnOutcome,
    WorldState,
)
from dungeon_agent.runner.core import RunSummary, Runner, TurnOutput, compute_turn_limit
from dungeon_agent.schemas import ReplayArtifact, RunMetadata, TurnRecord

DEFAULT_EVAL_SEEDS: tuple[int, int, int, int, int] = (11, 23, 37, 53, 71)
DEFAULT_THRESHOLD = 0.80


@dataclass(frozen=True, slots=True)
class SeedOutcome:
    seed: int
    run_id: str
    turns: int
    max_turns: int
    terminal: bool
    treasure_retrieved: bool
    exit_reached: bool
    success: bool


@dataclass(frozen=True, slots=True)
class EvalReport:
    seeds: tuple[int, ...]
    outcomes: tuple[SeedOutcome, ...]
    success_rate: float
    threshold: float
    passed: bool


class _SilentPrinter:
    def write_turn(self, output: TurnOutput) -> None:
        _ = output

    def write_summary(self, summary: RunSummary) -> None:
        _ = summary


class GoalAwareParserExecutorEngine:
    """Parser executor wrapper that marks runs complete when treasure returns to exit."""

    def __init__(
        self,
        *,
        world_factory: Callable[[int], WorldState],
        exit_room_id: str,
        treasure_item_id: str,
    ) -> None:
        self._engine = ParserExecutorEngine(world_factory)
        self._exit_room_id = exit_room_id
        self._treasure_item_id = treasure_item_id
        self._world: WorldState | None = None

    def reset(self, *, seed: int) -> str:
        observation = self._engine.reset(seed=seed)
        self._world = self._engine._world
        return observation

    def step(self, command: str) -> TurnOutcome:
        outcome = self._engine.step(command)
        world = self._engine._world
        self._world = world
        if world is None:
            return outcome
        if world.player.current_room_id == self._exit_room_id and self._treasure_item_id in world.player.inventory:
            world.done = True
            return TurnOutcome(
                observation_text=outcome.observation_text,
                result_text=outcome.result_text,
                done=True,
            )
        return outcome

    def snapshot(self) -> GameState:
        return self._engine.snapshot()


def _build_eval_world(_seed: int) -> WorldState:
    foyer = Room(
        room_id="foyer",
        name="Foyer",
        description="A drafty foyer with a white paint mark by the exit.",
        exits={Direction.NORTH: "vault"},
        has_ambient_light=True,
    )
    vault = Room(
        room_id="vault",
        name="Vault",
        description="A cramped vault lined with old lockboxes.",
        exits={Direction.SOUTH: "foyer"},
        has_ambient_light=True,
    )
    world = WorldState(
        rooms={"foyer": foyer, "vault": vault},
        items={
            "chest": Item(
                item_id="chest",
                name="Chest",
                short_description="an iron chest sits in the corner",
                detail="A heavy iron chest with an old latch.",
                portable=False,
                is_container=True,
            ),
            "sunshard_treasure": Item(
                item_id="sunshard_treasure",
                name="Sunshard Treasure",
                short_description="a bright shard glints within",
                detail="A warm crystal pulsing with faint amber light.",
            ),
        },
        player=PlayerState(current_room_id="foyer"),
    )
    world.place_item_in_room("chest", "vault")
    world.place_item_with_relation("sunshard_treasure", anchor_item_id="chest", relation_type=ItemRelationType.IN)
    return world


def resolve_eval_seeds(seeds: Sequence[int] | None = None) -> tuple[int, ...]:
    resolved = tuple(DEFAULT_EVAL_SEEDS if seeds is None else seeds)
    if len(resolved) != 5:
        raise ValueError("evaluation requires exactly 5 seeds")
    return resolved


def summarize_replay(artifact: ReplayArtifact) -> SeedOutcome:
    terminal = artifact.turns[-1].terminal if artifact.turns else False
    treasure_retrieved = any(turn.state_flags.has_treasure for turn in artifact.turns)
    exit_reached = any(turn.state_flags.at_exit and turn.state_flags.has_treasure for turn in artifact.turns)
    turns = len(artifact.turns)
    max_turns = artifact.metadata.max_turns
    success = terminal and treasure_retrieved and exit_reached and turns <= max_turns
    return SeedOutcome(
        seed=artifact.metadata.seed,
        run_id=artifact.metadata.run_id,
        turns=turns,
        max_turns=max_turns,
        terminal=terminal,
        treasure_retrieved=treasure_retrieved,
        exit_reached=exit_reached,
        success=success,
    )


def build_report(*, seeds: Sequence[int], outcomes: Sequence[SeedOutcome], threshold: float = DEFAULT_THRESHOLD) -> EvalReport:
    if not seeds:
        raise ValueError("evaluation requires at least one seed")
    success_rate = sum(1 for outcome in outcomes if outcome.success) / len(seeds)
    return EvalReport(
        seeds=tuple(seeds),
        outcomes=tuple(outcomes),
        success_rate=success_rate,
        threshold=threshold,
        passed=success_rate >= threshold,
    )


def run_single_seed(
    *,
    seed: int,
    max_turns: int,
    policy_factory: Callable[[], AgentPolicy] | None = None,
) -> SeedOutcome:
    policy = policy_factory() if policy_factory is not None else NullAgentPolicy()
    run_id = f"eval-{seed}"
    turns: list[TurnRecord] = []
    runner = Runner(
        mode="agent-only",
        seed=seed,
        run_id=run_id,
        max_turns=max_turns,
        engine=GoalAwareParserExecutorEngine(
            world_factory=_build_eval_world,
            exit_room_id="foyer",
            treasure_item_id="sunshard_treasure",
        ),
        policy=policy,
        printer=_SilentPrinter(),
        turn_record_sink=turns.append,
    )
    runner.run()
    artifact = ReplayArtifact(
        metadata=RunMetadata(
            run_id=run_id,
            seed=seed,
            max_turns=max_turns,
            model_provider="openai",
            model_name="evaluation",
        ),
        turns=turns,
        hints=[],
    )
    return summarize_replay(artifact)


def run_fixed_seed_benchmark(
    *,
    max_turns: int,
    threshold: float = DEFAULT_THRESHOLD,
    seeds: Sequence[int] | None = None,
    run_seed: Callable[[int, int], SeedOutcome] | None = None,
) -> EvalReport:
    resolved_seeds = resolve_eval_seeds(seeds)
    run_seed_fn = run_seed or (lambda seed, turn_limit: run_single_seed(seed=seed, max_turns=turn_limit))
    outcomes = [run_seed_fn(seed, max_turns) for seed in resolved_seeds]
    return build_report(seeds=resolved_seeds, outcomes=outcomes, threshold=threshold)


def render_report(report: EvalReport) -> str:
    lines = ["evaluation-summary"]
    for outcome in report.outcomes:
        status = "PASS" if outcome.success else "FAIL"
        lines.append(
            f"seed={outcome.seed} run_id={outcome.run_id} status={status} "
            f"turns={outcome.turns}/{outcome.max_turns} "
            f"treasure={outcome.treasure_retrieved} exit={outcome.exit_reached} terminal={outcome.terminal}"
        )
    threshold_pct = report.threshold * 100
    success_pct = report.success_rate * 100
    verdict = "PASS" if report.passed else "FAIL"
    lines.append(
        f"aggregate success_rate={success_pct:.1f}% threshold={threshold_pct:.1f}% verdict={verdict}"
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed-seed evaluation benchmark.")
    parser.add_argument("--max-turns", type=int, default=compute_turn_limit(2), help="Turn budget per seed run.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum passing success rate (default 0.80).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_fixed_seed_benchmark(max_turns=args.max_turns, threshold=args.threshold)
    print(render_report(report))
    return 0 if report.passed else 1
