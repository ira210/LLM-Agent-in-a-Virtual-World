from __future__ import annotations

from dungeon_agent.eval.harness import (
    DEFAULT_EVAL_SEEDS,
    DEFAULT_THRESHOLD,
    SeedOutcome,
    build_report,
    resolve_eval_seeds,
    run_fixed_seed_benchmark,
    summarize_replay,
)
from dungeon_agent.schemas import ActiveSubgoal, ReplayArtifact, RunMetadata, StateFlags, TurnRecord, ValidatorAction


def _turn(
    *,
    seed: int,
    turn_index: int,
    terminal: bool = False,
    has_treasure: bool = False,
    at_exit: bool = False,
) -> TurnRecord:
    return TurnRecord(
        run_id=f"eval-{seed}",
        seed=seed,
        turn_index=turn_index,
        observation_text="obs",
        proposed_command="LOOK",
        validated_command="LOOK",
        agent_command="LOOK",
        result_text="ok",
        active_subgoal=ActiveSubgoal.FIND_TREASURE,
        validator_action=ValidatorAction.ACCEPTED,
        state_flags=StateFlags(has_treasure=has_treasure, at_exit=at_exit),
        terminal=terminal,
    )


def test_summarize_replay_requires_treasure_exit_and_terminal() -> None:
    seed = 17
    passing = ReplayArtifact(
        metadata=RunMetadata(
            run_id=f"eval-{seed}",
            seed=seed,
            max_turns=6,
            model_provider="test",
            model_name="test",
        ),
        turns=[
            _turn(seed=seed, turn_index=0, has_treasure=False, at_exit=False),
            _turn(seed=seed, turn_index=1, terminal=True, has_treasure=True, at_exit=True),
        ],
        hints=[],
    )
    failing = ReplayArtifact(
        metadata=RunMetadata(
            run_id=f"eval-{seed + 1}",
            seed=seed + 1,
            max_turns=6,
            model_provider="test",
            model_name="test",
        ),
        turns=[_turn(seed=seed + 1, turn_index=0, terminal=True, has_treasure=True, at_exit=False)],
        hints=[],
    )

    assert summarize_replay(passing).success is True
    assert summarize_replay(failing).success is False


def test_build_report_threshold_pass_fail() -> None:
    seeds = resolve_eval_seeds()
    outcomes = [
        SeedOutcome(
            seed=seed,
            run_id=f"eval-{seed}",
            turns=4,
            max_turns=8,
            terminal=True,
            treasure_retrieved=True,
            exit_reached=True,
            success=index < 4,
        )
        for index, seed in enumerate(seeds)
    ]
    report = build_report(seeds=seeds, outcomes=outcomes, threshold=DEFAULT_THRESHOLD)
    strict_report = build_report(seeds=seeds, outcomes=outcomes, threshold=0.81)

    assert report.success_rate == 0.8
    assert report.passed is True
    assert strict_report.passed is False


def test_fixed_seed_benchmark_uses_deterministic_seed_set() -> None:
    seen: list[int] = []

    def fake_run(seed: int, max_turns: int) -> SeedOutcome:
        seen.append(seed)
        return SeedOutcome(
            seed=seed,
            run_id=f"eval-{seed}",
            turns=max_turns,
            max_turns=max_turns,
            terminal=False,
            treasure_retrieved=False,
            exit_reached=False,
            success=False,
        )

    report = run_fixed_seed_benchmark(max_turns=3, run_seed=fake_run)

    assert seen == list(DEFAULT_EVAL_SEEDS)
    assert report.seeds == DEFAULT_EVAL_SEEDS
