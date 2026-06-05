"""Evaluation module public surface."""

from dungeon_agent.eval.harness import (
    DEFAULT_EVAL_SEEDS,
    DEFAULT_THRESHOLD,
    EvalReport,
    SeedOutcome,
    build_report,
    render_report,
    resolve_eval_seeds,
    run_fixed_seed_benchmark,
    run_single_seed,
    summarize_replay,
)

__all__ = [
    "DEFAULT_EVAL_SEEDS",
    "DEFAULT_THRESHOLD",
    "EvalReport",
    "SeedOutcome",
    "build_report",
    "render_report",
    "resolve_eval_seeds",
    "run_fixed_seed_benchmark",
    "run_single_seed",
    "summarize_replay",
]
