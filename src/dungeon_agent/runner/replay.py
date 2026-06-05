"""Replay CLI for deterministic equivalence checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dungeon_agent.runner.replay_runner import replay_and_compare, replay_logs
from dungeon_agent.schemas import ReplayArtifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay and verify deterministic turn logs.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--turn-log", type=Path, help="Path to turn-log JSONL file.")
    source.add_argument(
        "--artifact",
        type=Path,
        help="Path to replay artifact JSON file containing turns + hints.",
    )
    parser.add_argument("--hint-log", type=Path, default=None, help="Path to hint event-stream JSONL file.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.artifact is not None:
        payload = json.loads(args.artifact.read_text(encoding="utf-8"))
        artifact = ReplayArtifact.model_validate(payload)
        result = replay_and_compare(turns=artifact.turns, hints=artifact.hints)
    else:
        result = replay_logs(args.turn_log, hint_log_path=args.hint_log)

    if result.success:
        print(
            f"replay-ok run_id={result.run_id} turns={result.actual_turns}/{result.expected_turns}"
        )
        return 0

    print(
        f"replay-mismatch run_id={result.run_id or '<unknown>'} "
        f"turns={result.actual_turns}/{result.expected_turns} reason={result.message}"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
