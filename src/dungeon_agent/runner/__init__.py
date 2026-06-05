"""Runtime/entrypoint module scaffold."""

from dungeon_agent.runner.hint_stream import (
    deserialize_hint_events_jsonl,
    read_hint_event_stream,
    serialize_hint_events_jsonl,
    write_hint_event_stream,
)
from dungeon_agent.runner.replay_runner import ReplayCheckResult, replay_and_compare, replay_logs
from dungeon_agent.runner.turn_log import (
    deserialize_turn_records_jsonl,
    read_turn_log,
    serialize_turn_records_jsonl,
    write_turn_log,
)
from dungeon_agent.runner.human_hints import (
    DecisionContext,
    FixedIntervalCheckpointPolicy,
    HintEligibilityContext,
    HintIngestionResult,
    HumanHintIngestionService,
)
from dungeon_agent.runner.core import (
    ConsoleTurnPrinter,
    RunSummary,
    Runner,
    TurnOutput,
    compute_turn_limit,
)

__all__ = [
    "DecisionContext",
    "FixedIntervalCheckpointPolicy",
    "HintEligibilityContext",
    "HintIngestionResult",
    "HumanHintIngestionService",
    "ConsoleTurnPrinter",
    "RunSummary",
    "Runner",
    "TurnOutput",
    "compute_turn_limit",
    "deserialize_hint_events_jsonl",
    "deserialize_turn_records_jsonl",
    "replay_and_compare",
    "replay_logs",
    "ReplayCheckResult",
    "read_turn_log",
    "read_hint_event_stream",
    "serialize_hint_events_jsonl",
    "serialize_turn_records_jsonl",
    "write_hint_event_stream",
    "write_turn_log",
]
