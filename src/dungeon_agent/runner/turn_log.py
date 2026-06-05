"""Replay turn-log JSONL serialization utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from dungeon_agent.schemas import TurnRecord


def serialize_turn_records_jsonl(records: Iterable[TurnRecord]) -> str:
    return "\n".join(record.model_dump_json() for record in records)


def deserialize_turn_records_jsonl(payload: str) -> list[TurnRecord]:
    return [TurnRecord.model_validate(json.loads(line)) for line in payload.splitlines() if line.strip()]


def write_turn_log(path: Path, records: Iterable[TurnRecord]) -> None:
    path.write_text(serialize_turn_records_jsonl(records), encoding="utf-8")


def read_turn_log(path: Path) -> list[TurnRecord]:
    return deserialize_turn_records_jsonl(path.read_text(encoding="utf-8"))
