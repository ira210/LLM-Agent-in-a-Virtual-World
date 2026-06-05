"""Separate human hint event stream serialization utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from dungeon_agent.schemas import HumanHintEvent, HumanHintEventStream


def serialize_hint_events_jsonl(events: Iterable[HumanHintEvent]) -> str:
    return "\n".join(event.model_dump_json() for event in events)


def deserialize_hint_events_jsonl(payload: str) -> list[HumanHintEvent]:
    return [
        HumanHintEvent.model_validate(json.loads(line))
        for line in payload.splitlines()
        if line.strip()
    ]


def write_hint_event_stream(path: Path, stream: HumanHintEventStream) -> None:
    path.write_text(serialize_hint_events_jsonl(stream.events), encoding="utf-8")


def read_hint_event_stream(path: Path, *, run_id: str) -> HumanHintEventStream:
    content = path.read_text(encoding="utf-8")
    events = deserialize_hint_events_jsonl(content)
    return HumanHintEventStream(run_id=run_id, events=events)
