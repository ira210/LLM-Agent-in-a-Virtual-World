"""Stable serialization schemas for turns, hints, logs, and replay."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ActiveSubgoal(str, Enum):
    FIND_LIGHT = "find_light"
    FIND_KEY = "find_key"
    FIND_TREASURE = "find_treasure"
    RETURN_TO_EXIT = "return_to_exit"


class ValidatorAction(str, Enum):
    ACCEPTED = "accepted"
    REWRITTEN = "rewritten"
    REPLAN = "replan"


class HintEventType(str, Enum):
    HUMAN_HINT = "human_hint"


class StateFlags(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    is_dark: bool = False
    has_light: bool = False
    has_treasure: bool = False
    at_exit: bool = False


class TurnRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    seed: int
    turn_index: int
    observation_text: str
    human_input_text: str | None = None
    proposed_command: str
    validated_command: str
    agent_command: str
    result_text: str
    active_subgoal: ActiveSubgoal
    loop_recovery_triggered: bool = False
    validator_action: ValidatorAction
    state_flags: StateFlags = Field(default_factory=StateFlags)
    terminal: bool = False
    consumed_turn: bool = True


class HumanHintEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    turn_index: int
    event_type: HintEventType = HintEventType.HUMAN_HINT
    hint_text: str


class HumanHintEventStream(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    events: list[HumanHintEvent] = Field(default_factory=list)


class RunMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    seed: int
    max_turns: int
    model_provider: str
    model_name: str


class ReplayArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metadata: RunMetadata
    turns: list[TurnRecord]
    hints: list[HumanHintEvent] = Field(default_factory=list)


def schema_bundle() -> dict[str, Any]:
    return {
        "turn_record": TurnRecord.model_json_schema(),
        "human_hint_event": HumanHintEvent.model_json_schema(),
        "human_hint_event_stream": HumanHintEventStream.model_json_schema(),
        "replay_artifact": ReplayArtifact.model_json_schema(),
    }
