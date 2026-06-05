"""Deterministic checkpoint-only human hint ingestion for agent turns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from dungeon_agent.agent.policy import PolicyInput
from dungeon_agent.schemas import HumanHintEvent


@dataclass(frozen=True, slots=True)
class HintEligibilityContext:
    turn_index: int
    room_id: str | None = None


class HintCheckpointPolicy(Protocol):
    def is_hint_eligible(self, context: HintEligibilityContext) -> bool:
        """Return whether the turn/room is eligible for human hints."""


@dataclass(frozen=True, slots=True)
class FixedIntervalCheckpointPolicy:
    turn_interval: int
    first_checkpoint_turn: int = 0
    eligible_room_ids: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.turn_interval <= 0:
            raise ValueError("turn_interval must be > 0")
        if self.first_checkpoint_turn < 0:
            raise ValueError("first_checkpoint_turn must be >= 0")

    def is_hint_eligible(self, context: HintEligibilityContext) -> bool:
        if context.turn_index < self.first_checkpoint_turn:
            return False
        if (context.turn_index - self.first_checkpoint_turn) % self.turn_interval != 0:
            return False
        if self.eligible_room_ids and context.room_id not in self.eligible_room_ids:
            return False
        return True


@dataclass(frozen=True, slots=True)
class HintIngestionResult:
    hint_text: str | None
    event: HumanHintEvent | None
    accepted: bool


@dataclass(frozen=True, slots=True)
class DecisionContext:
    observation_text: str
    human_input_text: str | None = None

    def to_policy_input(self) -> PolicyInput:
        return PolicyInput(
            observation_text=self.observation_text,
            human_input_text=self.human_input_text,
        )


class HumanHintIngestionService:
    """Accept hints only at deterministic checkpoints and emit separate hint events."""

    def __init__(
        self,
        *,
        mode: Literal["agent", "agent+human"] = "agent+human",
        checkpoint_policy: HintCheckpointPolicy,
    ) -> None:
        self._mode = mode
        self._checkpoint_policy = checkpoint_policy

    def ingest_hint(
        self,
        *,
        run_id: str,
        turn_index: int,
        room_id: str | None,
        hint_text: str | None,
    ) -> HintIngestionResult:
        normalized_hint = self._normalize_hint(hint_text)
        if self._mode != "agent+human" or normalized_hint is None:
            return HintIngestionResult(hint_text=None, event=None, accepted=False)

        context = HintEligibilityContext(turn_index=turn_index, room_id=room_id)
        if not self._checkpoint_policy.is_hint_eligible(context):
            return HintIngestionResult(hint_text=None, event=None, accepted=False)

        event = HumanHintEvent(run_id=run_id, turn_index=turn_index, hint_text=normalized_hint)
        return HintIngestionResult(hint_text=normalized_hint, event=event, accepted=True)

    def build_decision_context(
        self, *, observation_text: str, hint_result: HintIngestionResult
    ) -> DecisionContext:
        return DecisionContext(
            observation_text=observation_text,
            human_input_text=hint_result.hint_text,
        )

    @staticmethod
    def _normalize_hint(hint_text: str | None) -> str | None:
        if hint_text is None:
            return None
        normalized = " ".join(hint_text.split())
        return normalized or None
