"""Typed contracts for the game module boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class GameState:
    turn_index: int = 0
    done: bool = False


@dataclass(frozen=True, slots=True)
class TurnOutcome:
    observation_text: str
    result_text: str
    done: bool
    consumed_turn: bool = True


class GameEngine(Protocol):
    def reset(self, *, seed: int) -> str:
        """Reset world state and return initial observation text."""

    def step(self, command: str) -> TurnOutcome:
        """Apply one validated command and return turn outcome."""

    def snapshot(self) -> GameState:
        """Return minimal serializable state for replay/debug."""
