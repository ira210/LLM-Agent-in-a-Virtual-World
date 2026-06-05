"""Typed runtime settings and environment loading."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from random import SystemRandom
from typing import Mapping


def _env_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw!r}")


def _env_int(raw: str | None, default: int | None) -> int | None:
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    provider: str = "openai"
    model_name: str = "gpt-5.4-nano"
    temperature: float = 0.1
    max_output_tokens: int = 512
    request_timeout_seconds: int = 45


@dataclass(frozen=True, slots=True)
class RuntimeFlags:
    seed: int | None = None
    max_turns: int = 200
    allow_human_hints: bool = True
    enable_replay_log: bool = True
    dry_run: bool = False
    debug_output: bool = False

    def resolved_seed(self) -> int:
        if self.seed is not None:
            return self.seed
        return SystemRandom().randrange(0, 2**32)


@dataclass(frozen=True, slots=True)
class Settings:
    model: ModelConfig = ModelConfig()
    runtime: RuntimeFlags = RuntimeFlags()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        source = env or os.environ
        model_defaults = ModelConfig()
        runtime_defaults = RuntimeFlags()
        model = ModelConfig(
            provider=source.get("DUNGEON_AGENT_MODEL_PROVIDER", model_defaults.provider),
            model_name=source.get("DUNGEON_AGENT_MODEL_NAME", model_defaults.model_name),
            temperature=float(source.get("DUNGEON_AGENT_MODEL_TEMPERATURE", model_defaults.temperature)),
            max_output_tokens=int(
                source.get("DUNGEON_AGENT_MODEL_MAX_OUTPUT_TOKENS", model_defaults.max_output_tokens)
            ),
            request_timeout_seconds=int(
                source.get(
                    "DUNGEON_AGENT_MODEL_TIMEOUT_SECONDS", model_defaults.request_timeout_seconds
                )
            ),
        )
        runtime = RuntimeFlags(
            seed=_env_int(source.get("DUNGEON_AGENT_SEED"), runtime_defaults.seed),
            max_turns=int(source.get("DUNGEON_AGENT_MAX_TURNS", runtime_defaults.max_turns)),
            allow_human_hints=_env_bool(
                source.get("DUNGEON_AGENT_ALLOW_HUMAN_HINTS"), runtime_defaults.allow_human_hints
            ),
            enable_replay_log=_env_bool(
                source.get("DUNGEON_AGENT_ENABLE_REPLAY_LOG"), runtime_defaults.enable_replay_log
            ),
            dry_run=_env_bool(source.get("DUNGEON_AGENT_DRY_RUN"), runtime_defaults.dry_run),
            debug_output=_env_bool(source.get("DUNGEON_AGENT_DEBUG_OUTPUT"), runtime_defaults.debug_output),
        )
        return cls(model=model, runtime=runtime)


def load_settings_with_overrides(
    *,
    env: Mapping[str, str] | None = None,
    seed: int | None = None,
    max_turns: int | None = None,
    model_name: str | None = None,
    dry_run: bool | None = None,
    debug_output: bool | None = None,
) -> Settings:
    settings = Settings.from_env(env=env)
    model = settings.model if model_name is None else replace(settings.model, model_name=model_name)
    runtime = settings.runtime
    if seed is not None:
        runtime = replace(runtime, seed=seed)
    if max_turns is not None:
        runtime = replace(runtime, max_turns=max_turns)
    if dry_run is not None:
        runtime = replace(runtime, dry_run=dry_run)
    if debug_output is not None:
        runtime = replace(runtime, debug_output=debug_output)
    return replace(settings, model=model, runtime=runtime)
