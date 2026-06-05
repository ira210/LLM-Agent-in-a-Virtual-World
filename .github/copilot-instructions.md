# Copilot Instructions for `dungeon-agent`

## Build, test, and lint

This repository uses `uv` (Python 3.14 target).

```bash
# install deps
uv sync --dev

# run full test suite
uv run pytest -q

# run a single test file
uv run pytest -q tests/test_agent_policy.py

# run a single test case
uv run pytest -q tests/test_agent_policy.py::test_openai_policy_proposes_single_command

# lint
uv run ruff check .
```

Common runner commands:

```bash
# main runner
uv run dungeon-agent --mode agent-only --seed 7 --max-turns 150

# replay check
uv run dungeon-agent-replay --turn-log demo.turns.jsonl --hint-log demo.hints.jsonl

# fixed-seed eval harness
uv run dungeon-agent-eval --max-turns 12
```

For `agent-only` and `agent+human`, set:

```bash
OPENAI_API_KEY=...
OPENAI_PROJECT_ID=...
```

## High-level architecture

- `src/dungeon_agent/game/*` is a deterministic world simulator:
  - `domain.py` models rooms/items/player/world state.
  - `parser_executor.py` parses commands and mutates world state.
  - `observation.py` renders room/inventory text seen by the agent.
- `src/dungeon_agent/agent/*` is the decision layer:
  - `policy.py` builds `PolicyInput`, combines deterministic fast paths + memory-assisted exploration + LLM planning.
  - `tools.py` holds deterministic helpers: command validation, loop recovery, map memory graph/pathing, world memory, goals.
- `src/dungeon_agent/runner/*` orchestrates mode-specific turn loops:
  - `core.py` drives each turn, applies command validation/recovery, writes `TurnRecord`s, and integrates human hints.
  - `human_hints.py` gates hints via checkpoint policy.
  - `turn_log.py`, `hint_stream.py`, `replay_runner.py`, and `replay.py` support deterministic replay verification.
- `src/dungeon_agent/eval/harness.py` runs fixed-seed benchmark scenarios and computes pass/fail against a threshold.

The key integration boundary is: `Runner` ⇄ `AgentPolicy` ⇄ `ParserExecutorEngine` with `schemas.py` (`TurnRecord`, `HumanHintEvent`, `StateFlags`) as stable log/replay contracts.

## Key conventions in this codebase

- Keep world transitions deterministic and explicit. Parser/executor paths return concrete turn outcomes; avoid hidden side effects.
- Normalize commands through the validator/tool layer before execution (`CommandValidator`, loop recovery, rewrite behavior in `Runner`).
- Prefer deterministic assists for high-confidence choices (e.g., light acquisition, frontier exploration, loop recovery) and defer to LLM for ambiguous turns.
- Preserve replay compatibility:
  - `TurnRecord` and `HumanHintEvent` schemas are strict (`pydantic` models with `extra="forbid", frozen=True`).
  - If turn semantics change, update replay/eval tests and maintain deterministic behavior for seeded runs.
- Agent+human behavior is checkpoint-driven via `HumanHintIngestionService` + `FixedIntervalCheckpointPolicy`; don’t bypass this path when adding hint logic.
- Console output is centralized through `src/dungeon_agent/console.py` (Rich). Route new user/debug/status output through `ui_print`/`ui_prompt` rather than raw `print`/`input` in runner/policy paths.
