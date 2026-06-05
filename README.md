# Humanoid Software Engineering Internship - Intern Challenge: LLM Agent in a Virtual World

This demo project implements a simple text-based dungeon environment and an LLM-powered agent that can navigate it. The agent operates in three modes: fully autonomous, human-assisted, and human-only. The architecture emphasizes modularity, deterministic state transitions, and clear interfaces between the game engine and the agent policy.

## Setup (uv)

```bash
uv sync --dev
```

Set OpenAI credentials (agent modes fail fast without these):

```bash
export OPENAI_API_KEY=...
export OPENAI_PROJECT_ID=...
# optional:
export OPENAI_ORGANIZATION=...
```

## Run the 3 modes

`dungeon-agent` supports all required runner modes:

```bash
# 1) agent-only
uv run dungeon-agent --mode agent-only --seed 7 --max-turns 150

# 2) agent+human (hints accepted at checkpoints)
uv run dungeon-agent --mode agent+human --seed 7 --max-turns 150 --checkpoint-interval 1 --hint "take lamp"

# 3) human-only (pre-seeded commands)
uv run dungeon-agent --mode human-only --seed 7 --max-turns 150 --human-command "LOOK" --human-command "N"
```

## Architecture overview

- `game/`: deterministic world model + parser/executor (`GameEngine` boundary, rooms/items/state transitions).
- `agent/`: policy contract and OpenAI-backed planner implementation (`OpenAIAgentPolicy`).
- tool layer (`agent/tools.py`): deterministic map/path memory, command history, validator normalization, loop-recovery signals, and subgoal progression.
- `runner/`: orchestration loop wiring mode-specific control (agent-only, agent+human, human-only), logging, and replay artifacts.

## Replay + evaluation

Generate logs during a run:

```bash
uv run dungeon-agent --mode agent-only --seed 7 --max-turns 2 \
  --turn-log-path demo.turns.jsonl --hint-log-path demo.hints.jsonl
```

Replay determinism check:

```bash
uv run dungeon-agent-replay --turn-log demo.turns.jsonl --hint-log demo.hints.jsonl
```

Fixed-seed evaluation harness:

```bash
uv run dungeon-agent-eval --max-turns 12
```

Note: `agent-only` and `agent+human` now use the OpenAI policy and require valid OpenAI credentials.

## Manual multi-run smoke test

Run a batch of fully-agentic random-seed runs:

```bash
./scripts/agentic-smoke-runs.sh 10 150
```

- First argument: number of runs (`1..100`; hard safety cap at 100).
- Second argument: max turns per run (default `150`).

## Example snippet

```text
Starting dungeon-agent mode=agent-only model=gpt-5-mini seed=7 max_turns=2 room_count=15 dry_run=False
[turn 1/2] mode=agent-only
command: proposed='LOOK' validated='LOOK' emitted='LOOK' action=accepted
...
run-complete: mode=agent-only turns=2/2 status=turn-limit-reached
replay-ok run_id=run-7 turns=2/2
```

## Design rationale (mapped to BRIEF criteria)

- **Harness quality**: strict interfaces (`GameEngine`, `AgentPolicy`) keep world and decision logic decoupled.
- **Task-grounded behavior**: runner executes validated commands against environment state (not free-form text-only simulation).
- **Observation/action clarity**: textual observations + constrained parser verbs make state/action mapping explicit.
- **Usability/simplicity**: single CLI with three modes plus replay/eval commands for quick verification.
- **Thoughtful extensibility**: deterministic tool layer (validator, loop recovery, subgoals) supports safer future LLM policies.

## Demo artifact guidance (submission)

For challenge submission, include:

1. **Run log files** (`*.turns.jsonl`, `*.hints.jsonl`) from a deterministic seed.
2. **Terminal recording** showing one full run and replay check.

Suggested flow:

```bash
# optional: record terminal session on macOS/Linux
script demo-terminal.txt
uv run dungeon-agent --mode human-only --seed 7 --max-turns 6 --human-command "LOOK" --human-command "N" --turn-log-path demo.turns.jsonl --hint-log-path demo.hints.jsonl
uv run dungeon-agent-replay --turn-log demo.turns.jsonl --hint-log demo.hints.jsonl
exit
```

If you use a screen recorder (e.g., QuickTime, OBS, asciinema), capture the same command sequence and submit the video/cast plus JSONL logs.
