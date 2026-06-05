---
title: Design Specification for a Text-Driven 2D Dungeon LLM Agent Harness
version: 1.2
date_created: 2026-06-04
last_updated: 2026-06-04
owner: @rbubley
tags: [design, architecture, process, llm-agent, dungeon]
---

# Introduction

This specification defines the design requirements for a system with two explicit modules: a `game` module that hosts a text-driven 2D dungeon world, and an `agent` module that plays that game. The agent receives prose observations, can receive human input, issues parser-style commands, and must retrieve a treasure and exit the dungeon.

## 1. Purpose & Scope

This document specifies the functional and non-functional requirements for version 1 (v1) of the dungeon agent harness.

Scope includes:
- World model and generation constraints.
- Observation and action interfaces between environment and LLM.
- Interfaces between the `game` module and the `agent` module.
- Human-to-agent input channel requirements.
- Win conditions and run constraints.
- Logging, replay, and demo artifact requirements.

Out of scope for v1:
- Combat systems and monsters.
- Rich graphics or 3D rendering.
- Multi-agent gameplay.

Intended audience:
- Engineers implementing the harness.
- AI agents generating implementation code from this specification.

Assumptions:
- Implementation language is Python.
- Dependency and environment management uses `uv`.
- LLM provider target is OpenAI API first.

## 2. Definitions

- **LLM**: Large Language Model used for action selection.
- **Harness**: Integration layer connecting environment state/output to LLM input and parsing LLM actions.
- **Game Module**: The module that owns dungeon state, movement rules, command execution, and observation generation.
- **Agent Module**: The module that consumes observations (plus optional human input) and emits one game command per turn.
- **Human Input**: Operator-provided guidance text supplied to the agent module during a run.
- **Tool Layer**: Deterministic internal execution layer between model proposal and final emitted game command.
- **MapMemory**: Internal tool that automatically records discovered rooms, exits, and items.
- **GoalManager**: Internal tool that tracks explicit subgoals and current objective stage.
- **CommandValidator**: Internal tool that validates/normalizes proposed commands before emission.
- **LoopRecovery**: Internal tool that detects repeated command/state loops and forces a recovery strategy.
- **Room Template**: Hand-authored room content (description, objects, narrative details).
- **Procedural Layout**: Runtime arrangement of room templates into a dungeon graph/grid.
- **Observation**: Prose text returned to the agent each turn.
- **Action Command**: A single parser-style command submitted by the agent per turn.
- **Room Search**: A command that performs explicit environmental searching and is only effective in lit conditions.
- **Item Relation**: A positional or containment relationship between items (for example: inside, under, behind).
- **Darkness State**: Condition where the player lacks a light source and receives reduced situational detail.
- **Run Log**: Timestamped turn-by-turn record of observations, agent commands, and environment results.
- **Deterministic Replay**: Ability to reproduce a run using the same seed and command sequence.

## 3. Requirements, Constraints & Guidelines

- **REQ-001**: The environment shall be a 2D dungeon world represented as rooms connected by directional movement.
- **REQ-002**: Room narrative content shall be hand-authored.
- **REQ-003**: Dungeon layout shall be procedurally generated at run start.
- **REQ-004**: Typical generated runs shall contain 8 to 15 rooms.
- **REQ-005**: Every generated run shall be guaranteed solvable for the v1 objective.
- **REQ-006**: The primary objective shall be: retrieve treasure and exit the dungeon.
- **REQ-007**: Observations delivered to the agent shall be prose-only (no structured state block, no JSON contract in prompts).
- **REQ-008**: The agent shall emit exactly one command per turn.
- **REQ-009**: Accepted command style shall be classic text-adventure parser verbs with aliases.
- **REQ-010**: The parser shall accept at minimum movement, inventory, room inspection/search, and item interaction commands: `N`, `S`, `E`, `W`, `L`, `I`, `X`, `SEARCH`, `GET`, `TAKE`, `USE`, `OPEN`, `MOVE`, plus equivalent full-form verbs.
- **REQ-011**: In darkness, movement shall remain allowed.
- **REQ-012**: In darkness, observation text shall hide critical details including explicit exit listings and item details.
- **REQ-013**: The world shall include at least one reachable light source needed for reliable exploration.
- **REQ-014**: The system shall produce deterministic replay artifacts for fixed seeds.
- **REQ-015**: The repository deliverables shall include at least one run log and one terminal recording (GIF or video) demonstrating success.
- **REQ-016**: The implementation shall separate concerns into a `game` module and an `agent` module with a clear programmatic interface.
- **REQ-017**: The `agent` module shall support optional human input each turn and may use it when selecting the next command.
- **REQ-018**: The `agent` module shall implement a tool layer with the sequence: propose command -> validate/normalize -> emit final command.
- **REQ-019**: The tool layer shall include `MapMemory`, `GoalManager`, `CommandValidator`, and `LoopRecovery` in v1.
- **REQ-020**: `MapMemory` shall automatically persist discovered rooms, exits, and item observations without requiring model-authored notes.
- **REQ-021**: `GoalManager` shall track explicit subgoals in this order: find light -> find treasure -> return to exit.
- **REQ-022**: `CommandValidator` shall auto-rewrite invalid but near-valid proposals to the nearest valid alias; if rewrite is not possible, it shall trigger re-planning.
- **REQ-023**: `LoopRecovery` shall trigger when either (a) the same command is emitted for 3 consecutive turns, or (b) room oscillation is detected for 3 consecutive turns.
- **REQ-024**: Human input shall be sparse guidance and shall be logged as a separate event stream in run artifacts.
- **REQ-025**: Internal tools in v1 shall be deterministic rule-based components (no additional model calls inside tools).
- **REQ-026**: The game shall support room-level search actions, and search shall only provide meaningful discovery results when the room is lit.
- **REQ-027**: The item model shall support nested and positional relations, including at minimum: item inside item, item under item, and item behind item.
- **REQ-028**: Item interaction commands shall allow discovery/retrieval workflows across relations (for example: open container, move obstacle, then take revealed item).

- **SEC-001**: API keys shall be loaded from environment variables and shall not be embedded in source code or logs.
- **SEC-002**: Run logs shall not include secret values (keys, tokens, credentials).

- **CON-001**: v1 shall not include combat or monster mechanics.
- **CON-002**: The observation channel shall remain prose-only, even if internal state is structured in code.
- **CON-003**: Action parsing shall be deterministic and unambiguous for accepted aliases.
- **CON-004**: The model shall not bypass the tool layer to emit unvalidated commands directly to the game engine.

- **GUD-001**: Prefer concise prose descriptions that preserve key affordances relevant to planning.
- **GUD-002**: Invalid commands should produce explicit corrective feedback, not silent no-ops.
- **GUD-003**: Keep environment and harness interfaces modular so provider/model backends can be swapped later.

- **PAT-001**: Separate `game`, `agent`, `parser`, and `runner` responsibilities into distinct modules.
- **PAT-002**: Use seed-driven randomization for all layout generation affecting replay fidelity.
- **PAT-003**: Keep tool state (`MapMemory`, subgoal state, loop counters) explicit and serializable for replay/debugging.

## 4. Interfaces & Data Contracts

### 4.1 Module Boundary Contract

| Interface | Provider | Consumer | Description |
|---|---|---|---|
| `GameEngine` | `game` module | `agent` module / runner | Produces observation text, applies commands, returns turn results. |
| `AgentPolicy` | `agent` module | runner | Consumes observation text and optional human input, returns one command. |
| `HumanInputChannel` | runner/UI | `agent` module | Supplies optional operator guidance text per turn. |
| `ToolLayer` | `agent` module | `AgentPolicy` | Applies deterministic memory, goal tracking, validation, and loop recovery before command emission. |
| `MapMemory` | `ToolLayer` | `AgentPolicy`/`GoalManager` | Query/update discovered room graph and item knowledge. |
| `GoalManager` | `ToolLayer` | `AgentPolicy` | Maintains active subgoal and completion checks. |
| `CommandValidator` | `ToolLayer` | `AgentPolicy` | Validates and normalizes proposed commands to parser-accepted form. |
| `LoopRecovery` | `ToolLayer` | `AgentPolicy` | Detects repeating failures/oscillation and selects recovery action. |

### 4.2 Turn Loop Contract

| Field | Direction | Type | Description |
|---|---|---|---|
| `observation_text` | Environment -> Agent | string | Prose narrative for current turn state. |
| `human_input_text` | Human -> Agent | string or null | Optional operator guidance for current turn. |
| `proposed_command` | AgentPolicy -> ToolLayer | string | Candidate single command produced by model policy. |
| `validated_command` | ToolLayer -> Environment | string | Final parser-safe command emitted to game engine. |
| `agent_command` | Agent -> Environment | string | Backward-compatible field equal to `validated_command`. |
| `turn_result_text` | Environment -> Agent | string | Result prose after command execution. |
| `done` | Environment internal/exposed to runner | boolean | Indicates terminal state. |

### 4.3 Command Grammar (v1 Minimum)

Single-command per turn grammar:

```text
<command> := <movement> | <look> | <inventory> | <search> | <examine> | <take> | <use> | <open> | <move>
<movement> := N | S | E | W | NORTH | SOUTH | EAST | WEST | GO <direction>
<look> := L | LOOK
<inventory> := I | INVENTORY
<search> := SEARCH [ROOM | <target>]
<examine> := X <target> | EXAMINE <target>
<take> := GET <target> | TAKE <target>
<use> := USE <target> [ON <target>]
<open> := OPEN <target>
<move> := MOVE <target>
```

Item relation model (game-state contract):

```text
relation_type := IN | UNDER | BEHIND
item_location := room | inventory | relation(item_id, relation_type)
```

### 4.4 Replay Log Contract (JSON Lines recommended)

```json
{
  "run_id": "string",
  "seed": 12345,
  "turn_index": 7,
  "observation_text": "string",
  "human_input_text": "string or null",
  "proposed_command": "string",
  "validated_command": "string",
  "agent_command": "string",
  "result_text": "string",
  "active_subgoal": "find_light | find_treasure | return_to_exit",
  "loop_recovery_triggered": false,
  "validator_action": "accepted | rewritten | replan",
  "state_flags": {
    "is_dark": true,
    "has_light": false,
    "has_treasure": false,
    "at_exit": false
  },
  "terminal": false
}
```

Human hint event stream entry (separate channel):

```json
{
  "run_id": "string",
  "turn_index": 7,
  "event_type": "human_hint",
  "hint_text": "Try finding a light source before exploring deeper."
}
```

## 5. Acceptance Criteria

- **AC-001**: Given a new run with fixed seed, when generation completes, then the map contains 8-15 rooms and a valid path exists to obtain light, treasure, and exit.
- **AC-002**: Given prose-only mode, when a turn is rendered, then no JSON or structured state block is included in agent-facing observation text.
- **AC-003**: Given darkness and no light source, when `LOOK` is executed, then observation omits explicit exits and fine-grained object details.
- **AC-004**: Given darkness and no light source, when a movement command is executed, then movement is still processed if path exists.
- **AC-005**: Given valid alias commands (`N`, `L`, `GET TORCH`), when parsed, then each maps deterministically to the intended action.
- **AC-006**: Given a successful run, when objective is evaluated, then completion requires treasure possession and reaching dungeon exit.
- **AC-007**: Given the same seed and same command sequence, when replayed, then resulting state transitions and terminal outcome are identical.
- **AC-008**: Given a public repository submission, when reviewed, then it includes executable instructions, a run log, and terminal recording artifact.
- **AC-009**: Given the codebase architecture, when modules are inspected, then `game` and `agent` are separate modules with explicit interfaces.
- **AC-010**: Given human guidance is provided in a turn, when the agent selects an action, then the action-selection path can consume that guidance while still emitting exactly one valid command.
- **AC-011**: Given a model-proposed command, when processed by the tool layer, then exactly one validated parser-safe command is emitted.
- **AC-012**: Given newly discovered exits/items/rooms, when the turn commits, then `MapMemory` persists them for subsequent decisions.
- **AC-013**: Given run progression, when subgoal conditions change, then `GoalManager` transitions subgoals in order: find light -> find treasure -> return to exit.
- **AC-014**: Given an invalid but near-valid command, when validated, then `CommandValidator` rewrites it to a valid alias; otherwise it triggers re-plan.
- **AC-015**: Given repeated command or room oscillation for 3 consecutive turns, when detected, then `LoopRecovery` is triggered and recovery action is applied.
- **AC-016**: Given human hints are entered, when logs are written, then hints appear in a separate event stream and are replay-addressable.
- **AC-017**: Given deterministic seed and identical hint/command inputs, when replayed, then tool-layer decisions are identical.
- **AC-018**: Given an unlit room, when `SEARCH` is executed, then hidden details are not revealed and the result indicates insufficient light.
- **AC-019**: Given a lit room with hidden item relations, when valid interaction/search commands are executed, then items inside/under/behind other items can be discovered.
- **AC-020**: Given a relation-gated item, when prerequisite interaction occurs (`OPEN` or `MOVE` as applicable), then the gated item becomes retrievable with `GET`/`TAKE`.

## 6. Test Automation Strategy

- **Test Levels**: Unit, Integration, End-to-End.
- **Frameworks**: Python `pytest` for unit/integration; script-driven E2E harness runs.
- **Test Data Management**: Seed-controlled fixture generation for deterministic maps and known edge-state setup (darkness, invalid commands, near-exit).
- **CI/CD Integration**: Automated tests run in GitHub Actions on pull requests and main branch.
- **Coverage Requirements**: Minimum 80% coverage on world generation, parser, win-condition evaluation, and tool-layer modules.
- **Performance Testing**: Batch simulation tests (e.g., N seeded runs) must complete within defined CI time budget.

## 7. Rationale & Context

This design prioritizes harness quality over world complexity. A prose-only observation channel tests whether the harness can maintain reliable control without structured prompts. Deterministic parsing and replay are required to separate model behavior from environment bugs and to support reproducible evaluation. Limiting v1 to puzzle-navigation avoids conflating core harness validity with combat/system complexity.

The tool layer is required to improve robustness under imperfect model outputs. `MapMemory` and `GoalManager` make long-horizon behavior explicit, while `CommandValidator` and `LoopRecovery` prevent parser and local-loop failure modes from stalling runs. Deterministic, rule-based tools preserve replayability and make debugging attributable to either model policy or environment/tool logic.

## 8. Dependencies & External Integrations

### External Systems
- **EXT-001**: OpenAI-compatible chat/completions API endpoint for turn-level command generation.

### Third-Party Services
- **SVC-001**: LLM inference provider with acceptable latency for iterative turn loops.

### Infrastructure Dependencies
- **INF-001**: Local runtime supporting Python execution and environment variable configuration.

### Data Dependencies
- **DAT-001**: Hand-authored room template data bundled in repository (local files).

### Technology Platform Dependencies
- **PLT-001**: Python runtime managed with `uv` for environment/package reproducibility.

### Compliance Dependencies
- **COM-001**: Secrets management policy requiring no committed credentials and no secret leakage in logs.

## 9. Examples & Edge Cases

```text
Example observation (lit):
"You stand in an empty room. The dungeon exit lies to the south. Passages lead north, east, and west."

Example observation (dark):
"You are in darkness. Stone walls press close. You can feel open space nearby, but details are impossible to make out."

Example turn:
OBS: "You are in darkness..."
CMD: "L"
RESULT: "You peer into the dark but cannot make out exits or objects."

Search edge case:
OBS: "You are in darkness..."
CMD: "SEARCH ROOM"
RESULT: "It is too dark to search effectively."

Nested item edge case:
OBS: "A heavy crate sits against the wall."
CMD: "MOVE CRATE"
RESULT: "You drag the crate aside and reveal a small key behind it."
CMD: "TAKE KEY"
RESULT: "Taken."

Tool-layer example:
PROPOSED: "GO NORT"
VALIDATOR: rewritten -> "N"
EMITTED: "N"

Loop-recovery example:
Turns 11-13 oscillate between Room A and Room B.
LoopRecovery triggers at turn 13 and forces a non-repeating exploratory command.

Edge case:
CMD: "TAKE TREASURE" before treasure is present
RESULT: "You reach for it, but there is no treasure here."

Edge case:
CMD: "NE"
RESULT: "I do not understand that command."  (unless explicitly added to alias list)
```

## 10. Validation Criteria

- **VAL-001**: Static validation confirms command parser accepts required alias set and rejects unsupported abbreviations.
- **VAL-002**: Property/graph validation confirms solvability constraints for every generated map.
- **VAL-003**: Integration tests confirm darkness visibility restrictions while preserving movement behavior.
- **VAL-004**: End-to-end seeded run confirms successful treasure retrieval and exit.
- **VAL-005**: Replay test confirms deterministic equivalence for seed + command sequence.
- **VAL-006**: Artifact validation confirms presence of README instructions, run log, and terminal recording.
- **VAL-007**: Unit tests confirm `MapMemory` update/query correctness from observation parsing.
- **VAL-008**: Unit tests confirm `GoalManager` subgoal transitions under controlled state flags.
- **VAL-009**: Unit tests confirm `CommandValidator` accept/rewrite/replan paths.
- **VAL-010**: Unit tests confirm `LoopRecovery` triggers exactly at 3-turn thresholds for configured loop types.
- **VAL-011**: Integration tests confirm human hints are consumed by agent policy and emitted to separate hint log stream.
- **VAL-012**: Integration tests confirm `SEARCH` behavior differs correctly between lit and unlit rooms.
- **VAL-013**: Integration tests confirm retrieval paths for `IN`, `UNDER`, and `BEHIND` item relations.

## 11. Related Specifications / Further Reading

- `../BRIEF.md`
- OpenAI API reference documentation
- Python `uv` documentation
