"""Typed agent policy interface and OpenAI-backed planner implementation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import os
import re
import time
from typing import Protocol

from openai import APIError, OpenAI

from dungeon_agent.agent.tools import (
    CommandReference,
    CommandHistory,
    GoalManager,
    MapMemory,
    MapObservation,
    WorldMemory,
)
from dungeon_agent.console import ui_print
from dungeon_agent.schemas import StateFlags


@dataclass(frozen=True, slots=True)
class PolicyInput:
    observation_text: str
    human_input_text: str | None = None
    loop_warning_text: str | None = None
    is_dark: bool | None = None
    has_light: bool | None = None
    use_exploration_assist: bool = False
    command_reference_text: str | None = None
    moves_remaining: int | None = None


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    proposed_command: str


@dataclass(frozen=True, slots=True)
class _ToolCall:
    name: str
    arguments_json: str
    call_id: str


class AgentPolicy(Protocol):
    def propose_command(self, payload: PolicyInput) -> str:
        """Return one command candidate for this turn."""


class NullAgentPolicy:
    """Simple placeholder policy for scaffold stages."""

    def propose_command(self, payload: PolicyInput) -> str:
        _ = payload
        return "LOOK"


class OpenAIAgentPolicy:
    """LLM planner with deterministic per-run memory helpers."""

    def __init__(
        self,
        *,
        model_name: str,
        temperature: float = 0.1,
        max_output_tokens: int = 512,
        request_timeout_seconds: int = 45,
        debug_output: bool = False,
        client: OpenAI | None = None,
    ) -> None:
        env = _load_openai_env_with_dotenv()
        api_key = env.get("OPENAI_API_KEY", "").strip()
        project = env.get("OPENAI_PROJECT_ID", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for agent-only and agent+human modes.")
        if not project:
            raise RuntimeError(
                "OPENAI_PROJECT_ID is required for agent-only and agent+human modes."
            )

        self._client = client or OpenAI(
            api_key=api_key,
            project=project,
            organization=env.get("OPENAI_ORGANIZATION", "").strip() or None,
            timeout=request_timeout_seconds,
            max_retries=1,
        )
        self._model_name = model_name
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._request_timeout_seconds = request_timeout_seconds
        self._max_retries_on_truncation = 2
        self._debug_output = debug_output
        self._planner_system_prompt = self._build_planner_system_prompt()
        self._map = MapMemory()
        self._goals = GoalManager()
        self._history = CommandHistory()
        self._world = WorldMemory()
        self._last_room_id: str | None = None
        self._last_emitted_command: str | None = None
        self._scratchpad_history: list[str] = []
        self._recent_results: deque[str] = deque(maxlen=8)
        self._recent_outcomes: deque[tuple[str, str, str]] = deque(maxlen=24)
        self._blocked_room_commands: dict[str, set[str]] = {}

    def record_turn_feedback(
        self, *, emitted_command: str, observation_text: str, result_text: str
    ) -> None:
        room_id, _, _ = _parse_observation(observation_text)
        emitted = emitted_command.strip().upper()
        if room_id:
            self._history.record(command=emitted_command, room_id=room_id)
            if (
                self._last_room_id
                and room_id != self._last_room_id
                and emitted in {"N", "S", "E", "W"}
            ):
                self._map.add_edge(self._last_room_id, emitted, room_id)
            self._last_room_id = room_id
            if emitted.startswith("SEARCH"):
                lowered_result = result_text.lower()
                if "too dark" not in lowered_result and "parser error" not in lowered_result:
                    search_target = emitted.split(" ", 1)[1] if " " in emitted else "ROOM"
                    self._world.note_search(room_id=room_id, target=search_target)
        self._last_emitted_command = emitted
        lowered = result_text.lower()
        if "locked" in lowered:
            self._world.note_locked_target(result_text)
        if "clue" in lowered or "hint" in lowered:
            self._world.add_clue(result_text)
        if lowered.startswith("you take the "):
            item_name = result_text[13:].strip().rstrip(".")
            if item_name:
                self._world.seen_inventory.add(item_name.lower())
        if lowered.startswith("you are carrying:"):
            inventory_text = result_text.split(":", 1)[1]
            parsed_items = {
                part.split("(", 1)[0].strip().lower()
                for part in inventory_text.split(",")
                if part.strip()
            }
            self._world.seen_inventory = parsed_items
        if lowered.startswith("you are carrying nothing"):
            self._world.seen_inventory.clear()
        compact_result = " ".join(result_text.split())
        if compact_result:
            self._recent_results.append(f"{emitted}: {compact_result}"[:220])
            if room_id:
                self._recent_outcomes.append((room_id, emitted, compact_result))
                if _is_persistent_non_progress_result(compact_result, command=emitted):
                    self._blocked_room_commands.setdefault(room_id, set()).add(emitted)

    def propose_command(self, payload: PolicyInput) -> str:
        room_id, exits, item_descriptors = _parse_observation(payload.observation_text)
        if room_id:
            self._map.observe(MapObservation(room_id=room_id, exits=exits, items=item_descriptors))
            self._world.observe_room_items(room_id, item_descriptors)
            self._last_room_id = room_id
        has_light = payload.has_light if payload.has_light is not None else any(
            "lantern" in item.lower() or "lamp" in item.lower() for item in self._world.seen_inventory
        )
        needs_key = _needs_key_subgoal(world_memory=self._world)
        fast_path_command = _fast_path_command(
            payload=payload,
            exits=exits,
            item_descriptors=item_descriptors,
            has_light=has_light,
        )
        if fast_path_command is not None:
            self._debug(f"fast-path command selected: {fast_path_command}")
            return fast_path_command
        if payload.is_dark is True and not has_light:
            light_command = _deterministic_light_command(
                current_room=room_id,
                item_descriptors=item_descriptors,
                map_memory=self._map,
                world_memory=self._world,
            )
            if light_command:
                self._debug(f"darkness recovery command selected: {light_command}")
                return light_command

        history = self._history.last(4)
        repeat_escape_command = _repeat_failure_escape_command(
            payload=payload,
            current_room=room_id,
            exits=exits,
            map_memory=self._map,
            history=history,
            recent_outcomes=tuple(self._recent_outcomes),
        )
        if repeat_escape_command is not None:
            self._debug(f"repeat-failure escape command selected: {repeat_escape_command}")
            return repeat_escape_command

        stagnation_command = _stagnation_escape_command(
            payload=payload,
            current_room=room_id,
            exits=exits,
            map_memory=self._map,
            history=history,
            recent_outcomes=tuple(self._recent_outcomes),
        )
        if stagnation_command is not None:
            self._debug(f"stagnation escape command selected: {stagnation_command}")
            return stagnation_command

        key_hunt_command = _key_hunt_command(
            payload=payload,
            current_room=room_id,
            exits=exits,
            item_descriptors=item_descriptors,
            map_memory=self._map,
            world_memory=self._world,
            history=history,
        )
        if key_hunt_command is not None:
            self._debug(f"key-hunt command selected: {key_hunt_command}")
            return key_hunt_command

        exploration_command = _exploration_assist_command(
            payload=payload,
            current_room=room_id,
            exits=exits,
            item_descriptors=item_descriptors,
            map_memory=self._map,
            history=history,
        )
        if exploration_command is not None:
            self._debug(f"exploration command selected: {exploration_command}")
            return exploration_command

        active_subgoal = self._goals.update(
            StateFlags(
                has_light=has_light,
                has_treasure=any("sunshard" in d.lower() for d in self._world.seen_inventory),
            ),
            needs_key=needs_key,
        )
        memory_summary = _render_memory_summary(
            current_room=room_id,
            exits=exits,
            map_memory=self._map,
            world_memory=self._world,
            history=history,
            recent_results=tuple(self._recent_results),
            active_subgoal=active_subgoal.value,
            moves_remaining=payload.moves_remaining,
        )
        hint_text = payload.human_input_text or "None"
        loop_warning_text = payload.loop_warning_text or "None"
        environment_flags = (
            f"is_dark={payload.is_dark if payload.is_dark is not None else 'unknown'}; "
            f"has_light={payload.has_light if payload.has_light is not None else 'unknown'}"
        )
        command_reference = payload.command_reference_text or CommandReference.text()
        state_payload = {
            "observation": payload.observation_text,
            "human_hint": hint_text,
            "loop_warning": loop_warning_text,
            "environment_flags": environment_flags,
            "memory_summary": memory_summary,
            "command_reference": command_reference,
        }
        user_prompt = (
            "Turn state JSON:\n"
            f"{json.dumps(state_payload, separators=(',', ':'))}\n\n"
            "Return one command now."
        )
        self._debug(
            "request start "
            f"room={room_id or 'unknown'} exits={len(exits)} items={len(item_descriptors)} "
            f"history={len(history)} prompt_chars={len(self._planner_system_prompt) + len(user_prompt)}"
        )
        tools = self._policy_tool_definitions()

        try:
            request_started = time.monotonic()
            response = self._create_response(
                model=self._model_name,
                input_payload=[
                    {"role": "system", "content": self._planner_system_prompt},
                    {"role": "user", "content": "Use the JSON state to pick one high-value next command."},
                    {"role": "user", "content": user_prompt},
                ],
                max_output_tokens=self._max_output_tokens,
                tools=tools,
            )
            self._debug(
                f"request complete in {time.monotonic() - request_started:.2f}s "
                f"tokens_limit={self._max_output_tokens}"
            )
        except APIError as exc:
            self._debug(f"request failed: {exc}")
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

        response = self._resolve_tool_calls(
            response=response,
            max_output_tokens=self._max_output_tokens,
            tools=tools,
            current_room=room_id,
            exits=exits,
            item_descriptors=item_descriptors,
            history=history,
        )
        scratchpad_note, command = _extract_scratchpad_and_command(response)
        if scratchpad_note:
            self._scratchpad_history.append(scratchpad_note)
        if not command and _response_hit_max_output_tokens(response):
            self._debug("response truncated at max_output_tokens")
        retries = 0
        retry_budget = self._max_output_tokens
        while (
            not command
            and _response_hit_max_output_tokens(response)
            and retries < self._max_retries_on_truncation
        ):
            retry_budget = min(retry_budget * 2, 1024)
            retry_started = time.monotonic()
            response = self._create_response(
                model=self._model_name,
                input_payload=[
                    {"role": "system", "content": self._planner_system_prompt},
                    {"role": "user", "content": "Use the JSON state to pick one high-value next command."},
                    {"role": "user", "content": user_prompt},
                ],
                max_output_tokens=retry_budget,
                tools=tools,
            )
            self._debug(
                f"retry {retries + 1} complete in {time.monotonic() - retry_started:.2f}s "
                f"tokens_limit={retry_budget}"
            )
            response = self._resolve_tool_calls(
                response=response,
                max_output_tokens=retry_budget,
                tools=tools,
                current_room=room_id,
                exits=exits,
                item_descriptors=item_descriptors,
                history=history,
            )
            scratchpad_note, command = _extract_scratchpad_and_command(response)
            if scratchpad_note:
                self._scratchpad_history.append(scratchpad_note)
            if not command and _response_hit_max_output_tokens(response):
                self._debug("response truncated at max_output_tokens")
            retries += 1
        if command:
            command = self._avoid_blocked_room_command(
                command=command,
                current_room=room_id,
                exits=exits,
                map_memory=self._map,
                history=history,
            )
            self._debug(f"command selected: {command}")
            return command
        if _response_hit_max_output_tokens(response):
            rescue_command = self._rescue_command_after_truncation(
                observation_text=payload.observation_text,
                memory_summary=memory_summary,
                command_reference=command_reference,
                hint_text=hint_text,
                loop_warning_text=loop_warning_text,
                environment_flags=environment_flags,
            )
            if rescue_command:
                rescue_command = self._avoid_blocked_room_command(
                    command=rescue_command,
                    current_room=room_id,
                    exits=exits,
                    map_memory=self._map,
                    history=history,
                )
                self._debug(f"rescue command selected: {rescue_command}")
                return rescue_command
        fallback = _deterministic_fallback_command(payload.observation_text)
        fallback = self._avoid_blocked_room_command(
            command=fallback,
            current_room=room_id,
            exits=exits,
            map_memory=self._map,
            history=history,
        )
        self._debug(f"fallback command selected: {fallback}")
        return fallback

    def _create_response(
        self,
        *,
        model: str,
        input_payload: list[object],
        max_output_tokens: int,
        tools: list[dict[str, object]] | None = None,
        previous_response_id: str | None = None,
    ) -> object:
        request: dict[str, object] = {
            "model": model,
            "input": input_payload,
            "max_output_tokens": max_output_tokens,
            "timeout": self._request_timeout_seconds,
        }
        if tools:
            request["tools"] = tools
        if previous_response_id:
            request["previous_response_id"] = previous_response_id
        try:
            return self._client.responses.create(
                **request,
                temperature=self._temperature,
            )
        except APIError as exc:
            message = str(exc).lower()
            if "unsupported parameter" in message and "temperature" in message:
                return self._client.responses.create(**request)
            raise

    def _build_planner_system_prompt(self) -> str:
        return (
            "You are a dungeon planner.\n"
            "Plan loop each turn:\n"
            "1) Read current observation and compact memory.\n"
            "2) Use tools only when they improve certainty.\n"
            "3) Prefer meaningful progress: prioritize exploring/mapping new rooms; treat treasure as secondary.\n"
            "4) Avoid repeated failed interactions with the same target; pivot to exploration.\n"
            "5) Never abbreviate object names (e.g. use TAKE LANTERN, not TAKE L).\n"
            "6) Use concise noun targets only; do not include full descriptive clauses from room text.\n"
            "7) If dark and you have no light, prioritize obtaining light.\n"
            "8) Output exactly one command.\n"
            "Output format (exactly):\n"
            "COMMAND: <one command>\n"
            "SCRATCHPAD: <short reasoning, one line>\n"
            "Do not output anything else."
        )

    def _debug(self, message: str) -> None:
        if not self._debug_output:
            return
        ui_print(f"[agent-debug] {message}", role="agent_debug")

    def _avoid_blocked_room_command(
        self,
        *,
        command: str,
        current_room: str | None,
        exits: tuple[str, ...],
        map_memory: MapMemory,
        history: tuple[tuple[str, str], ...],
    ) -> str:
        if current_room is None:
            return command
        normalized = " ".join(command.strip().upper().split())
        blocked = self._blocked_room_commands.get(current_room, set())
        if normalized not in blocked:
            return command
        escaped = _movement_escape_command(
            current_room=current_room,
            exits=exits,
            map_memory=map_memory,
            history=history,
        )
        self._debug(f"blocked prior non-progress command '{normalized}', pivoting to {escaped}")
        return escaped

    def _rescue_command_after_truncation(
        self,
        *,
        observation_text: str,
        memory_summary: str,
        command_reference: str,
        hint_text: str,
        loop_warning_text: str,
        environment_flags: str,
    ) -> str | None:
        self._debug("attempting command-only rescue after truncation")
        rescue_system = (
            "Return exactly one line: COMMAND: <one command>. "
            "No scratchpad, no extra text."
        )
        rescue_user = (
            f"Observation:\n{observation_text}\n\n"
            f"Human hint:\n{hint_text}\n\n"
            f"Loop warning:\n{loop_warning_text}\n\n"
            f"Environment flags:\n{environment_flags}\n\n"
            f"Memory summary:\n{memory_summary}\n\n"
            f"Command reference:\n{command_reference}\n\n"
            "Return one command now."
        )
        rescue_started = time.monotonic()
        response = self._create_response(
            model=self._model_name,
            input_payload=[
                {"role": "system", "content": rescue_system},
                {"role": "user", "content": "Use the JSON state to pick one high-value next command."},
                {"role": "user", "content": rescue_user},
            ],
            max_output_tokens=min(self._max_output_tokens, 64),
        )
        self._debug(f"rescue request complete in {time.monotonic() - rescue_started:.2f}s")
        _, command = _extract_scratchpad_and_command(response)
        return command or None

    def _resolve_tool_calls(
        self,
        *,
        response: object,
        max_output_tokens: int,
        tools: list[dict[str, object]],
        current_room: str | None,
        exits: tuple[str, ...],
        item_descriptors: tuple[str, ...],
        history: tuple[tuple[str, str], ...],
    ) -> object:
        iterations = 0
        total_tool_calls = 0
        while iterations < 6:
            tool_calls = _extract_tool_calls(response)
            if not tool_calls:
                if total_tool_calls == 0:
                    self._debug("tool calls requested: none")
                self._debug(f"tool call count: {total_tool_calls}")
                return response
            total_tool_calls += len(tool_calls)
            self._debug(f"tool calls requested: {', '.join(call.name for call in tool_calls)}")
            outputs: list[dict[str, str]] = []
            for call in tool_calls:
                output_payload = self._execute_tool_call(
                    tool_call=call,
                    current_room=current_room,
                    exits=exits,
                    item_descriptors=item_descriptors,
                    history=history,
                )
                outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(output_payload, separators=(",", ":")),
                    }
                )
                self._debug(f"tool result ready: {call.name}")
            response = self._create_response(
                model=self._model_name,
                input_payload=outputs,
                max_output_tokens=max_output_tokens,
                tools=tools,
                previous_response_id=getattr(response, "id", None),
            )
            iterations += 1
        self._debug(f"tool call count: {total_tool_calls}")
        self._debug("tool loop max iterations reached; returning latest response")
        return response

    def _execute_tool_call(
        self,
        *,
        tool_call: _ToolCall,
        current_room: str | None,
        exits: tuple[str, ...],
        item_descriptors: tuple[str, ...],
        history: tuple[tuple[str, str], ...],
    ) -> dict[str, object]:
        try:
            arguments = json.loads(tool_call.arguments_json or "{}")
        except json.JSONDecodeError:
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        if tool_call.name == "get_current_context":
            return {
                "current_room": current_room,
                "exits": list(exits),
                "visible_items": list(item_descriptors),
                "searched_here": list(self._world.searched_targets_for_room(current_room)),
            }
        if tool_call.name == "get_recent_history":
            requested_limit = arguments.get("limit", 6)
            if not isinstance(requested_limit, int):
                requested_limit = 6
            limit = min(max(requested_limit, 1), 20)
            history_slice = self._history.last(limit)
            return {
                "events": [
                    {"command": command, "room_id": room_id} for command, room_id in history_slice
                ]
            }
        if tool_call.name == "get_recent_results":
            requested_limit = arguments.get("limit", 6)
            if not isinstance(requested_limit, int):
                requested_limit = 6
            limit = min(max(requested_limit, 1), 20)
            return {"results": list(tuple(self._recent_results)[-limit:])}
        if tool_call.name == "get_frontier_guidance":
            if current_room is None:
                return {"current_room": None, "unexplored_exits": [], "path_to_frontier": []}
            return {
                "current_room": current_room,
                "unexplored_exits": list(self._map.unexplored_exits(current_room)),
                "path_to_frontier": list(self._map.path_to_nearest_frontier(current_room)),
            }
        if tool_call.name == "get_world_notes":
            return {
                "locked_targets": sorted(self._world.locked_targets),
                "recent_clues": self._world.clue_notes[-5:],
                "inventory": sorted(self._world.seen_inventory),
            }
        if tool_call.name == "get_exploration_assist":
            suggestion = _exploration_assist_command(
                payload=PolicyInput(
                    observation_text="",
                    use_exploration_assist=True,
                ),
                current_room=current_room,
                exits=exits,
                item_descriptors=item_descriptors,
                map_memory=self._map,
                history=history,
            )
            return {"suggested_command": suggestion}
        return {"error": f"unknown_tool:{tool_call.name}"}

    def _policy_tool_definitions(self) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "name": "get_current_context",
                "description": "Get current room context including exits, visible items, and searched targets.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "get_recent_history",
                "description": "Get recent command history with room IDs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 20}
                    },
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "get_recent_results",
                "description": "Get recent command outcomes to avoid repeating failed actions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 20}
                    },
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "get_frontier_guidance",
                "description": "Get unexplored exits here and path to nearest exploration frontier.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "get_world_notes",
                "description": "Get locked targets, recent clues, and known inventory notes.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "get_exploration_assist",
                "description": "Return deterministic suggested command for frontier exploration.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        ]


def _load_openai_env_with_dotenv() -> dict[str, str]:
    merged: dict[str, str] = dict(os.environ)
    dotenv_path = Path(".env")
    if not dotenv_path.exists():
        return merged
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        merged.setdefault(key, value)
    return merged


def _parse_observation(text: str) -> tuple[str | None, tuple[str, ...], tuple[str, ...]]:
    cleaned = " ".join(text.split())
    room_match = re.match(r"^([^.]+)\.", cleaned)
    room_id = room_match.group(1).strip() if room_match else None

    exits: tuple[str, ...] = tuple()
    exits_match = re.search(r"Exits lead ([^.]+)\.", cleaned, flags=re.IGNORECASE)
    if exits_match:
        exits = tuple(part.strip() for part in exits_match.group(1).split(",") if part.strip())

    items: tuple[str, ...] = tuple()
    items_match = re.search(r"You notice (.+?)\.", cleaned, flags=re.IGNORECASE)
    if items_match:
        items = tuple(part.strip() for part in items_match.group(1).split(";") if part.strip())
    return room_id, exits, items


def _render_memory_summary(
    *,
    current_room: str | None,
    exits: tuple[str, ...],
    map_memory: MapMemory,
    world_memory: WorldMemory,
    history: tuple[tuple[str, str], ...],
    recent_results: tuple[str, ...],
    active_subgoal: str,
    moves_remaining: int | None,
) -> str:
    rooms = sorted(map_memory.discovered_rooms)
    locked_targets = sorted(world_memory.locked_targets)
    clues = world_memory.clue_notes[-3:]
    searched_here = ", ".join(world_memory.searched_targets_for_room(current_room)) or "none"
    history_text = " | ".join(f"{cmd}@{room}" for cmd, room in history[-3:]) or "none"
    result_text = " | ".join(recent_results[-3:]) or "none"
    rooms_text = ",".join(rooms[:8]) if rooms else "none"
    locked_text = "; ".join(locked_targets[:3]) if locked_targets else "none"
    clues_text = "; ".join(clues) if clues else "none"
    return (
        f"current_room={current_room or 'unknown'}\n"
        f"moves_remaining={moves_remaining if moves_remaining is not None else 'unknown'}\n"
        f"exits_now={', '.join(exits) if exits else 'unknown'}\n"
        f"discovered_rooms_count={len(rooms)}\n"
        f"discovered_rooms_sample={rooms_text}\n"
        f"active_subgoal={active_subgoal}\n"
        f"locked_targets={locked_text}\n"
        f"searched_here={searched_here}\n"
        f"recent_history={history_text}\n"
        f"recent_results={result_text}\n"
        f"recent_clues={clues_text}"
    )


def _extract_scratchpad_and_command(response: object) -> tuple[str, str]:
    text = getattr(response, "output_text", None)
    candidates: list[str] = []
    if isinstance(text, str) and text.strip():
        candidates.append(text.strip())

    output_items = getattr(response, "output", None)
    if isinstance(output_items, list):
        for item in output_items:
            content_items = getattr(item, "content", None)
            if not isinstance(content_items, list):
                continue
            for content in content_items:
                maybe_text = getattr(content, "text", None)
                if isinstance(maybe_text, str) and maybe_text.strip():
                    candidates.append(maybe_text.strip())
                if isinstance(content, dict):
                    dict_text = content.get("text")
                    if isinstance(dict_text, str) and dict_text.strip():
                        candidates.append(dict_text.strip())

    for raw in candidates:
        scratchpad = ""
        command = ""
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("SCRATCHPAD:"):
                scratchpad = stripped.split(":", 1)[1].strip()
            if stripped.upper().startswith("COMMAND:"):
                command = stripped.split(":", 1)[1].strip()
        if command:
            return scratchpad, command

        first_line = raw.splitlines()[0].strip() if raw.splitlines() else ""
        if "SCRATCHPAD:" in raw.upper() and "COMMAND:" not in raw.upper():
            continue
        if first_line:
            if first_line.upper().startswith("SCRATCHPAD:"):
                continue
            return "", first_line
    return "", ""


def _extract_tool_calls(response: object) -> list[_ToolCall]:
    output_items = getattr(response, "output", None)
    if not isinstance(output_items, list):
        return []
    calls: list[_ToolCall] = []
    for item in output_items:
        if isinstance(item, dict):
            item_type = item.get("type")
            name = item.get("name")
            call_id = item.get("call_id")
            arguments = item.get("arguments")
        else:
            item_type = getattr(item, "type", None)
            name = getattr(item, "name", None)
            call_id = getattr(item, "call_id", None)
            arguments = getattr(item, "arguments", None)
        if item_type != "function_call":
            continue
        if not isinstance(name, str) or not isinstance(call_id, str):
            continue
        if not isinstance(arguments, str):
            arguments = "{}"
        calls.append(_ToolCall(name=name, arguments_json=arguments, call_id=call_id))
    return calls


def _response_hit_max_output_tokens(response: object) -> bool:
    incomplete = getattr(response, "incomplete_details", None)
    reason = getattr(incomplete, "reason", None)
    return reason == "max_output_tokens"


def _deterministic_fallback_command(observation_text: str) -> str:
    _, exits, item_descriptors = _parse_observation(observation_text)
    for descriptor in item_descriptors:
        lowered = descriptor.lower()
        if "lantern" in lowered:
            return "TAKE LANTERN"
        if "lamp" in lowered:
            return "TAKE LAMP"
    direction_map = {"north": "N", "south": "S", "east": "E", "west": "W"}
    for exit_name in exits:
        candidate = direction_map.get(exit_name.lower())
        if candidate:
            return candidate
    if item_descriptors:
        return "SEARCH ROOM"
    return "INVENTORY"


def _deterministic_light_command(
    *,
    current_room: str | None,
    item_descriptors: tuple[str, ...],
    map_memory: MapMemory,
    world_memory: WorldMemory,
) -> str | None:
    for descriptor in item_descriptors:
        lowered = descriptor.lower()
        if "lantern" in lowered:
            return "TAKE LANTERN"
        if "lamp" in lowered:
            return "TAKE LAMP"
    if current_room is None:
        return None
    for room_id, remembered_items in world_memory.seen_items_by_room.items():
        if room_id == current_room:
            continue
        has_light_item = any(
            "lantern" in descriptor.lower() or "lamp" in descriptor.lower()
            for descriptor in remembered_items
        )
        if not has_light_item:
            continue
        path = map_memory.path_to(current_room, room_id)
        if path:
            return path[0]
    return None


def _needs_key_subgoal(*, world_memory: WorldMemory) -> bool:
    if not world_memory.locked_targets:
        return False
    return not any("key" in item_name for item_name in world_memory.seen_inventory)


def _key_hunt_command(
    *,
    payload: PolicyInput,
    current_room: str | None,
    exits: tuple[str, ...],
    item_descriptors: tuple[str, ...],
    map_memory: MapMemory,
    world_memory: WorldMemory,
    history: tuple[tuple[str, str], ...],
) -> str | None:
    if payload.human_input_text or payload.loop_warning_text:
        return None
    if not payload.use_exploration_assist:
        return None
    if current_room is None:
        return None
    if not _needs_key_subgoal(world_memory=world_memory):
        return None

    for descriptor in item_descriptors:
        lowered = descriptor.lower()
        if " key" in lowered or lowered.endswith("key"):
            if "coffer key" in lowered:
                return "TAKE COFFER KEY"
            if "vault key" in lowered:
                return "TAKE VAULT KEY"
            return "TAKE KEY"
    if "ROOM" not in world_memory.searched_targets_for_room(current_room):
        return "SEARCH ROOM"
    for room_id, remembered_items in world_memory.seen_items_by_room.items():
        if room_id == current_room:
            continue
        if not any(" key" in descriptor.lower() or descriptor.lower().endswith("key") for descriptor in remembered_items):
            continue
        path = map_memory.path_to(current_room, room_id)
        if path:
            return path[0]
    return _movement_escape_command(
        current_room=current_room,
        exits=exits,
        map_memory=map_memory,
        history=history,
    )


def _fast_path_command(
    *,
    payload: PolicyInput,
    exits: tuple[str, ...],
    item_descriptors: tuple[str, ...],
    has_light: bool,
) -> str | None:
    if payload.human_input_text or payload.loop_warning_text:
        return None
    if not has_light:
        for descriptor in item_descriptors:
            lowered = descriptor.lower()
            if "lantern" in lowered:
                return "TAKE LANTERN"
            if "lamp" in lowered:
                return "TAKE LAMP"
    lowered_observation = payload.observation_text.lower()
    likely_interactable_keywords = (
        " door",
        " gate",
        " chest",
        " lever",
        " switch",
        " pedestal",
        " altar",
        " well",
        " winch",
        " lock",
        " key",
    )
    if any(keyword in lowered_observation for keyword in likely_interactable_keywords):
        return None
    if len(exits) == 1 and not item_descriptors:
        direction_map = {"north": "N", "south": "S", "east": "E", "west": "W"}
        return direction_map.get(exits[0].lower())
    return None


def _exploration_assist_command(
    *,
    payload: PolicyInput,
    current_room: str | None,
    exits: tuple[str, ...],
    item_descriptors: tuple[str, ...],
    map_memory: MapMemory,
    history: tuple[tuple[str, str], ...],
) -> str | None:
    if payload.human_input_text or payload.loop_warning_text:
        return None
    if not payload.use_exploration_assist:
        return None
    if current_room is None or item_descriptors:
        return None

    unexplored_here = list(map_memory.unexplored_exits(current_room))
    if unexplored_here:
        return _prefer_non_backtracking_direction(unexplored_here, history)

    path_to_frontier = map_memory.path_to_nearest_frontier(current_room)
    if path_to_frontier:
        return path_to_frontier[0]
    if exits:
        direction_map = {"north": "N", "south": "S", "east": "E", "west": "W"}
        direct_exits = [direction_map[exit_name.lower()] for exit_name in exits if exit_name.lower() in direction_map]
        if direct_exits:
            return _prefer_non_backtracking_direction(direct_exits, history)
    return None


def _stagnation_escape_command(
    *,
    payload: PolicyInput,
    current_room: str | None,
    exits: tuple[str, ...],
    map_memory: MapMemory,
    history: tuple[tuple[str, str], ...],
    recent_outcomes: tuple[tuple[str, str, str], ...],
) -> str | None:
    if payload.human_input_text or payload.loop_warning_text:
        return None
    if not payload.use_exploration_assist:
        return None
    if current_room is None:
        return None

    same_room = [outcome for outcome in recent_outcomes if outcome[0] == current_room]
    if len(same_room) < 4:
        return None
    failed_attempts = [
        (command, result_text)
        for _, command, result_text in same_room[-8:]
        if _is_non_progress_result(result_text)
    ]
    if len(failed_attempts) < 2:
        return None

    target_tokens: dict[str, int] = {}
    for command, _ in failed_attempts:
        for token in _command_target_tokens(command):
            target_tokens[token] = target_tokens.get(token, 0) + 1
    if not any(count >= 2 for count in target_tokens.values()):
        return None

    unexplored_here = list(map_memory.unexplored_exits(current_room))
    if unexplored_here:
        return _prefer_non_backtracking_direction(unexplored_here, history)
    path_to_frontier = map_memory.path_to_nearest_frontier(current_room)
    if path_to_frontier:
        return path_to_frontier[0]
    if exits:
        direction_map = {"north": "N", "south": "S", "east": "E", "west": "W"}
        direct_exits = [direction_map[exit_name.lower()] for exit_name in exits if exit_name.lower() in direction_map]
        if direct_exits:
            return _prefer_non_backtracking_direction(direct_exits, history)
    return None


def _repeat_failure_escape_command(
    *,
    payload: PolicyInput,
    current_room: str | None,
    exits: tuple[str, ...],
    map_memory: MapMemory,
    history: tuple[tuple[str, str], ...],
    recent_outcomes: tuple[tuple[str, str, str], ...],
) -> str | None:
    if payload.human_input_text or payload.loop_warning_text:
        return None
    if not payload.use_exploration_assist:
        return None
    if current_room is None:
        return None
    if len(recent_outcomes) < 1 or not history:
        return None

    last_room, last_command, last_result = recent_outcomes[-1]
    if last_room != current_room:
        return None
    if last_command != history[-1][0]:
        return None
    if last_command.startswith("EXAMINE "):
        repeated_examine = sum(
            1
            for room_id, command, _ in recent_outcomes[-6:]
            if room_id == current_room and command == last_command
        )
        if repeated_examine >= 2:
            return _movement_escape_command(
                current_room=current_room,
                exits=exits,
                map_memory=map_memory,
                history=history,
            )
    if not _is_non_progress_result(last_result):
        return None

    return _movement_escape_command(
        current_room=current_room,
        exits=exits,
        map_memory=map_memory,
        history=history,
    )


def _movement_escape_command(
    *,
    current_room: str,
    exits: tuple[str, ...],
    map_memory: MapMemory,
    history: tuple[tuple[str, str], ...],
) -> str:
    unexplored_here = list(map_memory.unexplored_exits(current_room))
    if unexplored_here:
        return _prefer_non_backtracking_direction(unexplored_here, history)
    path_to_frontier = map_memory.path_to_nearest_frontier(current_room)
    if path_to_frontier:
        return path_to_frontier[0]
    if exits:
        direction_map = {"north": "N", "south": "S", "east": "E", "west": "W"}
        direct_exits = [direction_map[exit_name.lower()] for exit_name in exits if exit_name.lower() in direction_map]
        if direct_exits:
            return _prefer_non_backtracking_direction(direct_exits, history)
    return "SEARCH ROOM"


def _is_non_progress_result(result_text: str) -> bool:
    lowered = result_text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "find nothing",
            "nothing unusual",
            "cannot",
            "can't",
            "do not see",
            "does not fit",
            "need to hold",
            "parser error",
            "already moved",
            "already open",
            "already wearing",
        )
    )


def _is_persistent_non_progress_result(result_text: str, *, command: str) -> bool:
    lowered = result_text.lower()
    if any(phrase in lowered for phrase in ("already moved", "already open", "already wearing")):
        return True
    if command.startswith("SEARCH ") and any(phrase in lowered for phrase in ("find nothing", "nothing unusual")):
        return True
    if command.startswith("EXAMINE ") and "part of the scenery" in lowered:
        return True
    return False


def _command_target_tokens(command: str) -> tuple[str, ...]:
    tokens = [token for token in re.findall(r"[A-Z0-9']+", command.upper()) if token]
    if not tokens:
        return tuple()
    ignored = {
        "N",
        "S",
        "E",
        "W",
        "NORTH",
        "SOUTH",
        "EAST",
        "WEST",
        "GO",
        "LOOK",
        "L",
        "INVENTORY",
        "I",
        "SEARCH",
        "EXAMINE",
        "X",
        "TAKE",
        "GET",
        "MOVE",
        "OPEN",
        "USE",
        "WEAR",
        "ON",
        "THE",
        "A",
        "AN",
    }
    return tuple(token for token in tokens if token not in ignored and len(token) >= 3)


def _prefer_non_backtracking_direction(
    directions: list[str], history: tuple[tuple[str, str], ...]
) -> str:
    if not directions:
        return "N"
    reverse = {"N": "S", "S": "N", "E": "W", "W": "E"}
    last_command = history[-1][0] if history else None
    for direction in sorted(directions):
        if last_command in reverse and direction == reverse[last_command]:
            continue
        return direction
    return sorted(directions)[0]
