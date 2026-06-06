from __future__ import annotations

from pathlib import Path

import pytest

from dungeon_agent.agent.policy import OpenAIAgentPolicy, PolicyInput


class _FakeResponse:
    def __init__(
        self,
        output_text: str,
        *,
        output: list[dict[str, object]] | None = None,
        incomplete_reason: str | None = None,
        response_id: str | None = None,
    ) -> None:
        self.output_text = output_text
        self.output = output or []
        self.id = response_id
        if incomplete_reason is not None:
            self.incomplete_details = type("Incomplete", (), {"reason": incomplete_reason})()
        else:
            self.incomplete_details = None


class _FakeResponsesAPI:
    def __init__(self, output_text: str | list[_FakeResponse]) -> None:
        self._output_text = output_text
        self.calls = 0
        self.calls_kwargs: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _FakeResponse:
        self.calls += 1
        self.calls_kwargs.append(kwargs)
        if isinstance(self._output_text, list):
            index = min(self.calls - 1, len(self._output_text) - 1)
            return self._output_text[index]
        return _FakeResponse(self._output_text)


class _FakeClient:
    def __init__(self, output_text: str | list[_FakeResponse] = "GO NORTH") -> None:
        self.responses = _FakeResponsesAPI(output_text)


def test_openai_policy_fails_fast_without_key_or_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_PROJECT_ID", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIAgentPolicy(model_name="gpt-5-mini", client=_FakeClient())

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(RuntimeError, match="OPENAI_PROJECT_ID"):
        OpenAIAgentPolicy(model_name="gpt-5-mini", client=_FakeClient())


def test_openai_policy_proposes_single_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=_FakeClient("GO NORTH"))
    command = policy.propose_command(
        PolicyInput(
            observation_text="Entry Vestibule. Exits lead north, east.",
            human_input_text=None,
            command_reference_text="N|S|E|W",
        )
    )

    assert command == "GO NORTH"


def test_openai_policy_extracts_command_from_scratchpad_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    policy = OpenAIAgentPolicy(
        model_name="gpt-5-mini",
        client=_FakeClient("SCRATCHPAD: move toward light\nCOMMAND: TAKE LANTERN"),
    )
    command = policy.propose_command(
        PolicyInput(
            observation_text=(
                "Entry Vestibule. Exits lead north, east. "
                "You notice a brass lantern hangs above the bench."
            ),
            human_input_text=None,
            command_reference_text="N|S|E|W",
        )
    )
    assert command == "TAKE LANTERN"


def test_openai_policy_ignores_scratchpad_only_output_and_uses_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    policy = OpenAIAgentPolicy(
        model_name="gpt-5-mini",
        client=_FakeClient("SCRATCHPAD: check exits and avoid repeating look"),
    )
    command = policy.propose_command(
        PolicyInput(
            observation_text="Entry Vestibule. Exits lead north, east.",
            human_input_text=None,
            command_reference_text="N|S|E|W",
        )
    )
    assert command == "N"


def test_openai_policy_falls_back_to_look_when_no_command_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=_FakeClient(""))
    command = policy.propose_command(
        PolicyInput(
            observation_text="Entry Vestibule. Exits lead north, east.",
            human_input_text=None,
            command_reference_text="N|S|E|W",
        )
    )
    assert command == "N"


def test_openai_policy_retries_when_initial_response_hits_max_tokens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    client = _FakeClient(
        [
            _FakeResponse("", incomplete_reason="max_output_tokens"),
            _FakeResponse("TAKE LANTERN"),
        ]
    )
    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=client)
    command = policy.propose_command(
        PolicyInput(
            observation_text="Entry Vestibule. Exits lead north, east.",
            human_input_text=None,
            command_reference_text="N|S|E|W",
        )
    )
    assert command == "TAKE LANTERN"


def test_openai_policy_extracts_command_before_scratchpad_when_truncated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    client = _FakeClient([_FakeResponse("COMMAND: E", incomplete_reason="max_output_tokens")])
    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=client)
    command = policy.propose_command(
        PolicyInput(
            observation_text="Entry Vestibule. Exits lead north, east.",
            command_reference_text="N|S|E|W",
        )
    )

    assert command == "E"
    assert client.responses.calls == 1


def test_openai_policy_uses_rescue_request_before_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    client = _FakeClient(
        [
            _FakeResponse("", incomplete_reason="max_output_tokens"),
            _FakeResponse("", incomplete_reason="max_output_tokens"),
            _FakeResponse("", incomplete_reason="max_output_tokens"),
            _FakeResponse("COMMAND: W"),
        ]
    )
    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=client)
    command = policy.propose_command(
        PolicyInput(
            observation_text="Entry Vestibule. Exits lead north, east.",
            command_reference_text="N|S|E|W",
        )
    )

    assert command == "W"
    assert client.responses.calls == 4


def test_openai_policy_inventory_updates_only_on_successful_take(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=_FakeClient("LOOK"))
    policy.record_turn_feedback(
        emitted_command="TAKE MASON'S HAMMER",
        observation_text="Entry Vestibule. Exits lead north, east.",
        result_text="You do not see 'mason's hammer' to take.",
    )
    assert "mason's hammer" not in policy._world.seen_inventory

    policy.record_turn_feedback(
        emitted_command="TAKE MASON'S HAMMER",
        observation_text="Entry Vestibule. Exits lead north, east.",
        result_text="You take the Mason's Hammer.",
    )
    assert "mason's hammer" in policy._world.seen_inventory


def test_openai_policy_records_search_memory_except_dark_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=_FakeClient("LOOK"))
    policy.record_turn_feedback(
        emitted_command="SEARCH ROOM",
        observation_text="Entry Vestibule. Exits lead north, east.",
        result_text="You search the area carefully but find nothing new.",
    )
    assert policy._world.searched_targets_for_room("Entry Vestibule") == ("ROOM",)

    policy.record_turn_feedback(
        emitted_command="SEARCH ROOM",
        observation_text="Entry Vestibule. Exits lead north, east.",
        result_text="It is too dark to search effectively.",
    )
    assert policy._world.searched_targets_for_room("Entry Vestibule") == ("ROOM",)


def test_openai_policy_fallback_prefers_taking_visible_light(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=_FakeClient(""))
    command = policy.propose_command(
        PolicyInput(
            observation_text=(
                "Entry Vestibule. Exits lead north, east. "
                "You notice a brass lantern hangs above the bench."
            ),
            human_input_text=None,
            command_reference_text="N|S|E|W",
        )
    )
    assert command == "TAKE LANTERN"


def test_openai_policy_prioritizes_light_when_dark_without_light(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=_FakeClient("LOOK"))
    command = policy.propose_command(
        PolicyInput(
            observation_text=(
                "Flooded Antechamber. Exits lead north. "
                "You notice a brass lantern hangs above the bench."
            ),
            is_dark=True,
            has_light=False,
            command_reference_text="N|S|E|W",
        )
    )
    assert command == "TAKE LANTERN"


def test_openai_policy_fast_path_uses_single_exit_without_api_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    client = _FakeClient("LOOK")
    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=client)
    command = policy.propose_command(
        PolicyInput(
            observation_text="Flooded Passage. Exits lead north.",
            human_input_text=None,
            command_reference_text="N|S|E|W",
        )
    )

    assert command == "N"
    assert client.responses.calls == 0


def test_openai_policy_single_exit_fast_path_skips_when_interactable_feature_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    client = _FakeClient("COMMAND: OPEN BRONZE DOOR")
    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=client)
    command = policy.propose_command(
        PolicyInput(
            observation_text=(
                "Vault Antechamber. A circular room of polished basalt channels every sound "
                "toward a sealed bronze door. Exits lead north."
            ),
            command_reference_text="N|S|E|W",
        )
    )

    assert command == "OPEN BRONZE DOOR"
    assert client.responses.calls == 1


def test_openai_policy_fast_path_takes_visible_light_without_api_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    client = _FakeClient("LOOK")
    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=client)
    command = policy.propose_command(
        PolicyInput(
            observation_text=(
                "Flooded Antechamber. Exits lead north. "
                "You notice a brass lantern hangs above the bench."
            ),
            command_reference_text="N|S|E|W",
        )
    )

    assert command == "TAKE LANTERN"
    assert client.responses.calls == 0


def test_openai_policy_debug_output_emits_timing_lines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    policy = OpenAIAgentPolicy(
        model_name="gpt-5-mini",
        client=_FakeClient("GO NORTH"),
        debug_output=True,
    )
    _ = policy.propose_command(
        PolicyInput(
            observation_text="Entry Vestibule. Exits lead north, east.",
            command_reference_text="N|S|E|W",
        )
    )
    captured = capsys.readouterr()
    assert "[agent-debug] request start" in captured.out
    assert "[agent-debug] request complete" in captured.out
    assert "[agent-debug] tool calls requested: none" in captured.out
    assert "[agent-debug] command selected: GO NORTH" in captured.out


def test_openai_policy_includes_recent_results_in_memory_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    client = _FakeClient("COMMAND: E")
    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=client)
    policy.record_turn_feedback(
        emitted_command="USE DAMP ROPE COIL ON BUCKET",
        observation_text="Well of Whispers. Exits lead east.",
        result_text="You need to hold 'damp rope coil' before using it.",
    )
    _ = policy.propose_command(
        PolicyInput(
            observation_text="Well of Whispers. Exits lead east, west.",
            command_reference_text="N|S|E|W",
        )
    )

    first_call_input = client.responses.calls_kwargs[0]["input"]
    assert isinstance(first_call_input, list)
    user_prompt = first_call_input[2]["content"]
    assert "recent_results=USE DAMP ROPE COIL ON BUCKET: You need to hold 'damp rope coil' before using it." in user_prompt


def test_openai_policy_executes_openai_function_call_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    client = _FakeClient(
        [
            _FakeResponse(
                "",
                output=[
                    {
                        "type": "function_call",
                        "name": "get_frontier_guidance",
                        "call_id": "call_1",
                        "arguments": "{}",
                    }
                ],
                response_id="resp_1",
            ),
            _FakeResponse("COMMAND: E"),
        ]
    )
    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=client)
    command = policy.propose_command(
        PolicyInput(
            observation_text="Entry Vestibule. Exits lead north, east.",
            command_reference_text="N|S|E|W",
        )
    )

    assert command == "E"
    assert client.responses.calls == 2
    assert "tools" in client.responses.calls_kwargs[0]
    assert client.responses.calls_kwargs[1]["previous_response_id"] == "resp_1"


def test_openai_policy_explores_unvisited_exit_without_api_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    client = _FakeClient("LOOK")
    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=client)
    command = policy.propose_command(
        PolicyInput(
            observation_text="Hall. Exits lead north, east.",
            use_exploration_assist=True,
            command_reference_text="N|S|E|W",
        )
    )

    assert command == "E"
    assert client.responses.calls == 0


def test_openai_policy_escapes_room_after_repeated_failed_item_fiddling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    client = _FakeClient("COMMAND: SEARCH WELL")
    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=client)
    for command in (
        "USE ROPE COIL ON DENTED BUCKET",
        "USE ROPE COIL ON BUCKET",
        "SEARCH WELL",
        "USE ROPE COIL ON WELL",
    ):
        policy.record_turn_feedback(
            emitted_command=command,
            observation_text=(
                "Well of Whispers. Exits lead north, east, west. "
                "You notice a dented bucket hangs over the dry shaft."
            ),
            result_text="You find nothing unusual about the well bucket.",
        )

    proposed = policy.propose_command(
        PolicyInput(
            observation_text=(
                "Well of Whispers. Exits lead north, east, west. "
                "You notice a dented bucket hangs over the dry shaft."
            ),
            use_exploration_assist=True,
            command_reference_text="N|S|E|W",
        )
    )

    assert proposed == "SEARCH WELL"
    assert client.responses.calls == 1
    user_prompt = client.responses.calls_kwargs[0]["input"][2]["content"]
    assert "Avoid repeating known failed commands in this room" in user_prompt


def test_openai_policy_immediately_pivots_after_already_moved_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    client = _FakeClient("COMMAND: MOVE CHALK STUB")
    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=client)
    policy.record_turn_feedback(
        emitted_command="MOVE CHALK STUB",
        observation_text=(
            "Entry Vestibule. Exits lead north, south, east. "
            "You notice a snapped chalk stub lies near the doorway."
        ),
        result_text="You have already moved the chalk stub.",
    )

    proposed = policy.propose_command(
        PolicyInput(
            observation_text=(
                "Entry Vestibule. Exits lead north, south, east. "
                "You notice a snapped chalk stub lies near the doorway."
            ),
            use_exploration_assist=True,
            command_reference_text="N|S|E|W",
        )
    )

    assert proposed == "MOVE CHALK STUB"
    assert client.responses.calls == 1
    user_prompt = client.responses.calls_kwargs[0]["input"][2]["content"]
    assert "Avoid repeating known failed commands in this room: MOVE CHALK STUB." in user_prompt


def test_openai_policy_unblocks_open_after_successful_key_use(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    client = _FakeClient("COMMAND: OPEN PRAYER CHEST")
    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=client)
    policy.record_turn_feedback(
        emitted_command="OPEN PRAYER CHEST",
        observation_text="Sanctum. Exits lead north, east.",
        result_text="The prayer chest is locked.",
    )
    policy.record_turn_feedback(
        emitted_command="USE VAULT KEY ON PRAYER CHEST",
        observation_text="Sanctum. Exits lead north, east.",
        result_text="You unlock the prayer chest with the vault key.",
    )
    proposed = policy.propose_command(
        PolicyInput(
            observation_text=(
                "Sanctum. Exits lead north, east. "
                "You notice a prayer chest rests beneath the winch."
            ),
            use_exploration_assist=True,
            command_reference_text="N|S|E|W",
        )
    )

    assert proposed == "OPEN PRAYER CHEST"
    assert client.responses.calls == 1


def test_openai_policy_pivots_after_repeated_examine_same_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    client = _FakeClient("COMMAND: EXAMINE CHALK STUB")
    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=client)
    for _ in range(2):
        policy.record_turn_feedback(
            emitted_command="EXAMINE CHALK STUB",
            observation_text=(
                "Entry Vestibule. Exits lead north, south, east. "
                "You notice a snapped chalk stub lies near the doorway."
            ),
            result_text="A snapped chalk stub worn smooth by handling.",
        )

    proposed = policy.propose_command(
        PolicyInput(
            observation_text=(
                "Entry Vestibule. Exits lead north, south, east. "
                "You notice a snapped chalk stub lies near the doorway."
            ),
            use_exploration_assist=True,
            command_reference_text="N|S|E|W",
        )
    )

    assert proposed == "EXAMINE CHALK STUB"
    assert client.responses.calls == 1
    user_prompt = client.responses.calls_kwargs[0]["input"][2]["content"]
    assert "Recent repetition: 'EXAMINE CHALK STUB' was already used in this room." in user_prompt


def test_openai_policy_warns_on_immediate_duplicate_examine_in_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    client = _FakeClient("COMMAND: EXAMINE CHALK STUB")
    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=client)
    policy.record_turn_feedback(
        emitted_command="EXAMINE CHALK STUB",
        observation_text=(
            "Entry Vestibule. Exits lead north, south, east. "
            "You notice a snapped chalk stub lies near the doorway."
        ),
        result_text="A snapped chalk stub worn smooth by handling.",
    )

    proposed = policy.propose_command(
        PolicyInput(
            observation_text=(
                "Entry Vestibule. Exits lead north, south, east. "
                "You notice a snapped chalk stub lies near the doorway."
            ),
            use_exploration_assist=True,
            command_reference_text="N|S|E|W",
        )
    )

    first_call_input = client.responses.calls_kwargs[0]["input"]
    assert isinstance(first_call_input, list)
    user_prompt = first_call_input[2]["content"]
    assert "Recent repetition: 'EXAMINE CHALK STUB' was already used in this room." in user_prompt
    assert proposed == "EXAMINE CHALK STUB"
    assert client.responses.calls == 1


def test_openai_policy_warns_after_nonconsecutive_repeated_examine_in_same_room(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    client = _FakeClient("COMMAND: EXAMINE CHALK STUB")
    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=client)
    observation = (
        "Entry Vestibule. Exits lead north, south, west. "
        "You notice a snapped chalk stub lies near the doorway; a tarnished tapestry hangs crooked on iron hooks."
    )
    policy.record_turn_feedback(
        emitted_command="EXAMINE CHALK STUB",
        observation_text=observation,
        result_text="A snapped chalk stub worn smooth by handling.",
    )
    policy.record_turn_feedback(
        emitted_command="EXAMINE TAPESTRY",
        observation_text=observation,
        result_text="A tarnished tapestry crusted with mildew.",
    )
    _ = policy.propose_command(
        PolicyInput(
            observation_text=observation,
            use_exploration_assist=True,
            command_reference_text="N|S|E|W",
        )
    )

    assert client.responses.calls == 1
    user_prompt = client.responses.calls_kwargs[0]["input"][2]["content"]
    assert "You already examined these targets in this unchanged room" in user_prompt
    assert "EXAMINE CHALK STUB" in user_prompt
    assert "EXAMINE TAPESTRY" in user_prompt


def test_openai_policy_warns_on_revisiting_unchanged_room_with_prior_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    client = _FakeClient("COMMAND: EXAMINE PRAYER CHEST")
    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=client)
    observation = (
        "Moonwell Sanctum. Exits lead south, west. "
        "You notice an obsidian mirror rests in a cedar case; a prayer chest rests beneath the winch."
    )
    policy.record_turn_feedback(
        emitted_command="EXAMINE PRAYER CHEST",
        observation_text=observation,
        result_text="A cedar chest with iron corners and a narrow lockplate. It is locked.",
    )

    _ = policy.propose_command(
        PolicyInput(
            observation_text=observation,
            use_exploration_assist=True,
            command_reference_text="N|S|E|W",
        )
    )

    user_prompt = client.responses.calls_kwargs[0]["input"][2]["content"]
    assert "Room appears unchanged since your last visit." in user_prompt
    assert "EXAMINE PRAYER CHEST" in user_prompt


def test_openai_policy_sets_find_key_subgoal_when_lock_known_and_no_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    client = _FakeClient("COMMAND: E")
    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=client)
    policy.record_turn_feedback(
        emitted_command="OPEN PRAYER CHEST",
        observation_text="Sanctum. Exits lead north, east.",
        result_text="The prayer chest is locked.",
    )
    _ = policy.propose_command(
        PolicyInput(
            observation_text="Sanctum. Exits lead north, east.",
            command_reference_text="N|S|E|W",
        )
    )

    user_prompt = client.responses.calls_kwargs[0]["input"][2]["content"]
    assert "active_subgoal=find_key" in user_prompt


def test_openai_policy_avoids_blocked_room_command_on_revisit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    client = _FakeClient("COMMAND: SEARCH WELL")
    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=client)
    policy.record_turn_feedback(
        emitted_command="SEARCH WELL",
        observation_text=(
            "Well of Whispers. Exits lead north, east, west. "
            "You notice a dented bucket hangs over the dry shaft."
        ),
        result_text="You find nothing unusual about the well bucket.",
    )
    policy.record_turn_feedback(
        emitted_command="N",
        observation_text="Other Hall. Exits lead south.",
        result_text="You move north to Other Hall.",
    )
    proposed = policy.propose_command(
        PolicyInput(
            observation_text=(
                "Well of Whispers. Exits lead north, east, west. "
                "You notice a dented bucket hangs over the dry shaft."
            ),
            use_exploration_assist=True,
            command_reference_text="N|S|E|W",
        )
    )

    assert proposed == "SEARCH WELL"
    assert client.responses.calls == 1
    user_prompt = client.responses.calls_kwargs[0]["input"][2]["content"]
    assert "Avoid repeating known failed commands in this room: SEARCH WELL." in user_prompt


def test_openai_policy_avoids_reopening_known_locked_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    client = _FakeClient("COMMAND: OPEN PRAYER CHEST")
    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=client)
    policy.record_turn_feedback(
        emitted_command="OPEN PRAYER CHEST",
        observation_text="Sanctum. Exits lead north, east.",
        result_text="The prayer chest is locked.",
    )
    proposed = policy.propose_command(
        PolicyInput(
            observation_text="Sanctum. Exits lead north, east.",
            use_exploration_assist=True,
            command_reference_text="N|S|E|W",
        )
    )

    assert proposed == "E"
    assert client.responses.calls == 0


def test_openai_policy_warns_after_do_not_see_result_for_room_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    client = _FakeClient("COMMAND: EXAMINE COFFER")
    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=client)
    observation = (
        "Well of Whispers. Exits lead north, south. "
        "You notice a chain hook dangles from the pulley axle; a dented bucket hangs over the dry shaft."
    )
    policy.record_turn_feedback(
        emitted_command="EXAMINE COFFER",
        observation_text=observation,
        result_text="You do not see 'coffer' here.",
    )
    _ = policy.propose_command(
        PolicyInput(
            observation_text=observation,
            use_exploration_assist=True,
            command_reference_text="N|S|E|W",
        )
    )

    assert client.responses.calls == 1
    user_prompt = client.responses.calls_kwargs[0]["input"][2]["content"]
    assert "Avoid repeating known failed commands in this room: EXAMINE COFFER." in user_prompt


def test_openai_policy_rewrites_search_direction_to_navigation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    client = _FakeClient("COMMAND: SEARCH SOUTH")
    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=client)
    proposed = policy.propose_command(
        PolicyInput(
            observation_text=(
                "Vault Antechamber. Exits lead south, west. "
                "You notice an ironbound coffer sits on the dais."
            ),
            use_exploration_assist=True,
            command_reference_text="N|S|E|W",
        )
    )

    assert proposed == "S"
    assert client.responses.calls == 1


def test_openai_policy_blocks_take_target_not_present_here(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    client = _FakeClient("COMMAND: TAKE MASON'S HAMMER")
    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=client)
    proposed = policy.propose_command(
        PolicyInput(
            observation_text=(
                "Vault Antechamber. Exits lead south, west. "
                "You notice an ironbound coffer sits on the dais."
            ),
            use_exploration_assist=True,
            command_reference_text="N|S|E|W",
        )
    )

    assert proposed == "S"
    assert client.responses.calls == 1


def test_openai_policy_avoids_retaking_known_non_portable_item(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")

    client = _FakeClient("COMMAND: TAKE DENTED BUCKET")
    policy = OpenAIAgentPolicy(model_name="gpt-5-mini", client=client)
    policy.record_turn_feedback(
        emitted_command="TAKE DENTED BUCKET",
        observation_text="Well of Whispers. Exits lead north, east.",
        result_text="item 'dented_bucket' is not portable",
    )
    proposed = policy.propose_command(
        PolicyInput(
            observation_text=(
                "Well of Whispers. Exits lead north, east. "
                "You notice a dented bucket hangs over the dry shaft."
            ),
            use_exploration_assist=True,
            command_reference_text="N|S|E|W",
        )
    )

    assert proposed == "TAKE DENTED BUCKET"
    assert client.responses.calls == 1
    user_prompt = client.responses.calls_kwargs[0]["input"][2]["content"]
    assert "Avoid repeating known failed commands in this room: TAKE DENTED BUCKET." in user_prompt
