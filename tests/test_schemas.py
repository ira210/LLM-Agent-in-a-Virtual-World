from dungeon_agent.schemas import (
    ActiveSubgoal,
    HumanHintEvent,
    HumanHintEventStream,
    ReplayArtifact,
    RunMetadata,
    StateFlags,
    TurnRecord,
    ValidatorAction,
)


def test_turn_record_round_trip() -> None:
    record = TurnRecord(
        run_id="run-1",
        seed=1,
        turn_index=0,
        observation_text="obs",
        human_input_text=None,
        proposed_command="go nort",
        validated_command="N",
        agent_command="N",
        result_text="You move north.",
        active_subgoal=ActiveSubgoal.FIND_LIGHT,
        loop_recovery_triggered=False,
        validator_action=ValidatorAction.REWRITTEN,
        state_flags=StateFlags(),
        terminal=False,
    )
    payload = record.model_dump(mode="json")
    restored = TurnRecord.model_validate(payload)
    assert restored.validated_command == "N"


def test_replay_artifact_shape() -> None:
    replay = ReplayArtifact(
        metadata=RunMetadata(
            run_id="run-1",
            seed=1,
            max_turns=10,
            model_provider="openai",
            model_name="gpt-5-mini",
        ),
        turns=[],
        hints=[HumanHintEvent(run_id="run-1", turn_index=0, hint_text="Find a light")],
    )
    assert replay.hints[0].event_type.value == "human_hint"


def test_human_hint_event_stream_shape() -> None:
    stream = HumanHintEventStream(
        run_id="run-1",
        events=[HumanHintEvent(run_id="run-1", turn_index=1, hint_text="Search lit rooms only.")],
    )
    assert stream.events[0].event_type.value == "human_hint"
