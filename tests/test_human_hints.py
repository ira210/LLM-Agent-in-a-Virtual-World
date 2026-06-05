from dungeon_agent.runner.hint_stream import (
    deserialize_hint_events_jsonl,
    serialize_hint_events_jsonl,
)
from dungeon_agent.runner.human_hints import (
    FixedIntervalCheckpointPolicy,
    HumanHintIngestionService,
)
from dungeon_agent.schemas import HumanHintEvent, HumanHintEventStream


def test_hints_are_accepted_only_at_checkpoints() -> None:
    service = HumanHintIngestionService(
        mode="agent+human",
        checkpoint_policy=FixedIntervalCheckpointPolicy(
            turn_interval=2,
            first_checkpoint_turn=0,
            eligible_room_ids=frozenset({"foyer"}),
        ),
    )

    accepted = service.ingest_hint(
        run_id="run-1", turn_index=0, room_id="foyer", hint_text="find light first"
    )
    rejected = service.ingest_hint(
        run_id="run-1", turn_index=1, room_id="foyer", hint_text="go east"
    )
    room_rejected = service.ingest_hint(
        run_id="run-1", turn_index=2, room_id="cellar", hint_text="go east"
    )

    assert accepted.accepted is True
    assert accepted.event is not None
    assert accepted.event.turn_index == 0
    assert rejected.accepted is False
    assert rejected.event is None
    assert room_rejected.accepted is False


def test_hint_stream_jsonl_serialization_round_trip() -> None:
    stream = HumanHintEventStream(
        run_id="run-1",
        events=[
            HumanHintEvent(run_id="run-1", turn_index=0, hint_text="Find a light."),
            HumanHintEvent(run_id="run-1", turn_index=2, hint_text="Return to exit."),
        ],
    )

    payload = serialize_hint_events_jsonl(stream.events)
    restored_events = deserialize_hint_events_jsonl(payload)

    assert restored_events == stream.events


def test_no_hint_path_attaches_clean_empty_context() -> None:
    service = HumanHintIngestionService(
        mode="agent+human",
        checkpoint_policy=FixedIntervalCheckpointPolicy(turn_interval=1, first_checkpoint_turn=0),
    )
    no_hint = service.ingest_hint(run_id="run-1", turn_index=0, room_id="foyer", hint_text=None)
    context = service.build_decision_context(
        observation_text="You are in a room.", hint_result=no_hint
    )
    policy_input = context.to_policy_input()

    assert no_hint.accepted is False
    assert no_hint.event is None
    assert policy_input.human_input_text is None
